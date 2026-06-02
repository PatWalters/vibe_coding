#!/usr/bin/env python
"""
Shape + pharmacophore screening with RDKit Open3DAlign (O3A).

Takes a single reference molecule and a library, generates a conformer
ensemble for each library molecule, aligns every conformer onto the
reference with Crippen-based O3A (pharmacophore-like atom typing), keeps
the best-scoring pose per molecule, and writes an SDF. Output is in input
order by default; pass --sort to rank by O3A score.

Each output molecule carries SD properties:
    O3A_Score          - O3A alignment score (higher = better overlap)
    O3A_RMSD           - RMSD of the matched atoms after alignment
    Shape_Tanimoto     - 1 - ShapeTanimotoDist (1.0 = identical shape)
    Best_ConfId        - which generated conformer won

With --enumerate-stereo, centers whose stereochemistry is UNDEFINED are
expanded into all stereoisomers (defined centers are preserved); each
isomer is aligned and only the best-fitting one is written, with extra
tags N_Undefined_Stereo, N_Stereoisomers, and Chosen_Isomer_SMILES.

Usage:
    python o3a_screen.py reference.sdf library.sdf out.sdf [--nconfs 20]

The reference may be an SDF (first molecule, conformer used as-is) or a
SMILES file (a conformer is generated for it).

The library may be an SDF, a whitespace-delimited SMILES file, or a CSV.
For a CSV, columns default to SMILES / Name / pEC50 but can be overridden
with --smiles-col / --name-col / --activity-col. The Name value is written
to both the SDF title line and a same-named SD tag, and the activity value
is written as an SD tag.
"""
from __future__ import annotations

import argparse
import csv
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem, rdMolDescriptors, rdShapeHelpers, rdFMCS, rdMolAlign,
)
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers, StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # silence embedding/parse chatter


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_reference(path: str) -> Chem.Mol:
    """Load the reference. If it has no 3D conformer, embed one."""
    mol = None
    if path.lower().endswith((".sdf", ".mol")):
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        for m in suppl:
            if m is not None:
                mol = m
                break
    else:  # treat as SMILES (one per line; first valid one used)
        with open(path) as fh:
            for line in fh:
                smi = line.split()[0].strip() if line.strip() else ""
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        break

    if mol is None:
        sys.exit(f"[error] could not load a reference molecule from {path}")

    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            sys.exit("[error] failed to embed the reference molecule")
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def iter_library(path: str, smiles_col: str = "SMILES",
                 name_col: str = "Name", activity_col: str = "pEC50"):
    """Yield (index, mol) for each parseable molecule in the library.

    Supports SDF, whitespace-delimited SMILES files, and CSV. For a CSV the
    `smiles_col`/`name_col`/`activity_col` arguments select the relevant
    columns; the name and activity values are stashed on the molecule as
    SD properties (keyed by the column names) so the caller can propagate
    them to the aligned output.
    """
    lower = path.lower()
    if lower.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        for i, m in enumerate(suppl):
            yield i, m
    elif lower.endswith(".csv"):
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            if smiles_col not in fields:
                sys.exit(f"[error] CSV {path} has no '{smiles_col}' column "
                         f"(columns: {', '.join(fields)})")
            has_name = name_col in fields
            has_activity = activity_col in fields
            if not has_name:
                print(f"[warn] CSV has no '{name_col}' column; molecules "
                      f"will be auto-named", file=sys.stderr)
            if not has_activity:
                print(f"[warn] CSV has no '{activity_col}' column; no "
                      f"activity tag will be written", file=sys.stderr)
            for i, row in enumerate(reader):
                smi = (row.get(smiles_col) or "").strip()
                if not smi:
                    yield i, None
                    continue
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    if has_name:
                        name = (row.get(name_col) or "").strip()
                        if name:
                            m.SetProp("_Name", name)
                            m.SetProp(name_col, name)
                    if has_activity:
                        act = (row.get(activity_col) or "").strip()
                        if act:
                            m.SetProp(activity_col, act)
                yield i, m
    else:  # SMILES file: "SMILES [name]" per line
        with open(path) as fh:
            for i, line in enumerate(fh):
                parts = line.split()
                if not parts:
                    yield i, None
                    continue
                m = Chem.MolFromSmiles(parts[0])
                if m is not None and len(parts) > 1:
                    m.SetProp("_Name", parts[1])
                yield i, m


# ----------------------------------------------------------------------
# Core alignment
# ----------------------------------------------------------------------
def embed_ensemble(mol: Chem.Mol, n_confs: int, seed: int = 1) -> Chem.Mol:
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5  # drop near-duplicate conformers
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not cids:  # embedding can fail for awkward structures; try once more loosely
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    # light optimization improves O3A on strained embeds
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    return mol


def embed_ensemble_constrained(mol: Chem.Mol, ref: Chem.Mol,
                               amap, n_confs: int, seed: int = 1,
                               force_constant: float = 500.0) -> Chem.Mol:
    """Embed conformers with the MCS core pinned to the reference geometry.

    `amap` is [(prbIdx, refIdx), ...] on the heavy-atom (no-H) molecules.
    The shared core is built directly onto the reference's coordinates via a
    coordMap, so flexible shared rings (e.g. piperazine) adopt the reference's
    pucker instead of an independent one. A restrained MMFF minimization then
    relaxes the non-core atoms while holding the core in place.
    """
    molH = Chem.AddHs(mol)
    # map heavy-atom probe indices -> reference coordinates
    ref_conf = ref.GetConformer()
    coord_map = {p: ref_conf.GetAtomPosition(r) for p, r in amap}
    core_prb_idx = [p for p, _ in amap]

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.5

    cids = []
    for i in range(n_confs):
        m_try = Chem.AddHs(Chem.Mol(mol))
        if AllChem.EmbedMolecule(m_try, coordMap=coord_map,
                                 randomSeed=seed + i, maxAttempts=50) != 0:
            continue
        # restrained minimization: pin core heavy atoms with a stiff harmonic
        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                m_try, AllChem.MMFFGetMoleculeProperties(m_try))
            if ff is not None:
                for idx in core_prb_idx:
                    ff.MMFFAddPositionConstraint(idx, 0.0, force_constant)
                ff.Minimize(maxIts=300)
        except Exception:
            pass
        conf = m_try.GetConformer()
        new_id = molH.AddConformer(conf, assignId=True)
        cids.append(new_id)

    if not cids:  # fall back to unconstrained if pinned embedding failed entirely
        return embed_ensemble(mol, n_confs, seed)
    return molH


def best_alignment(prb: Chem.Mol, ref: Chem.Mol, crippen_ref, n_confs: int):
    """O3A-align every conformer of prb onto ref; return the best (score, rmsd, cid)."""
    crippen_prb = rdMolDescriptors._CalcCrippenContribs(prb)
    best = (-1.0, None, -1)  # score, rmsd, cid
    for cid in range(prb.GetNumConformers()):
        try:
            o3a = AllChem.GetCrippenO3A(
                prb, ref, crippen_prb, crippen_ref, prbCid=cid, refCid=0
            )
            rmsd = o3a.Align()
            score = o3a.Score()
        except Exception:
            continue
        if score > best[0]:
            best = (score, rmsd, cid)
    return best


def shape_tanimoto(prb: Chem.Mol, ref: Chem.Mol, cid: int) -> float:
    try:
        dist = rdShapeHelpers.ShapeTanimotoDist(prb, ref, confId1=cid, confId2=0)
        return 1.0 - dist
    except Exception:
        return float("nan")


_LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def strip_salts(mol: Chem.Mol) -> Chem.Mol:
    """Remove disconnected fragments (e.g. HCl, Na salts), keeping the parent.

    Returns the largest organic fragment; if the molecule is a single
    fragment it is returned unchanged. SD properties and the title are
    preserved.
    """
    if "." not in Chem.MolToSmiles(mol):
        return mol
    try:
        stripped = _LARGEST_FRAGMENT_CHOOSER.choose(mol)
    except Exception:
        return mol
    if stripped is None:
        return mol
    # LargestFragmentChooser does not copy SD props / title; carry them over
    for prop in mol.GetPropNames(includePrivate=True):
        stripped.SetProp(prop, mol.GetProp(prop))
    return stripped


def count_undefined_stereo(mol: Chem.Mol) -> int:
    """Number of unassigned (potential) stereocenters, incl. double bonds."""
    si = Chem.FindPotentialStereo(mol)
    return sum(1 for e in si if str(e.specified) == "Unspecified")


def stereo_variants(mol: Chem.Mol, max_isomers: int = 32):
    """Yield stereoisomers expanding only UNDEFINED centers.

    Defined stereochemistry is preserved. A molecule with no undefined
    centers yields itself unchanged.
    """
    opts = StereoEnumerationOptions(
        onlyUnassigned=True,   # leave already-defined centers alone
        unique=True,
        maxIsomers=max_isomers,
    )
    produced = False
    for iso in EnumerateStereoisomers(mol, options=opts):
        produced = True
        yield iso
    if not produced:           # safety net: always yield at least the input
        yield mol


def find_mcs_atom_maps(prb: Chem.Mol, ref: Chem.Mol, min_atoms: int = 6):
    """Return a list of atomMap [(prbIdx, refIdx), ...] candidates from the MCS.

    Uses a relaxed MCS (ringMatchesRingOnly but NOT completeRingsOnly) so a
    shared core can extend across rings that differ only in substitution
    pattern (e.g. an extra methyl or shifted ring heteroatom) -- this is what
    forces e.g. isopropyl-to-isopropyl correspondence instead of letting the
    overlap term map it onto the wrong group.

    Returns [] if no MCS of at least min_atoms is found. Multiple candidates
    are returned to account for topological symmetry (e.g. a phenyl that can
    match two ways); the caller should try each and keep the best fit.
    """
    res = rdFMCS.FindMCS(
        [prb, ref],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=False,
        timeout=30,
    )
    if res.canceled or res.numAtoms < min_atoms:
        return []
    patt = Chem.MolFromSmarts(res.smartsString)
    if patt is None:
        return []
    prb_matches = prb.GetSubstructMatches(patt, uniquify=False)
    ref_matches = ref.GetSubstructMatches(patt, uniquify=False)
    if not prb_matches or not ref_matches:
        return []
    # Pair each probe match against the (single) reference match. Trying every
    # probe symmetry variant against the first ref match covers the symmetry
    # we care about while keeping the candidate count small.
    ref_match = ref_matches[0]
    maps = []
    for pm in prb_matches:
        maps.append(list(zip(pm, ref_match)))
    return maps


def mcs_align_structure(mol: Chem.Mol, ref: Chem.Mol, crippen_ref,
                        n_confs: int, keep_hs: bool, min_mcs_atoms: int,
                        constrain_core: bool = False):
    """Align by overlaying the maximum common substructure.

    Embeds an ensemble, and for each conformer aligns onto the reference using
    the MCS atom mapping (rdMolAlign.AlignMol with a fixed atomMap), trying all
    symmetric mappings. Keeps the lowest-RMSD pose. Returns the same contract
    as align_structure: (score, out_mol). Here 'score' is -RMSD so that the
    'higher is better' selection logic in the caller still works, and the O3A
    score is additionally computed and stored for reference.

    If constrain_core is True, conformers are generated with the MCS core
    pinned to the reference geometry, so shared flexible rings adopt the
    reference's conformation rather than an independent one.

    Returns (None, None) if no usable MCS exists (caller may fall back to O3A).
    """
    atom_maps = find_mcs_atom_maps(mol, ref, min_atoms=min_mcs_atoms)
    if not atom_maps:
        return None, None

    if constrain_core:
        # use the first (any) mapping to pin the core during embedding;
        # AlignMol below still tries every symmetric mapping for final fit
        prb = embed_ensemble_constrained(mol, ref, atom_maps[0], n_confs)
    else:
        prb = embed_ensemble(mol, n_confs)
    if prb.GetNumConformers() == 0:
        return None, None

    best = None  # (rmsd, cid, used_map)
    for cid in range(prb.GetNumConformers()):
        for amap in atom_maps:
            try:
                rmsd = rdMolAlign.AlignMol(prb, ref, prbCid=cid, refCid=0,
                                           atomMap=amap)
            except Exception:
                continue
            if best is None or rmsd < best[0]:
                best = (rmsd, cid, amap)
    if best is None:
        return None, None

    rmsd, cid, used_map = best
    shp = shape_tanimoto(prb, ref, cid)

    # also report an O3A score on the chosen pose for continuity of metrics
    try:
        crippen_prb = rdMolDescriptors._CalcCrippenContribs(prb)
        o3a = AllChem.GetCrippenO3A(prb, ref, crippen_prb, crippen_ref,
                                    prbCid=cid, refCid=0)
        o3a_score = o3a.Score()
    except Exception:
        o3a_score = float("nan")

    out = Chem.Mol(prb)
    for c in list(range(out.GetNumConformers())):
        if c != cid:
            out.RemoveConformer(c)
    if not keep_hs:
        out = Chem.RemoveHs(out)
    out.SetProp("Align_Method", "MCS+constrained" if constrain_core else "MCS")
    out.SetProp("MCS_RMSD", f"{rmsd:.4f}")
    out.SetProp("MCS_NumAtoms", str(len(used_map)))
    out.SetProp("O3A_Score", f"{o3a_score:.4f}")
    out.SetProp("Shape_Tanimoto", f"{shp:.4f}")
    out.SetProp("Best_ConfId", str(cid))
    # selection key: smaller RMSD is better -> negate so caller's max() works
    return -rmsd, out


def align_structure(mol: Chem.Mol, ref: Chem.Mol, crippen_ref,
                    n_confs: int, keep_hs: bool):
    """Embed an ensemble, O3A-align all conformers, return the best pose.

    Returns (score, out_mol) or (None, None) on failure.
    """
    prb = embed_ensemble(mol, n_confs)
    if prb.GetNumConformers() == 0:
        return None, None
    score, rmsd, cid = best_alignment(prb, ref, crippen_ref, n_confs)
    if cid < 0:
        return None, None
    shp = shape_tanimoto(prb, ref, cid)

    out = Chem.Mol(prb)
    for c in list(range(out.GetNumConformers())):
        if c != cid:
            out.RemoveConformer(c)
    if not keep_hs:
        out = Chem.RemoveHs(out)
    out.SetProp("Align_Method", "O3A")
    out.SetProp("O3A_Score", f"{score:.4f}")
    out.SetProp("O3A_RMSD", f"{rmsd:.4f}")
    out.SetProp("Shape_Tanimoto", f"{shp:.4f}")
    out.SetProp("Best_ConfId", str(cid))
    return score, out


def align_dispatch(mol, ref, crippen_ref, n_confs, keep_hs,
                   use_mcs, mcs_fallback, min_mcs_atoms, constrain_core=False):
    """Choose MCS or O3A alignment based on flags, with optional fallback."""
    if use_mcs:
        score, out = mcs_align_structure(
            mol, ref, crippen_ref, n_confs, keep_hs, min_mcs_atoms,
            constrain_core=constrain_core,
        )
        if out is not None:
            return score, out
        if not mcs_fallback:
            return None, None
        # fall through to O3A when MCS is too small / absent
    return align_structure(mol, ref, crippen_ref, n_confs, keep_hs)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="reference SDF/MOL or SMILES file")
    ap.add_argument("library", help="library SDF, SMILES, or CSV file")
    ap.add_argument("output", help="output SDF (sorted by O3A score)")
    ap.add_argument("--nconfs", type=int, default=20,
                    help="conformers generated per library molecule (default 20)")
    ap.add_argument("--smiles-col", default="SMILES",
                    help="CSV column holding SMILES (default 'SMILES')")
    ap.add_argument("--name-col", default="Name",
                    help="CSV column holding the molecule name (default 'Name')")
    ap.add_argument("--activity-col", default="pEC50",
                    help="CSV column holding the activity value (default 'pEC50')")
    ap.add_argument("--keep-hs", action="store_true",
                    help="keep explicit hydrogens in the output (default: strip)")
    ap.add_argument("--sort", action="store_true",
                    help="sort output by O3A score (default: preserve input order)")
    ap.add_argument("--enumerate-stereo", action="store_true",
                    help="for centers with UNDEFINED stereochemistry, enumerate "
                         "stereoisomers and output the best-fitting one")
    ap.add_argument("--max-isomers", type=int, default=32,
                    help="cap on stereoisomers enumerated per molecule (default 32)")
    ap.add_argument("--mcs-align", action="store_true",
                    help="align by overlaying the maximum common substructure "
                         "(forces correct group correspondence on shared scaffolds) "
                         "instead of O3A's own atom matching")
    ap.add_argument("--no-mcs-fallback", action="store_true",
                    help="with --mcs-align, skip (rather than O3A-align) molecules "
                         "that share no sufficient common substructure with the reference")
    ap.add_argument("--min-mcs-atoms", type=int, default=6,
                    help="minimum MCS size (atoms) to accept an MCS alignment (default 6)")
    ap.add_argument("--constrain-core", action="store_true",
                    help="with --mcs-align, generate conformers with the shared core "
                         "pinned to the reference geometry so flexible shared rings "
                         "(e.g. piperazine) adopt the reference's conformation")
    args = ap.parse_args()

    ref = load_reference(args.reference)
    crippen_ref = rdMolDescriptors._CalcCrippenContribs(ref)
    print(f"[info] reference loaded: {ref.GetProp('_Name') if ref.HasProp('_Name') else '(unnamed)'}")

    results = []  # (score, mol_single_conf)
    n_in = n_ok = n_fail = 0
    for idx, mol in iter_library(args.library, args.smiles_col,
                                 args.name_col, args.activity_col):
        n_in += 1
        if mol is None:
            n_fail += 1
            print(f"[warn] molecule {idx}: unparseable, skipping", file=sys.stderr)
            continue

        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"

        # drop stray disconnected fragments (HCl/Na salts, solvents) so they
        # don't propagate into the aligned, written-out structure
        mol = strip_salts(mol)

        # SD tags to carry through from the input (e.g. CSV name/activity);
        # the aligned output is rebuilt from scratch, so they must be re-applied.
        carry_tags = {k: mol.GetProp(k) for k in (args.name_col, args.activity_col)
                      if mol.HasProp(k)}

        # Decide which structures to evaluate for this entry
        if args.enumerate_stereo:
            n_undef = count_undefined_stereo(mol)
            variants = list(stereo_variants(mol, args.max_isomers))
        else:
            n_undef = 0
            variants = [mol]

        # Align each variant; keep the best-scoring one
        best_score, best_out, best_smi = None, None, None
        for variant in variants:
            score, out = align_dispatch(
                variant, ref, crippen_ref, args.nconfs, args.keep_hs,
                use_mcs=args.mcs_align,
                mcs_fallback=not args.no_mcs_fallback,
                min_mcs_atoms=args.min_mcs_atoms,
                constrain_core=args.constrain_core,
            )
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score, best_out = score, out
                best_smi = Chem.MolToSmiles(variant)

        if best_out is None:
            n_fail += 1
            print(f"[warn] {name}: embedding/alignment failed for all "
                  f"variants, skipping", file=sys.stderr)
            continue

        best_out.SetProp("_Name", name)
        for k, v in carry_tags.items():
            best_out.SetProp(k, v)
        if args.enumerate_stereo:
            best_out.SetProp("N_Undefined_Stereo", str(n_undef))
            best_out.SetProp("N_Stereoisomers", str(len(variants)))
            best_out.SetProp("Chosen_Isomer_SMILES", best_smi)

        results.append((best_score, best_out))
        n_ok += 1
        extra = (f"  isomers={len(variants):2d} (undef={n_undef})"
                 if args.enumerate_stereo else "")
        method = best_out.GetProp("Align_Method")
        shp = float(best_out.GetProp("Shape_Tanimoto"))
        if method.startswith("MCS"):
            rmsd = float(best_out.GetProp("MCS_RMSD"))
            o3a = best_out.GetProp("O3A_Score")
            tag = "MCS*" if "constrained" in method else "MCS "
            print(f"[ok] {name:24s} {tag} rmsd={rmsd:6.3f}  "
                  f"O3A={o3a:>8s}  shapeT={shp:5.3f}{extra}")
        else:
            rmsd = float(best_out.GetProp("O3A_RMSD"))
            print(f"[ok] {name:24s} O3A  score={best_score:8.3f}  "
                  f"rmsd={rmsd:6.3f}  shapeT={shp:5.3f}{extra}")

    # Output order: input order by default, score-sorted only if requested.
    # In MCS mode the selection key is -RMSD, so 'best first' still holds.
    if args.sort:
        results.sort(key=lambda t: t[0], reverse=True)
    writer = Chem.SDWriter(args.output)
    for _, m in results:
        writer.write(m)
    writer.close()

    if args.sort:
        order = "sorted best-fit first (by MCS RMSD)" if args.mcs_align \
                else "sorted by O3A_Score"
    else:
        order = "in input order"
    print(f"\n[done] read {n_in}, aligned {n_ok}, skipped {n_fail}")
    print(f"[done] wrote {n_ok} molecules to {args.output} ({order})")


if __name__ == "__main__":
    main()
