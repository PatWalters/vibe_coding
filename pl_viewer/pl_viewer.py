#!/usr/bin/env python3
"""
Build a self-contained HTML viewer for a series of protein-ligand complexes.

Left  : the protein structure rendered with Mol*, zoomed on the binding site.
Right : a table of ligands -- 2D structure (aligned on the maximum common
        substructure), name, and pEC50 -- with a checkbox per row that toggles
        that ligand's 3D pose in the Mol* viewer.

The CSV and SDF are linked through the molecule name ("Name" column in the CSV,
title line of each SDF record). The ligand 3D coordinates are assumed to live in
the same frame as the protein (the SDF is "aligned"/docked into this structure),
so the binding site is taken as the centroid of the ligand atoms.

Usage:
    python pl_viewer.py \
        --csv PXR_series.csv \
        --sdf pxr_aligned.sdf \
        --pdb x03363-1_chainB_protein.pdb \
        --out pl_viewer.html
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

# Distinct, color-blind-friendly carbon colors cycled across checked ligands.
LIGAND_COLORS = [
    (0.90, 0.29, 0.21),  # red
    (0.20, 0.49, 0.96),  # blue
    (0.18, 0.65, 0.34),  # green
    (0.86, 0.51, 0.11),  # orange
    (0.61, 0.35, 0.71),  # purple
    (0.09, 0.63, 0.69),  # teal
    (0.85, 0.36, 0.62),  # pink
    (0.50, 0.55, 0.16),  # olive
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="PXR_series.csv")
    p.add_argument("--sdf", default="pxr_aligned.sdf")
    p.add_argument("--pdb", default="x03363-1_chainB_protein.pdb")
    p.add_argument("--out", default="pl_viewer.html")
    p.add_argument("--name-col", default="Name")
    p.add_argument("--activity-col", default="pEC50")
    p.add_argument("--molstar-version", default="4.9.0",
                   help="Mol* version pulled from the jsDelivr CDN.")
    p.add_argument("--pocket-radius", type=float, default=14.0,
                   help="Camera focus radius (A) around the ligand centroid.")
    p.add_argument("--pocket-cutoff", type=float, default=5.0,
                   help="Residues with any atom within this distance (A) of any "
                        "ligand atom form the wireframe binding site.")
    p.add_argument("--hbond-max", type=float, default=3.6,
                   help="Max heavy-atom donor..acceptor distance (A) for an H-bond.")
    p.add_argument("--hbond-min", type=float, default=2.4,
                   help="Min heavy-atom donor..acceptor distance (A) for an H-bond.")
    return p.parse_args()


def load_sdf(path):
    """name -> 3D RDKit mol (sanitized)."""
    mols = {}
    supplier = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if not name:
            continue
        mols[name] = mol
    return mols


def aligned_2d_depictions(mols_by_name, ordered_names):
    """Generate 2D coords for every ligand aligned on the shared MCS core.

    Returns (depictions_by_name, mcs_match_by_name) where each depiction is a
    fresh 2D RDKit mol and the match is the tuple of atom indices in the MCS.
    """
    mols = [mols_by_name[n] for n in ordered_names]

    rdDepictor.SetPreferCoordGen(True)

    template = None
    if len(mols) >= 2:
        mcs = rdFMCS.FindMCS(
            mols,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=60,
        )
        if mcs.numAtoms > 0 and not mcs.canceled:
            template = Chem.MolFromSmarts(mcs.smartsString)
            try:
                rdDepictor.Compute2DCoords(template)
            except Exception:
                template = None

    depictions = {}
    matches = {}
    for name in ordered_names:
        m = Chem.Mol(mols_by_name[name])
        m.RemoveAllConformers()
        match = ()
        if template is not None:
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(m, template)
                match = m.GetSubstructMatch(template)
            except Exception:
                rdDepictor.Compute2DCoords(m)
        else:
            rdDepictor.Compute2DCoords(m)
        depictions[name] = m
        matches[name] = match
    return depictions, matches


def render_svg(mol, highlight_atoms, width=300, height=220):
    """Render a 2D mol to a transparent inline SVG, highlighting the MCS core."""
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1
    highlight_bonds = []
    hl_atom_set = set(highlight_atoms)
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in hl_atom_set and bond.GetEndAtomIdx() in hl_atom_set:
            highlight_bonds.append(bond.GetIdx())
    core_color = (0.80, 0.90, 1.0)  # pale blue wash over the shared scaffold
    atom_colors = {a: core_color for a in highlight_atoms}
    bond_colors = {b: core_color for b in highlight_bonds}
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def ligand_centroid(mols):
    """Mean position over all atoms of all ligands -> binding-site center."""
    n = 0
    sx = sy = sz = 0.0
    for mol in mols:
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            sx += p.x
            sy += p.y
            sz += p.z
            n += 1
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [sx / n, sy / n, sz / n]


# Side-chain heavy atoms that can H-bond, by (residue, atom name).
# 'D' = donor only, 'A' = acceptor only, 'B' = both (depending on protonation).
PROT_HBOND_SIDECHAIN = {
    ("ARG", "NE"): "D", ("ARG", "NH1"): "D", ("ARG", "NH2"): "D",
    ("ASN", "ND2"): "D", ("ASN", "OD1"): "A",
    ("GLN", "NE2"): "D", ("GLN", "OE1"): "A",
    ("ASP", "OD1"): "A", ("ASP", "OD2"): "A",
    ("GLU", "OE1"): "A", ("GLU", "OE2"): "A",
    ("HIS", "ND1"): "B", ("HIS", "NE2"): "B",
    ("LYS", "NZ"): "D",
    ("SER", "OG"): "B",
    ("THR", "OG1"): "B",
    ("TYR", "OH"): "B",
    ("TRP", "NE1"): "D",
    ("CYS", "SG"): "B",
    ("MET", "SD"): "A",
}


def prot_hbond_role(resname, atomname):
    """(is_donor, is_acceptor) for a protein heavy atom from atom/residue names,
    so explicit hydrogens are not required (crystal PDBs usually lack them)."""
    resname = resname.strip().upper()
    atomname = atomname.strip().upper()
    # Backbone amide N donates (proline has no amide H); backbone carbonyl O
    # (and the terminal OXT carboxylate) accept.
    if atomname == "N":
        return (resname != "PRO", False)
    if atomname in ("O", "OXT"):
        return (False, True)
    role = PROT_HBOND_SIDECHAIN.get((resname, atomname))
    if role is None:
        return (False, False)
    return (role in ("D", "B"), role in ("A", "B"))


def parse_pdb_atoms(pdb_text):
    """Parse ATOM/HETATM records into per-atom arrays + per-residue grouping.

    Returns dict with: coords (N,3 float), elements (list), line (list of raw
    lines), reskey (list of (chain, resSeq, iCode) per atom), and boolean
    donor/acceptor arrays marking H-bond-capable heavy atoms.
    """
    coords, elements, lines, reskey = [], [], [], []
    donor, acceptor = [], []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        elem = line[76:78].strip().upper()
        if not elem:
            elem = line[12:16].strip()[:1].upper()
        d, a = prot_hbond_role(line[17:20], line[12:16])
        coords.append((x, y, z))
        elements.append(elem)
        lines.append(line.rstrip("\n"))
        reskey.append((line[21], line[22:26], line[26]))
        donor.append(d)
        acceptor.append(a)
    return {
        "coords": np.asarray(coords, dtype=float) if coords else np.empty((0, 3)),
        "elements": elements,
        "lines": lines,
        "reskey": reskey,
        "donor": np.asarray(donor, dtype=bool),
        "acceptor": np.asarray(acceptor, dtype=bool),
    }


def pocket_pdb(prot, lig_coords, cutoff):
    """PDB text with only the residues lining the binding site (whole residues
    that have any atom within `cutoff` of any ligand atom)."""
    if prot["coords"].shape[0] == 0 or lig_coords.shape[0] == 0:
        return "\n".join(prot["lines"]) + "\nEND\n"
    # min distance from each protein atom to the ligand atom cloud
    diff = prot["coords"][:, None, :] - lig_coords[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    near = d2.min(axis=1) <= cutoff * cutoff
    keep_res = {prot["reskey"][i] for i in np.nonzero(near)[0]}
    kept = [prot["lines"][i] for i in range(len(prot["lines"]))
            if prot["reskey"][i] in keep_res]
    return "\n".join(kept) + "\nEND\n"


# RDKit chemical-feature factory: reliable donor/acceptor perception on the
# ligand (the SDF is loaded with explicit hydrogens, so donors are unambiguous).
_FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)


def ligand_hbond_atoms(mol):
    """(donor_atom_idx_set, acceptor_atom_idx_set) for the ligand."""
    donors, acceptors = set(), set()
    for feat in _FEATURE_FACTORY.GetFeaturesForMol(mol):
        fam = feat.GetFamily()
        if fam == "Donor":
            donors.update(feat.GetAtomIds())
        elif fam == "Acceptor":
            acceptors.update(feat.GetAtomIds())
    return donors, acceptors


def hbond_molblock(mol, prot, dmin, dmax, max_bonds=8):
    """Build a tiny MDL molblock of dashed-line segments for putative H-bonds
    between this ligand and the protein.

    A contact is only drawn when the two atoms form a real donor..acceptor pair
    (ligand donor with protein acceptor, or ligand acceptor with protein donor),
    so two acceptors sitting near each other are no longer mistaken for an H-bond.

    Returns (molblock_or_None, count).
    """
    if prot["coords"].shape[0] == 0:
        return None, 0
    conf = mol.GetConformer()

    lig_donors, lig_acceptors = ligand_hbond_atoms(mol)
    if not lig_donors and not lig_acceptors:
        return None, 0

    polar_idx = np.nonzero(prot["donor"] | prot["acceptor"])[0]
    if polar_idx.size == 0:
        return None, 0
    ppos = prot["coords"][polar_idx]
    p_donor = prot["donor"][polar_idx]
    p_acceptor = prot["acceptor"][polar_idx]

    pairs = []
    for i in range(mol.GetNumAtoms()):
        l_donor = i in lig_donors
        l_acceptor = i in lig_acceptors
        if not (l_donor or l_acceptor):
            continue
        p = conf.GetAtomPosition(i)
        lpos = np.array([p.x, p.y, p.z])
        d = np.linalg.norm(ppos - lpos, axis=1)
        # keep only true donor..acceptor pairings within the distance window
        paired = (l_donor & p_acceptor) | (l_acceptor & p_donor)
        keep = (d >= dmin) & (d <= dmax) & paired
        for j in np.nonzero(keep)[0]:
            pairs.append((d[j], lpos, ppos[j]))

    if not pairs:
        return None, 0
    pairs.sort(key=lambda t: t[0])
    pairs = pairs[:max_bonds]

    natoms = len(pairs) * 2
    nbonds = len(pairs)
    out = ["", "     hbonds", "",
           "%3d%3d  0  0  0  0  0  0  0  0999 V2000" % (natoms, nbonds)]
    for _, a, b in pairs:
        out.append("%10.4f%10.4f%10.4f O   0  0  0  0  0  0  0  0  0  0  0  0"
                    % (a[0], a[1], a[2]))
        out.append("%10.4f%10.4f%10.4f N   0  0  0  0  0  0  0  0  0  0  0  0"
                    % (b[0], b[1], b[2]))
    for k in range(len(pairs)):
        out.append("%3d%3d  1  0" % (2 * k + 1, 2 * k + 2))
    out.append("M  END")
    return "\n".join(out) + "\n", len(pairs)


def build_html(records, pdb_text, centroid, args):
    data = {
        "pocketPdb": pdb_text,  # already filtered to binding-site residues
        "centroid": centroid,
        "pocketRadius": args.pocket_radius,
        "ligands": records,
        "colors": LIGAND_COLORS,
        "activityLabel": args.activity_col,
    }
    payload = json.dumps(data)
    ms = args.molstar_version
    return TEMPLATE.replace("__MOLSTAR_VERSION__", ms).replace(
        "__PAYLOAD__", payload
    )



TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Protein-Ligand Series Viewer</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/molstar@__MOLSTAR_VERSION__/build/viewer/molstar.css" />
<script src="https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js"></script>
<style>
  :root { --border:#d9dee3; --bg:#f6f8fa; --accent:#2563eb; --hb:#e6a800; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  #layout { display:flex; height:100vh; }
  #left  { flex:1 1 54%; position:relative; min-width:340px; background:#000; }
  #viewer { position:absolute; inset:0; }
  #right { flex:1 1 46%; min-width:420px; display:flex; flex-direction:column; border-left:1px solid var(--border); background:#fff; }
  #header { padding:10px 14px; border-bottom:1px solid var(--border); background:var(--bg); }
  #header h1 { font-size:15px; margin:0 0 4px; }
  #header .sub { font-size:12px; color:#57606a; }
  #header .sub b { color:var(--hb); }
  #toolbar { padding:6px 14px; border-bottom:1px solid var(--border); display:flex; gap:8px; align-items:center; font-size:12px; }
  #toolbar button { font-size:12px; padding:3px 8px; border:1px solid var(--border); background:#fff; border-radius:6px; cursor:pointer; }
  #toolbar button:hover { background:var(--bg); }
  #searchbar { padding:6px 14px; border-bottom:1px solid var(--border); display:flex; gap:8px; align-items:center; font-size:12px; }
  #searchbar select, #searchbar input { font-size:12px; border:1px solid var(--border); border-radius:6px; background:#fff; }
  #searchbar select { padding:3px 4px; cursor:pointer; }
  #searchInput { flex:1; padding:4px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  #searchInput:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 2px rgba(37,99,235,.15); }
  #searchClear { border:1px solid var(--border); background:#fff; border-radius:6px; cursor:pointer; padding:3px 8px; color:#57606a; }
  #searchClear:hover { background:var(--bg); }
  #searchInfo { color:#57606a; white-space:nowrap; min-width:64px; text-align:right; }
  #searchInfo.bad { color:#d1242f; }
  #gridwrap { overflow:auto; flex:1; padding:12px; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:12px; }
  .card { border:1px solid var(--border); border-radius:10px; background:#fff; padding:8px 10px 10px; display:flex; flex-direction:column; transition:box-shadow .12s, border-color .12s; }
  .card:hover { box-shadow:0 2px 10px rgba(0,0,0,.08); }
  .card.on { border-color:var(--accent); box-shadow:0 0 0 2px rgba(37,99,235,.15); }
  .card.dragging { opacity:.35; }
  .card .top { display:flex; align-items:center; gap:6px; }
  .card .handle { cursor:grab; color:#aeb6bf; font-size:13px; line-height:1; padding:2px 1px; user-select:none; touch-action:none; }
  .card .handle:active { cursor:grabbing; }
  .card .top label { display:flex; align-items:center; gap:7px; cursor:pointer; flex:1; min-width:0; }
  .card .name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:3px; flex:0 0 auto; border:1px solid rgba(0,0,0,.2); }
  .struct { text-align:center; margin:4px 0; }
  .struct svg { display:block; margin:0 auto; max-width:100%; height:auto; }
  .meta { display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:12px; }
  .act { font-variant-numeric: tabular-nums; font-weight:600; }
  .hb { font-size:10px; color:var(--hb); white-space:nowrap; }
  .actbar { flex:1; height:6px; border-radius:3px; background:#e7ecf2; overflow:hidden; }
  .actbar > i { display:block; height:100%; background:var(--accent); }
  input[type=checkbox] { width:16px; height:16px; cursor:pointer; flex:0 0 auto; }
  #status { padding:6px 14px; font-size:11px; color:#57606a; border-top:1px solid var(--border); }
</style>
</head>
<body>
<div id="layout">
  <div id="left"><div id="viewer"></div></div>
  <div id="right">
    <div id="header">
      <h1>Protein&ndash;Ligand Series Viewer</h1>
      <div class="sub">Binding site shown as wireframe, zoomed in. Check a ligand to show its 3D pose; predicted <b>H&ndash;bonds</b> to the protein are drawn as gold dashes. 2D structures are aligned on the maximum common substructure (shared core shaded blue). Drag the <span style="color:#8a929b">&#9776;</span> handle to reorder tiles.</div>
    </div>
    <div id="toolbar">
      <button id="showAll">Show all</button>
      <button id="hideAll">Hide all</button>
      <button id="refocus">Refocus pocket</button>
      <label style="display:flex;align-items:center;gap:5px;">sort:
        <select id="sortMode" style="font-size:12px;padding:2px 4px;border:1px solid var(--border);border-radius:6px;background:#fff;cursor:pointer;">
          <option value="none">file order</option>
          <option value="activity"></option>
          <option value="selected">by selected</option>
        </select>
      </label>
      <span id="count" style="margin-left:auto;color:#57606a;"></span>
    </div>
    <div id="searchbar">
      <select id="searchMode" title="Search mode">
        <option value="name">Name</option>
        <option value="smarts">SMARTS</option>
      </select>
      <input id="searchInput" type="text" placeholder="Search by name&hellip;" autocomplete="off" spellcheck="false" />
      <button id="searchClear" title="Clear search">&times;</button>
      <span id="searchInfo"></span>
    </div>
    <div id="gridwrap"><div id="grid"></div></div>
    <div id="status">Loading Mol*&hellip;</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/molstar@__MOLSTAR_VERSION__/build/viewer/molstar.js"></script>
<script>
const DATA = __PAYLOAD__;
const HB_COLOR = 0xE6A800;

function rgbCss(c){ return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`; }
function molstarColor(c){ return ((Math.round(c[0]*255)<<16) | (Math.round(c[1]*255)<<8) | Math.round(c[2]*255)) >>> 0; }

const status = document.getElementById('status');
const actLabel = DATA.activityLabel || 'pEC50';

// RDKit.js powers the SMARTS substructure search; it loads asynchronously.
let RDKit = null;
const rdkitReady = (typeof initRDKitModule === 'function'
    ? initRDKitModule() : Promise.reject(new Error('RDKit_minimal.js not loaded')))
  .then(m => { RDKit = m; return m; })
  .catch(err => { console.error('RDKit failed to load:', err); return null; });

// label the activity sort option with the actual activity column from the input
document.querySelector('#sortMode option[value="activity"]').textContent =
  `by ${actLabel} (high→low)`;

const acts = DATA.ligands.map(l => l.activity).filter(v => v !== null && v !== undefined);
const aMin = Math.min(...acts), aMax = Math.max(...acts);

(async function () {
  const viewer = await molstar.Viewer.create('viewer', {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    viewportShowExpand: true,
    viewportShowSelectionMode: false,
  });
  const plugin = viewer.plugin;

  // --- binding site as wireframe ---
  status.textContent = 'Loading binding site…';
  const pdbData = await plugin.builders.data.rawData({ data: DATA.pocketPdb, label: 'binding site' });
  const pTraj = await plugin.builders.structure.parseTrajectory(pdbData, 'pdb');
  const pModel = await plugin.builders.structure.createModel(pTraj);
  const pStruct = await plugin.builders.structure.createStructure(pModel);
  const pComp = await plugin.builders.structure.tryCreateComponentStatic(pStruct, 'all', { label: 'binding site' });
  await plugin.builders.structure.representation.addRepresentation(pComp, {
    type: 'line',
    color: 'element-symbol',
    typeParams: { sizeFactor: 1.4, lineSizeAttenuation: false },
  });

  const lig = {};   // name -> { dataRef, hbRef, color }
  DATA.ligands.forEach((l, i) => { lig[l.name] = { dataRef: null, hbRef: null, color: DATA.colors[i % DATA.colors.length] }; });

  function focusPocket() {
    plugin.managers.camera.focusSphere({ center: DATA.centroid, radius: DATA.pocketRadius });
  }
  focusPocket();

  async function showLigand(l) {
    const entry = lig[l.name];
    if (entry.dataRef) return;
    const data = await plugin.builders.data.rawData({ data: l.molblock, label: l.name });
    const traj = await plugin.builders.structure.parseTrajectory(data, 'mol');
    const model = await plugin.builders.structure.createModel(traj);
    const struct = await plugin.builders.structure.createStructure(model);
    const comp = await plugin.builders.structure.tryCreateComponentStatic(struct, 'all', { label: l.name });
    await plugin.builders.structure.representation.addRepresentation(comp, {
      type: 'ball-and-stick',
      color: 'element-symbol',
      colorParams: { carbonColor: { name: 'uniform', params: { value: molstarColor(entry.color) } } },
      typeParams: { sizeFactor: 0.28 },
    });
    entry.dataRef = data.ref;

    // H-bonds: dashed gold lines to the protein
    if (l.hbonds) {
      const hbData = await plugin.builders.data.rawData({ data: l.hbonds, label: l.name + ' H-bonds' });
      const hbTraj = await plugin.builders.structure.parseTrajectory(hbData, 'mol');
      const hbModel = await plugin.builders.structure.createModel(hbTraj);
      const hbStruct = await plugin.builders.structure.createStructure(hbModel);
      const hbComp = await plugin.builders.structure.tryCreateComponentStatic(hbStruct, 'all', { label: l.name + ' H-bonds' });
      await plugin.builders.structure.representation.addRepresentation(hbComp, {
        type: 'line',
        color: 'uniform',
        colorParams: { value: HB_COLOR },
        typeParams: { sizeFactor: 3.5, lineSizeAttenuation: false, dashedLines: true, dashLength: 0.3 },
      });
      entry.hbRef = hbData.ref;
    }
  }

  async function hideLigand(l) {
    const entry = lig[l.name];
    const b = plugin.state.data.build();
    let any = false;
    if (entry.dataRef) { b.delete(entry.dataRef); entry.dataRef = null; any = true; }
    if (entry.hbRef)   { b.delete(entry.hbRef);   entry.hbRef = null;   any = true; }
    if (any) await b.commit();
  }

  function updateCount() {
    const n = Object.values(lig).filter(e => e.dataRef).length;
    document.getElementById('count').textContent = n + ' shown';
  }

  // --- build grid of cards ---
  const grid = document.getElementById('grid');
  const cardEls = [];   // original (file) order, used to restore when unsorted
  DATA.ligands.forEach((l, i) => {
    const color = DATA.colors[i % DATA.colors.length];
    const card = document.createElement('div');
    card.className = 'card';

    const top = document.createElement('div');
    top.className = 'top';
    const handle = document.createElement('span');
    handle.className = 'handle'; handle.textContent = '☰'; handle.title = 'Drag to reorder';
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = rgbCss(color);
    const nm = document.createElement('span');
    nm.className = 'name'; nm.textContent = l.name; nm.title = l.name;
    label.append(cb, sw, nm);
    top.append(handle, label);

    // Only the handle initiates a drag, so checkboxes/text stay interactive.
    handle.addEventListener('pointerdown', () => { card.draggable = true; });
    card.addEventListener('dragstart', e => {
      card.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
    });
    card.addEventListener('dragend', () => {
      card.classList.remove('dragging');
      card.draggable = false;
    });

    const struct = document.createElement('div');
    struct.className = 'struct';
    struct.innerHTML = l.svg;

    const meta = document.createElement('div');
    meta.className = 'meta';
    let actHtml;
    if (l.activity === null || l.activity === undefined) {
      actHtml = `<span class="act">${actLabel}: &ndash;</span>`;
    } else {
      const frac = aMax > aMin ? (l.activity - aMin) / (aMax - aMin) : 1;
      actHtml = `<span class="act">${actLabel} ${l.activity.toFixed(2)}</span>` +
                `<span class="actbar"><i style="width:${(15 + frac * 85).toFixed(0)}%"></i></span>`;
    }
    const hbTxt = l.hbondCount ? `<span class="hb">${l.hbondCount} H&ndash;bond${l.hbondCount > 1 ? 's' : ''}</span>` : '';
    meta.innerHTML = actHtml + hbTxt;

    card.append(top, struct, meta);
    grid.appendChild(card);
    cardEls.push({ el: card, activity: l.activity, name: l.name, smiles: l.smiles, rdmol: undefined });

    cb.addEventListener('change', async () => {
      cb.disabled = true;
      try {
        if (cb.checked) { await showLigand(l); card.classList.add('on'); }
        else            { await hideLigand(l); card.classList.remove('on'); }
      } finally {
        cb.disabled = false;
        updateCount();
        if (document.getElementById('sortMode').value === 'selected') applySort('selected');
      }
    });
  });

  // grid-aware drag reordering: live-move the dragged card to the nearest slot
  function dragAfter(x, y) {
    const cards = [...grid.querySelectorAll('.card:not(.dragging)')];
    let best = null, bestDist = Infinity, bestBox = null;
    for (const el of cards) {
      const b = el.getBoundingClientRect();
      const d = Math.hypot(x - (b.left + b.width / 2), y - (b.top + b.height / 2));
      if (d < bestDist) { bestDist = d; best = el; bestBox = b; }
    }
    if (!best) return null;
    const before = y < bestBox.top + bestBox.height / 2 ||
                   (Math.abs(y - (bestBox.top + bestBox.height / 2)) <= bestBox.height / 2 &&
                    x < bestBox.left + bestBox.width / 2);
    return before ? best : best.nextSibling;
  }
  grid.addEventListener('dragover', e => {
    const dragging = grid.querySelector('.dragging');
    if (!dragging) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const after = dragAfter(e.clientX, e.clientY);
    if (after == null) grid.appendChild(dragging);
    else if (after !== dragging) grid.insertBefore(dragging, after);
  });
  grid.addEventListener('drop', e => e.preventDefault());

  document.getElementById('showAll').addEventListener('click', () => {
    for (const cb of grid.querySelectorAll('input[type=checkbox]')) {
      if (!cb.checked) { cb.checked = true; cb.dispatchEvent(new Event('change')); }
    }
  });
  document.getElementById('hideAll').addEventListener('click', () => {
    for (const cb of grid.querySelectorAll('input[type=checkbox]')) {
      if (cb.checked) { cb.checked = false; cb.dispatchEvent(new Event('change')); }
    }
  });
  document.getElementById('refocus').addEventListener('click', focusPocket);

  function applySort(mode) {
    let order;
    if (mode === 'activity') {
      order = [...cardEls].sort((a, b) => {
        const av = a.activity, bv = b.activity;
        if (av == null && bv == null) return 0;
        if (av == null) return 1;            // missing activity sinks to bottom
        if (bv == null) return -1;
        return bv - av;                       // high -> low
      });
    } else if (mode === 'selected') {
      // checked ligands first; stable sort keeps existing order within each group
      order = [...cardEls].sort((a, b) =>
        (a.el.classList.contains('on') ? 0 : 1) - (b.el.classList.contains('on') ? 0 : 1));
    } else {
      order = cardEls;                        // restore original file order
    }
    for (const c of order) grid.appendChild(c.el);
  }
  document.getElementById('sortMode').addEventListener('change', e => applySort(e.target.value));

  // --- search / filter ---
  const searchInput = document.getElementById('searchInput');
  const searchMode = document.getElementById('searchMode');
  const searchInfo = document.getElementById('searchInfo');

  // lazily build + cache an RDKit mol per ligand for SMARTS matching
  function getRdkitMol(c) {
    if (c.rdmol !== undefined) return c.rdmol;
    let m = null;
    try {
      m = RDKit.get_mol(c.smiles || '');
      if (m && !m.is_valid()) { m.delete(); m = null; }
    } catch (e) { m = null; }
    c.rdmol = m;
    return m;
  }

  function showOnly(predicate) {
    let n = 0;
    for (const c of cardEls) {
      const hit = predicate(c);
      c.el.style.display = hit ? '' : 'none';
      if (hit) n++;
    }
    return n;
  }

  function setInfo(text, bad) {
    searchInfo.textContent = text;
    searchInfo.classList.toggle('bad', !!bad);
  }

  function applyFilter() {
    const q = searchInput.value.trim();
    if (!q) {                       // empty query -> show everything
      for (const c of cardEls) c.el.style.display = '';
      setInfo('', false);
      return;
    }
    if (searchMode.value === 'smarts') {
      if (!RDKit) { setInfo('loading…', false); return; }
      let qmol = null;
      try { qmol = RDKit.get_qmol(q); } catch (e) { qmol = null; }
      if (!qmol || !qmol.is_valid()) {
        if (qmol) qmol.delete();
        setInfo('invalid SMARTS', true);
        return;                     // leave current view untouched on a bad pattern
      }
      const n = showOnly(c => {
        const m = getRdkitMol(c);
        if (!m) return false;
        try { return JSON.parse(m.get_substruct_match(qmol)).atoms !== undefined; }
        catch (e) { return false; }
      });
      qmol.delete();
      setInfo(n + (n === 1 ? ' match' : ' matches'), false);
    } else {                        // name substring (case-insensitive)
      const ql = q.toLowerCase();
      const n = showOnly(c => c.name.toLowerCase().includes(ql));
      setInfo(n + (n === 1 ? ' match' : ' matches'), false);
    }
  }

  searchInput.addEventListener('input', applyFilter);
  searchMode.addEventListener('change', () => {
    searchInput.placeholder = searchMode.value === 'smarts'
      ? 'Substructure SMARTS, e.g. c1ccccc1…' : 'Search by name…';
    applyFilter();
  });
  document.getElementById('searchClear').addEventListener('click', () => {
    searchInput.value = '';
    applyFilter();
    searchInput.focus();
  });
  // re-run once RDKit finishes loading, in case a SMARTS query is already typed
  rdkitReady.then(() => { if (searchMode.value === 'smarts' && searchInput.value.trim()) applyFilter(); });

  updateCount();
  status.textContent = `${DATA.ligands.length} ligands · binding site as wireframe · gold dashes = predicted H-bonds`;
})().catch(err => {
  status.textContent = 'Error: ' + err;
  console.error(err);
});
</script>
</body>
</html>
"""



def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    if args.name_col not in df.columns:
        sys.exit(f"CSV is missing the name column '{args.name_col}'. Found: {list(df.columns)}")

    sdf_mols = load_sdf(args.sdf)

    # Keep CSV order; only rows whose ligand exists in the SDF (need 2D + 3D).
    ordered_names = []
    for name in df[args.name_col].astype(str):
        if name in sdf_mols:
            ordered_names.append(name)
        else:
            print(f"[warn] '{name}' in CSV but not in SDF; skipping", file=sys.stderr)
    if not ordered_names:
        sys.exit("No CSV names matched SDF records (matched on molecule name).")

    depictions, matches = aligned_2d_depictions(sdf_mols, ordered_names)

    act_by_name = {}
    if args.activity_col in df.columns:
        for _, row in df.iterrows():
            act_by_name[str(row[args.name_col])] = row[args.activity_col]

    with open(args.pdb) as fh:
        pdb_text = fh.read()
    prot = parse_pdb_atoms(pdb_text)

    lig_mols = [sdf_mols[n] for n in ordered_names]
    all_lig_coords = np.vstack([m.GetConformer().GetPositions() for m in lig_mols])

    records = []
    total_hbonds = 0
    for name in ordered_names:
        m3d = sdf_mols[name]
        svg = render_svg(depictions[name], matches[name])
        molblock = Chem.MolToMolBlock(m3d, kekulize=True)
        hb_mb, hb_n = hbond_molblock(m3d, prot, args.hbond_min, args.hbond_max)
        total_hbonds += hb_n
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(m3d))
        except Exception:
            smiles = Chem.MolToSmiles(m3d)
        act = act_by_name.get(name, None)
        if act is not None and pd.isna(act):
            act = None
        records.append({
            "name": name,
            "svg": svg,
            "molblock": molblock,
            "smiles": smiles,
            "hbonds": hb_mb,
            "hbondCount": hb_n,
            "activity": None if act is None else float(act),
        })

    centroid = ligand_centroid(lig_mols)
    pocket_text = pocket_pdb(prot, all_lig_coords, args.pocket_cutoff)
    pocket_res = {prot["reskey"][i] for i in range(len(prot["lines"]))
                  if prot["lines"][i] in pocket_text}

    html = build_html(records, pocket_text, centroid, args)
    with open(args.out, "w") as fh:
        fh.write(html)

    print(f"Wrote {args.out}  ({len(records)} ligands, "
          f"{len(pocket_res)} binding-site residues as wire, "
          f"{total_hbonds} putative H-bonds, "
          f"center = [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}])")


if __name__ == "__main__":
    main()
