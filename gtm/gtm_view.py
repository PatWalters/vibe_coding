#!/usr/bin/env python3
"""GTM-based activity viewer: pie charts colored by binned activity.

Subcommands
-----------
generate  Train a GTM on a SMILES file (with a binned-activity column such as
          log2fc_bin) and write a coordinates CSV
          (SMILES,Name,bin,index,x,y,node_index).
view      Build a self-contained HTML viewer from a coordinates CSV.
          Each grid cell is a pie chart of the activity-bin distribution
          (low = red, medium = yellow, high = green). Clicking a pie loads that
          node's molecules into a sortable table on the right, with each
          structure rendered inline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CELL_PX = 44
PIE_RADIUS = 19
MIN_LOG_RADIUS_FRAC = 0.35

# Activity-bin colors. Low activity = red, medium = yellow, high = green.
BIN_COLORS = {
    "low": "#d62728",     # red
    "med": "#ffbf00",     # yellow/amber
    "high": "#2ca02c",    # green
}
BIN_LABELS = {
    "low": "low",
    "med": "medium",
    "high": "high",
}
# Order in which wedges are drawn (clockwise from 12 o'clock).
BIN_ORDER = ["low", "med", "high"]

# Synonyms accepted in the binned-activity column, normalized to the canonical
# keys above.
BIN_ALIASES = {
    "low": "low", "lo": "low", "l": "low", "0": "low",
    "med": "med", "medium": "med", "mid": "med", "m": "med", "1": "med",
    "high": "high", "hi": "high", "h": "high", "2": "high",
}

TABLE_STRUCTURE_PX = 120
TOOLTIP_STRUCTURE_PX = 150


def normalize_bin(value) -> str | None:
    """Map an arbitrary bin label to one of {low, med, high} or None."""
    if value is None:
        return None
    key = str(value).strip().lower()
    return BIN_ALIASES.get(key)


# ===========================================================================
# generate
# ===========================================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {suffix}")


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    import useful_rdkit_utils as uru
    from chemographykit.gtm import GTM
    from chemographykit.utils.molecules import calculate_latent_coords
    from rdkit.rdBase import BlockLogs
    from tqdm.auto import tqdm

    raw = read_table(args.input)
    print(f"Loaded {len(raw)} rows from {args.input}")

    for col, flag in [(args.smiles_col, "--smiles-col"),
                      (args.activity_col, "--activity-col")]:
        if col not in raw.columns:
            raise SystemExit(
                f"Column {col!r} (from {flag}) not found. "
                f"Available columns: {list(raw.columns)}"
            )

    df = pd.DataFrame({
        "SMILES": raw[args.smiles_col].astype(str),
        "bin_raw": raw[args.activity_col],
    })
    if args.name_col:
        if args.name_col not in raw.columns:
            raise SystemExit(
                f"Column {args.name_col!r} (from --name-col) not found. "
                f"Available columns: {list(raw.columns)}"
            )
        df["Name"] = raw[args.name_col].astype(str)

    # Normalize bins; drop rows whose bin label is unrecognized.
    df["bin"] = df["bin_raw"].apply(normalize_bin)
    bad = df["bin"].isna()
    if bad.any():
        offenders = sorted(set(map(str, df.loc[bad, "bin_raw"].unique())))[:10]
        print(f"Dropping {int(bad.sum())} rows with unrecognized bin labels "
              f"(e.g. {offenders}). Accepted: {sorted(set(BIN_ALIASES))}.")
        df = df[~bad].reset_index(drop=True)
    df = df.drop(columns=["bin_raw"])
    if df.empty:
        raise SystemExit("No rows with a recognized activity bin remain.")

    with BlockLogs():
        df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"]).drop(columns=["mol"]).reset_index(drop=True)
    print(f"After SMILES validation: {len(df)} rows")
    if df.empty:
        raise SystemExit("No valid SMILES remain after parsing.")

    if "Name" not in df.columns:
        df["Name"] = [f"MOL_{i:07d}" for i in range(len(df))]

    counts = df["bin"].value_counts().to_dict()
    print("Bin counts: " + ", ".join(f"{BIN_LABELS[b]}={counts.get(b, 0)}"
                                     for b in BIN_ORDER))

    smi2fp = uru.Smi2Fp()
    print("Computing fingerprints…")
    with BlockLogs():
        fps = np.stack([smi2fp.get_np(s) for s in tqdm(df["SMILES"])])

    if args.sample and len(df) > args.sample:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(len(df), size=args.sample, replace=False)
        train_fps = fps[train_idx]
        print(f"Training GTM on a random sample of {args.sample}/{len(df)}.")
    else:
        train_fps = fps
        print(f"Training GTM on all {len(train_fps)} molecules.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_nodes = args.grid_size ** 2
    rbf_edge = max(2, int(round(args.grid_size * 2 / 3)))
    num_basis = rbf_edge ** 2
    print(f"Fitting GTM ({args.grid_size}x{args.grid_size} = {num_nodes} nodes, "
          f"{rbf_edge}x{rbf_edge} = {num_basis} RBF centers, device={device})…")
    gtm_model = GTM(
        num_nodes=num_nodes,
        num_basis_functions=num_basis,
        basis_width=1.0,
        reg_coeff=1.0,
        device=device,
        standardize=False,
        pca_scale=True,
        pca_engine="torch",
        max_iter=args.max_iter,
    )
    train_t = torch.tensor(train_fps, dtype=torch.float64, device=device)
    gtm_model.fit(train_t)

    print("Projecting all molecules onto the GTM…")
    n_chunks = max(1, len(fps) // args.chunk_size)
    chunk_indices = np.array_split(np.arange(len(fps)), n_chunks)
    out_frames = []
    for ch in tqdm(chunk_indices):
        X_t = torch.tensor(fps[ch], dtype=torch.float64, device=device)
        responsibilities, _ = gtm_model.project(X_t)
        R_np = responsibilities.detach().to("cpu").numpy()
        if R_np.shape[0] != len(ch):
            R_np = R_np.T
        out_frames.append(
            calculate_latent_coords(R_np, correction=True, return_node=True)
        )
    gtm_df = pd.concat(out_frames, ignore_index=True)

    if "index" not in gtm_df.columns:
        gtm_df.insert(0, "index", np.arange(len(gtm_df)))

    base = df[["SMILES", "Name", "bin"]].reset_index(drop=True)
    merged = pd.concat([base, gtm_df.reset_index(drop=True)], axis=1)

    final_cols = ["SMILES", "Name", "bin", "index", "x", "y", "node_index"]
    missing = [c for c in final_cols if c not in merged.columns]
    if missing:
        raise SystemExit(
            f"GTM projection did not produce expected columns: missing {missing}. "
            f"Got: {list(merged.columns)}"
        )

    merged[final_cols].to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(merged)} rows)")


# ===========================================================================
# view
# ===========================================================================

_CLASS_ATTR_RE = re.compile(r" class='[^']*'")
_RDKIT_NS_RE   = re.compile(r"\s*xmlns:rdkit='[^']*'")
_XLINK_NS_RE   = re.compile(r"\s*xmlns:xlink='[^']*'")
_XMLSPACE_RE   = re.compile(r"\s*xml:space='preserve'")
_HDRCMT_RE     = re.compile(r"<!-- END OF HEADER -->\s*")


def render_smiles_svg(smiles: str, size: int) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1].lstrip()
    svg = _CLASS_ATTR_RE.sub("", svg)
    svg = _RDKIT_NS_RE.sub("", svg)
    svg = _XLINK_NS_RE.sub("", svg)
    svg = _XMLSPACE_RE.sub("", svg)
    svg = _HDRCMT_RE.sub("", svg)
    return svg


def build_grid_skeleton_svg(grid_size: int, populated_cells: set) -> str:
    """Render the grid skeleton (background + hit rectangles + cell groups).

    Pies are drawn into these groups by JS.
    """
    width = grid_size * CELL_PX
    height = grid_size * CELL_PX

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'class="pie-grid" preserveAspectRatio="xMidYMid meet">',
        '<g stroke="#eee" stroke-width="1">',
    ]
    for i in range(grid_size + 1):
        parts.append(
            f'<line x1="{i * CELL_PX}" y1="0" x2="{i * CELL_PX}" y2="{height}"/>'
        )
        parts.append(
            f'<line x1="0" y1="{i * CELL_PX}" x2="{width}" y2="{i * CELL_PX}"/>'
        )
    parts.append("</g>")

    for col in range(1, grid_size + 1):
        for row in range(1, grid_size + 1):
            x0 = (col - 1) * CELL_PX
            y0 = (grid_size - row) * CELL_PX
            cell_id = f"{col}_{row}"
            has_data = (col, row) in populated_cells
            klass = "cell" + ("" if has_data else " empty")
            parts.append(
                f'<g class="{klass}" data-cell="{cell_id}">'
                f'<rect class="hit" x="{x0}" y="{y0}" '
                f'width="{CELL_PX}" height="{CELL_PX}" '
                f'fill="white" fill-opacity="0"/>'
                f'<g class="pie"></g>'
                f'</g>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def collect_cell_data(df: pd.DataFrame) -> dict[str, dict]:
    """For each populated cell, return the molecules it contains.

    Each molecule carries its rendered SVG so the table on the right can
    display it without re-rendering in the browser.
    """
    out: dict[str, dict] = {}
    for (col, row), group in df.groupby(["col", "row"]):
        molecules = []
        for _, r in group.iterrows():
            svg = render_smiles_svg(r["SMILES"], TABLE_STRUCTURE_PX)
            if svg is None:
                continue
            molecules.append({
                "name": str(r["Name"]),
                "smiles": str(r["SMILES"]),
                "bin": str(r["bin"]),
                "svg": svg,
            })
        if molecules:
            out[f"{col}_{row}"] = {"molecules": molecules}
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GTM activity viewer</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 18px; color: #222; }}
  h1 {{ margin: 0 0 4px 0; font-size: 20px; }}
  p.sub {{ margin: 0 0 14px 0; color: #666; font-size: 13px; }}
  .container {{ display: flex; gap: 28px; align-items: flex-start; }}
  .panel-title {{ margin: 0 0 8px 0; font-size: 14px; color: #444; }}
  .pie-grid {{ width: 640px; height: 640px; background: #fff;
               border: 1px solid #ddd; border-radius: 4px; }}
  .cell {{ cursor: pointer; }}
  .cell .hit {{ pointer-events: all; }}
  .cell:hover .hit {{ fill: #5b9bd5; fill-opacity: 0.12; }}
  .cell.selected .hit {{ fill: #5b9bd5; fill-opacity: 0.25;
                          stroke: #2c6fb6; stroke-width: 1.5; }}
  .cell.empty {{ cursor: default; }}
  .cell.empty:hover .hit {{ fill-opacity: 0; }}
  .controls {{ display: flex; flex-direction: column; gap: 10px;
               margin-top: 12px; padding: 10px;
               border: 1px solid #ddd; border-radius: 4px;
               background: #fafafa; font-size: 13px; max-width: 640px; }}
  .toggle-row {{ display: flex; align-items: center; gap: 10px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px;
             margin-top: 6px; font-size: 13px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 2px; }}

  .right {{ width: 620px; }}
  .right-header {{ display: flex; justify-content: space-between;
                    align-items: baseline; margin-bottom: 8px; }}
  .sort-row {{ display: flex; align-items: center; gap: 8px;
               margin-bottom: 8px; font-size: 12px; color: #444; }}
  .sort-row button {{ font-size: 12px; padding: 3px 9px; cursor: pointer;
                       border: 1px solid #ccc; border-radius: 4px;
                       background: #fff; color: #444; }}
  .sort-row button.active {{ background: #5b9bd5; border-color: #2c6fb6;
                              color: #fff; }}
  .grid-wrap {{ border: 1px solid #ddd; border-radius: 4px;
                background: #fff; max-height: 720px; overflow: auto;
                padding: 10px; }}
  .mol-grid {{ display: grid;
               grid-template-columns: repeat(auto-fill, minmax({struct_px}px, 1fr));
               gap: 10px; }}
  .mol-card {{ border: 1px solid #eee; border-top: 4px solid #ccc;
               border-radius: 4px; padding: 6px; background: #fff;
               display: flex; flex-direction: column; align-items: center; }}
  .mol-card.low {{ border-top-color: {low_color}; }}
  .mol-card.med {{ border-top-color: {med_color}; }}
  .mol-card.high {{ border-top-color: {high_color}; }}
  .mol-card svg {{ width: {struct_px}px; height: {struct_px}px; display: block; }}
  .mol-card .m-name {{ font-size: 11px; color: #222; margin-top: 4px;
                        max-width: {struct_px}px; overflow: hidden;
                        text-overflow: ellipsis; white-space: nowrap;
                        text-align: center; }}
  .mol-card .m-bin {{ font-size: 11px; font-weight: 600; margin-top: 2px; }}
  .mol-card.low  .m-bin {{ color: {low_color}; }}
  .mol-card.med  .m-bin {{ color: {med_color}; }}
  .mol-card.high .m-bin {{ color: {high_color}; }}
  .placeholder {{ padding: 24px; color: #888; font-style: italic;
                   text-align: center; }}

  #tooltip {{ position: fixed; pointer-events: none; z-index: 1000;
              background: #fff; border: 1px solid #bbb; border-radius: 4px;
              padding: 6px 8px; font-size: 11px; color: #222;
              box-shadow: 0 2px 6px rgba(0,0,0,0.15);
              display: none; max-width: 200px; }}
  #tooltip .tt-count {{ font-weight: 600; margin-bottom: 4px; font-size: 12px; }}
  #tooltip .tt-bins {{ margin-bottom: 4px; }}
  #tooltip .tt-bin {{ display: flex; align-items: center; gap: 4px; line-height: 1.3; }}
  #tooltip .tt-bin .sw {{ width: 10px; height: 10px; border-radius: 2px;
                          display: inline-block; }}
  #tooltip .tt-svg {{ width: {tt_px}px; height: {tt_px}px; }}
  #tooltip .tt-svg svg {{ width: 100%; height: 100%; }}
  #tooltip .tt-name {{ margin-top: 2px; overflow: hidden;
                       text-overflow: ellipsis; white-space: nowrap; }}
</style>
</head>
<body>
<h1>GTM activity viewer</h1>
<p class="sub">Generative topographic mapping of {n_mols} molecules. Each pie shows the activity-bin distribution of the molecules at that node. Click a pie to load its molecules into the structure grid on the right.</p>
<div class="container">
  <div class="left">
    <div class="panel-title">{grid_size} × {grid_size} GTM grid · activity-bin distribution per node</div>
    {grid_svg}
    <div class="controls">
      <div class="toggle-row">
        <input type="checkbox" id="log-size" checked>
        <label for="log-size">Scale pie radius by log(count)</label>
      </div>
      <div class="legend" id="legend"></div>
    </div>
  </div>
  <div class="right">
    <div class="right-header">
      <div class="panel-title" id="right-title">Molecules in selected node</div>
      <div class="panel-title" id="right-stats" style="color:#666"></div>
    </div>
    <div class="sort-row">
      <span>Sort by:</span>
      <button type="button" data-key="bin" class="active">Activity</button>
      <button type="button" data-key="name">Name</button>
    </div>
    <div class="grid-wrap">
      <div class="mol-grid" id="mol-grid">
        <div class="placeholder">Select a cell on the left.</div>
      </div>
    </div>
  </div>
</div>
<div id="tooltip" role="tooltip" aria-hidden="true"></div>
<script id="cell-data" type="application/json">{cell_json}</script>
<script>
(function () {{
  const CELLS = JSON.parse(document.getElementById('cell-data').textContent);
  const CELL_PX = {cell_px};
  const PIE_RADIUS = {pie_radius};
  const MIN_LOG_RADIUS_FRAC = {min_log_radius_frac};
  const COLORS = {{ low: "{low_color}", med: "{med_color}", high: "{high_color}" }};
  const BIN_LABELS = {{ low: "low", med: "medium", high: "high" }};
  const BIN_ORDER = ["low", "med", "high"];
  const BIN_RANK = {{ low: 0, med: 1, high: 2 }};
  const GRID_SIZE = {grid_size};

  const logBox  = document.getElementById('log-size');
  const legend  = document.getElementById('legend');
  const grid    = document.getElementById('mol-grid');
  const title   = document.getElementById('right-title');
  const stats   = document.getElementById('right-stats');
  const tooltip = document.getElementById('tooltip');

  let selected = null;
  let currentMols = [];
  let currentSort = {{ key: "bin", desc: true }};

  function escapeHtml(s) {{
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }}

  function binCounts(bins) {{
    const c = {{ low: 0, med: 0, high: 0 }};
    for (const b of bins) if (b in c) c[b] += 1;
    return c;
  }}

  function pieWedgePath(cx, cy, r, startDeg, endDeg) {{
    const a1 = (startDeg - 90) * Math.PI / 180;
    const a2 = (endDeg   - 90) * Math.PI / 180;
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2);
    const largeArc = (endDeg - startDeg) > 180 ? 1 : 0;
    return `M ${{cx.toFixed(2)}},${{cy.toFixed(2)}} ` +
           `L ${{x1.toFixed(2)}},${{y1.toFixed(2)}} ` +
           `A ${{r.toFixed(2)}},${{r.toFixed(2)}} 0 ${{largeArc}},1 ` +
           `${{x2.toFixed(2)}},${{y2.toFixed(2)}} Z`;
  }}

  function svgEl(name, attrs, children) {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (children) for (const c of children) el.appendChild(c);
    return el;
  }}

  function renderPies() {{
    const logSize = logBox.checked;

    let maxLog = 1.0;
    for (const id in CELLS) {{
      const lg = Math.log1p(CELLS[id].molecules.length);
      if (lg > maxLog) maxLog = lg;
    }}

    document.querySelectorAll('.cell').forEach(g => {{
      const pie = g.querySelector('.pie');
      while (pie.firstChild) pie.removeChild(pie.firstChild);

      if (g.classList.contains('empty')) return;

      const cellId = g.dataset.cell;
      const data = CELLS[cellId];
      if (!data) return;

      const [colStr, rowStr] = cellId.split('_');
      const col = parseInt(colStr, 10);
      const row = parseInt(rowStr, 10);
      const x0 = (col - 1) * CELL_PX;
      const y0 = (GRID_SIZE - row) * CELL_PX;
      const cx = x0 + CELL_PX / 2;
      const cy = y0 + CELL_PX / 2;

      const total = data.molecules.length;
      const counts = binCounts(data.molecules.map(m => m.bin));

      let radius = PIE_RADIUS;
      if (logSize && maxLog > 0) {{
        const frac = MIN_LOG_RADIUS_FRAC +
                     (1 - MIN_LOG_RADIUS_FRAC) * (Math.log1p(total) / maxLog);
        radius = PIE_RADIUS * frac;
      }}

      g.dataset.total  = String(total);
      g.dataset.lowCt  = String(counts.low);
      g.dataset.medCt  = String(counts.med);
      g.dataset.highCt = String(counts.high);

      let angle = 0;
      let drawn = 0;
      for (const bin of BIN_ORDER) {{
        const c = counts[bin];
        if (c === 0) continue;
        drawn += 1;
        const fraction = c / total;
        const end = angle + fraction * 360.0;
        if (fraction >= 0.999 && drawn === 1) {{
          pie.appendChild(svgEl("circle", {{
            cx: cx.toFixed(2), cy: cy.toFixed(2), r: radius.toFixed(2),
            fill: COLORS[bin], stroke: "white", "stroke-width": "0.6",
          }}));
        }} else {{
          pie.appendChild(svgEl("path", {{
            d: pieWedgePath(cx, cy, radius, angle, end),
            fill: COLORS[bin], stroke: "white", "stroke-width": "0.6",
          }}));
        }}
        angle = end;
      }}
    }});
  }}

  function renderLegend() {{
    legend.innerHTML = [
      ['low',  'low activity'],
      ['med',  'medium activity'],
      ['high', 'high activity'],
    ].map(([k, lbl]) => `
      <div class="legend-item">
        <span class="legend-swatch" style="background:${{COLORS[k]}}"></span>${{lbl}}
      </div>
    `).join('');
  }}

  function renderGrid() {{
    if (!currentMols.length) {{
      grid.innerHTML = '<div class="placeholder">Select a cell on the left.</div>';
      stats.textContent = "";
      return;
    }}
    const sorted = currentMols.slice();
    const k = currentSort.key;
    const dir = currentSort.desc ? -1 : 1;
    sorted.sort((a, b) => {{
      let av, bv;
      if (k === "bin") {{ av = BIN_RANK[a.bin]; bv = BIN_RANK[b.bin]; }}
      else {{ av = String(a.name); bv = String(b.name); }}
      if (av < bv) return -1 * dir;
      if (av > bv) return  1 * dir;
      return 0;
    }});

    grid.innerHTML = sorted.map(m => {{
      const label = BIN_LABELS[m.bin] || m.bin;
      return `
        <div class="mol-card ${{m.bin}}">
          ${{m.svg}}
          <div class="m-name" title="${{escapeHtml(m.name)}}">${{escapeHtml(m.name)}}</div>
          <div class="m-bin">${{escapeHtml(label)}}</div>
        </div>
      `;
    }}).join('');

    const counts = binCounts(currentMols.map(m => m.bin));
    stats.textContent =
      `${{currentMols.length}} molecules · ` +
      `low: ${{counts.low}}, medium: ${{counts.med}}, high: ${{counts.high}}`;
  }}

  // --- events -------------------------------------------------------------

  logBox.addEventListener('change', renderPies);

  document.querySelectorAll('.sort-row button[data-key]').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const key = btn.dataset.key;
      if (currentSort.key === key) {{
        currentSort.desc = !currentSort.desc;
      }} else {{
        currentSort.key = key;
        currentSort.desc = (key === "bin");
      }}
      document.querySelectorAll('.sort-row button[data-key]').forEach(b =>
        b.classList.toggle('active', b.dataset.key === currentSort.key));
      renderGrid();
    }});
  }});

  function placeTooltip(evt) {{
    const pad = 14;
    const tw = tooltip.offsetWidth;
    const th = tooltip.offsetHeight;
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + tw > window.innerWidth - 8) x = evt.clientX - tw - pad;
    if (y + th > window.innerHeight - 8) y = evt.clientY - th - pad;
    tooltip.style.left = Math.max(4, x) + 'px';
    tooltip.style.top  = Math.max(4, y) + 'px';
  }}

  document.querySelectorAll('.cell').forEach(g => {{
    if (g.classList.contains('empty')) return;
    const cellId = g.dataset.cell;

    g.addEventListener('click', () => {{
      const data = CELLS[cellId];
      if (!data) return;
      if (selected) selected.classList.remove('selected');
      g.classList.add('selected');
      selected = g;
      const [col, row] = cellId.split('_');
      title.textContent = `Cell (col ${{col}}, row ${{row}})`;
      currentMols = data.molecules;
      currentSort = {{ key: "bin", desc: true }};
      document.querySelectorAll('.sort-row button[data-key]').forEach(b =>
        b.classList.toggle('active', b.dataset.key === currentSort.key));
      renderGrid();
    }});

    g.addEventListener('mouseenter', (evt) => {{
      const data = CELLS[cellId];
      if (!data) return;
      const total = data.molecules.length;
      const noun = total === 1 ? 'molecule' : 'molecules';
      const lowCt  = g.dataset.lowCt  || '0';
      const medCt  = g.dataset.medCt  || '0';
      const highCt = g.dataset.highCt || '0';
      const ex = data.molecules[0];
      const exHtml = ex
        ? `<div class="tt-svg">${{ex.svg}}</div>
           <div class="tt-name" title="${{escapeHtml(ex.name)}}">${{escapeHtml(ex.name)}}</div>`
        : '';
      tooltip.innerHTML = `
        <div class="tt-count">${{total}} ${{noun}}</div>
        <div class="tt-bins">
          <div class="tt-bin"><span class="sw" style="background:${{COLORS.low}}"></span>low: ${{lowCt}}</div>
          <div class="tt-bin"><span class="sw" style="background:${{COLORS.med}}"></span>medium: ${{medCt}}</div>
          <div class="tt-bin"><span class="sw" style="background:${{COLORS.high}}"></span>high: ${{highCt}}</div>
        </div>
        ${{exHtml}}
      `;
      tooltip.style.display = 'block';
      placeTooltip(evt);
    }});

    g.addEventListener('mousemove', placeTooltip);
    g.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
  }});

  renderLegend();
  renderPies();
  renderGrid();
}})();
</script>
</body>
</html>
"""


def cmd_view(args: argparse.Namespace) -> None:
    if args.grid_size < 1:
        raise SystemExit("--grid-size must be >= 1")

    df = pd.read_csv(args.input)

    # Resolve column names, tolerating either the canonical `generate` output
    # or a raw CSV where the user names the SMILES / bin columns directly.
    rename = {}
    if "SMILES" not in df.columns and args.smiles_col in df.columns:
        rename[args.smiles_col] = "SMILES"
    if "bin" not in df.columns and args.activity_col in df.columns:
        rename[args.activity_col] = "bin"
    if args.name_col and "Name" not in df.columns and args.name_col in df.columns:
        rename[args.name_col] = "Name"
    if rename:
        df = df.rename(columns=rename)

    if args.name_col and "Name" not in df.columns:
        raise SystemExit(
            f"Column {args.name_col!r} (from --name-col) not found. "
            f"Got: {list(df.columns)}"
        )

    required = {"x", "y", "SMILES", "bin"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"missing column(s) in {args.input.name}: {sorted(missing)}. "
            f"Got: {list(df.columns)}"
        )

    if "Name" not in df.columns:
        df["Name"] = [f"MOL_{i:07d}" for i in range(len(df))]

    df["bin"] = df["bin"].apply(normalize_bin)
    df = df.dropna(subset=["x", "y", "bin"]).copy()
    if df.empty:
        raise SystemExit("No rows with valid x, y, and a recognized bin remain.")

    df["col"] = np.floor(df["x"]).astype(int)
    df["row"] = np.floor(df["y"]).astype(int)
    df = df[(df["col"].between(1, args.grid_size)) &
            (df["row"].between(1, args.grid_size))].copy()

    populated = set(map(tuple, df[["col", "row"]].drop_duplicates().to_numpy()))
    grid_svg = build_grid_skeleton_svg(args.grid_size, populated)
    cell_data = collect_cell_data(df)

    cell_json = json.dumps(cell_data, separators=(",", ":"))
    cell_json = cell_json.replace("</", "<\\/")

    page = HTML_TEMPLATE.format(
        grid_svg=grid_svg,
        cell_json=cell_json,
        grid_size=args.grid_size,
        n_mols=len(df),
        cell_px=CELL_PX,
        pie_radius=PIE_RADIUS,
        min_log_radius_frac=MIN_LOG_RADIUS_FRAC,
        low_color=BIN_COLORS["low"],
        med_color=BIN_COLORS["med"],
        high_color=BIN_COLORS["high"],
        struct_px=TABLE_STRUCTURE_PX,
        tt_px=TOOLTIP_STRUCTURE_PX,
    )
    args.output.write_text(page)
    print(f"wrote {args.output}  ({args.output.stat().st_size / 1024:.1f} KB, "
          f"{len(cell_data)} populated cells, "
          f"{len(df)} molecules)")


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gtm_ml.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{generate,view}")

    g = sub.add_parser(
        "generate",
        help="train a GTM on a SMILES + binned-activity file and write a coordinates CSV",
        description="Train a GTM on a SMILES file with a binned-activity column "
                    "(low/med/high). Output a CSV with "
                    "SMILES,Name,bin,index,x,y,node_index.",
    )
    g.add_argument("input", type=Path,
                   help="Input file (.csv/.tsv/.parquet).")
    g.add_argument("--smiles-col", default="smiles",
                   help="Name of the SMILES column (default: smiles).")
    g.add_argument("--activity-col", default="log2fc_bin",
                   help="Name of the binned-activity column with low/med/high "
                        "labels (default: log2fc_bin).")
    g.add_argument("--name-col", default=None,
                   help="Optional column to use as the molecule name "
                        "(default: auto-generated MOL_####### ids).")
    g.add_argument("--grid-size", type=int, required=True,
                   help="Square grid edge length (e.g. 15 → 15x15 grid).")
    g.add_argument("--output", type=Path, default=Path("gtm_activity_coords.csv"),
                   help="Output CSV path (default: gtm_activity_coords.csv).")
    g.add_argument("--sample", type=int, default=100_000,
                   help="Random sample size used to fit the GTM "
                        "(default: 100000; use 0 to fit on all rows).")
    g.add_argument("--chunk-size", type=int, default=10_000,
                   help="Rows per projection chunk (default: 10000).")
    g.add_argument("--max-iter", type=int, default=200,
                   help="GTM EM iterations (default: 200).")
    g.add_argument("--seed", type=int, default=42,
                   help="Random seed for the training-set sample (default: 42).")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser(
        "view",
        help="build an interactive HTML viewer from a coordinates CSV",
        description="Render a clickable pie-chart grid (activity-bin "
                    "distribution per node: low=red, medium=yellow, high=green) "
                    "with a log-size toggle and a sortable molecule table.",
    )
    v.add_argument("--input", type=Path, required=True,
                   help="Input coordinates CSV produced by `generate` "
                        "(or any CSV with SMILES, bin, x, y columns).")
    v.add_argument("--smiles-col", default="smiles",
                   help="SMILES column name if not already 'SMILES' (default: smiles).")
    v.add_argument("--activity-col", default="log2fc_bin",
                   help="Binned-activity column name if not already 'bin' "
                        "(default: log2fc_bin).")
    v.add_argument("--name-col", default=None,
                   help="Column to use as the molecule name if not already "
                        "'Name' (default: auto-generated MOL_####### ids).")
    v.add_argument("--grid-size", type=int, default=15,
                   help="Square grid edge length (default: 15).")
    v.add_argument("--output", type=Path, default=Path("gtm_activity.html"),
                   help="Output HTML path (default: gtm_activity.html).")
    v.set_defaults(func=cmd_view)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
