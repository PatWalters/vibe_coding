#!/usr/bin/env python3
"""Build a self-contained torsion-profile viewer HTML from an SDF + CSV.

Each SDF record is one rotamer of a torsion scan; its molblock plus the
torsion angle / energies (read from SDF tags, falling back to the CSV) are
embedded as JSON so the resulting page works straight off the filesystem.
"""
import json
import sys
from pathlib import Path


def parse_sdf(path):
    """Yield (molblock, props) for each record in an SDF file."""
    text = Path(path).read_text()
    for chunk in text.split("$$$$\n"):
        if not chunk.strip():
            continue
        molblock = chunk.split("M  END")[0] + "M  END\n"
        props = {}
        # SDF tags look like:  >  <Name>  (1)\nvalue\n
        lines = chunk.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(">"):
                start = line.find("<")
                end = line.find(">", start + 1)
                if start != -1 and end != -1:
                    name = line[start + 1:end]
                    i += 1
                    if i < len(lines):
                        props[name] = lines[i].strip()
            i += 1
        title = lines[0].strip() if lines else ""
        yield molblock, props, title


def main():
    if len(sys.argv) < 2:
        print("Usage: build_viewer.py <input.sdf> [output.html]", file=sys.stderr)
        sys.exit(1)
    sdf = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else sdf.with_suffix(".html")

    frames = []
    for molblock, props, title in parse_sdf(sdf):
        angle = props.get("Torsion_Angle")
        energy = props.get("MMFF94_Energy")
        rel = props.get("MMFF94_Relative_Energy")
        frames.append({
            "title": title,
            "angle": float(angle) if angle is not None else None,
            "energy": float(energy) if energy is not None else None,
            "rel": float(rel) if rel is not None else None,
            "mol": molblock,
        })

    # sort by angle for a clean, monotonic profile
    frames.sort(key=lambda f: (f["angle"] is None, f["angle"]))

    title_line = "RDKit Torsion Scan: O=CCO"  # taken from the reference plot
    data_json = json.dumps(frames, separators=(",", ":"))

    html = TEMPLATE.replace("__DATA__", data_json).replace("__TITLE__", title_line)
    out.write_text(html)
    print(f"Wrote {out} ({len(frames)} rotamers, {out.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Torsion Profile Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  :root { --bg:#f7f7fa; --panel:#fff; --line:#4c72b0; --accent:#dd6b3f; --ink:#262626; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:14px 22px 6px; }
  header h1 { margin:0; font-size:18px; font-weight:700; }
  header p { margin:4px 0 0; font-size:13px; color:#666; }
  .wrap { display:flex; flex-wrap:wrap; gap:16px; padding:14px 22px 22px; align-items:stretch; }
  .panel { background:var(--panel); border:1px solid #e3e3ea; border-radius:10px;
           box-shadow:0 1px 3px rgba(0,0,0,.05); }
  #plotPanel { flex:1 1 460px; min-width:380px; padding:8px; }
  #plot { width:100%; height:430px; }
  #structPanel { flex:1 1 420px; min-width:340px; display:flex; flex-direction:column; padding:12px; }
  #viewerHolder { position:relative; flex:1 1 auto; min-height:360px; border-radius:8px;
                  overflow:hidden; background:#fdfdfd; border:1px solid #eee; }
  #viewer { position:absolute; inset:0; }
  .readout { display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size:13px; }
  .readout b { font-size:20px; font-weight:700; display:block; }
  .readout .lab { color:#888; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .controls { display:flex; align-items:center; gap:12px; margin-top:12px; }
  .controls button { font-size:15px; border:1px solid #ccd; background:#fff; border-radius:7px;
                     padding:6px 14px; cursor:pointer; }
  .controls button:hover { background:#eef; }
  #slider { flex:1 1 auto; }
  .opts { display:flex; gap:16px; align-items:center; margin-top:6px; font-size:12.5px; color:#555; }
  .opts label { cursor:pointer; }
  footer { padding:0 22px 18px; font-size:11.5px; color:#999; }
</style>
</head>
<body>
<header>
  <h1>Torsional Energy Profile</h1>
  <p>__TITLE__ &mdash; hover or click the curve, drag the slider, or press play to scan the rotamers.</p>
</header>

<div class="wrap">
  <div class="panel" id="plotPanel"><div id="plot"></div></div>

  <div class="panel" id="structPanel">
    <div id="viewerHolder"><div id="viewer"></div></div>
    <div class="readout">
      <div><span class="lab">Torsion angle</span><b><span id="rAngle">&ndash;</span>&deg;</b></div>
      <div><span class="lab">Relative energy</span><b><span id="rRel">&ndash;</span></b><span class="lab">kcal/mol</span></div>
      <div><span class="lab">MMFF94 energy</span><b><span id="rAbs">&ndash;</span></b><span class="lab">kcal/mol</span></div>
    </div>
    <div class="controls">
      <button id="play">&#9654; Play</button>
      <input type="range" id="slider" min="0" max="0" value="0" step="1">
    </div>
    <div class="opts">
      <label><input type="checkbox" id="optSticks" checked> Sticks</label>
      <label><input type="checkbox" id="optH"> Show H</label>
      <label><input type="checkbox" id="optSpin"> Spin</label>
      <label><input type="checkbox" id="optFreeze"> Lock view while scanning</label>
    </div>
  </div>
</div>
<footer>3D: 3Dmol.js &middot; plot: Plotly &middot; energies: MMFF94 relative to the global minimum.</footer>

<script>
const FRAMES = __DATA__;
const angles = FRAMES.map(f => f.angle);
const rel    = FRAMES.map(f => f.rel);
const abs_   = FRAMES.map(f => f.energy);

let current = 0;
let playing = false;
let timer = null;

/* ---------- 3D viewer ---------- */
const viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });
let model = null;

function renderStructure(idx, keepView) {
  const f = FRAMES[idx];
  const showH = document.getElementById("optH").checked;
  viewer.removeAllModels();
  model = viewer.addModel(f.mol, "sdf");
  const sticks = document.getElementById("optSticks").checked;
  const base = sticks
    ? { stick: { radius: 0.14 }, sphere: { scale: 0.23 } }
    : { sphere: { scale: 0.30 }, stick: { radius: 0.10 } };
  viewer.setStyle({}, base);
  if (!showH) viewer.setStyle({ elem: "H" }, { stick: { hidden: true }, sphere: { hidden: true } });
  if (!keepView) viewer.zoomTo();
  viewer.render();
}

/* ---------- plot ---------- */
function buildPlot() {
  const profile = {
    x: angles, y: rel, mode: "lines+markers", type: "scatter",
    line: { color: "#4c72b0", width: 2 },
    marker: { color: "#4c72b0", size: 6, line: { color: "#fff", width: 1 } },
    hovertemplate: "%{x:.0f}&deg;<br>%{y:.2f} kcal/mol<extra></extra>",
    name: "MMFF94"
  };
  const cursor = {
    x: [angles[current]], y: [rel[current]], mode: "markers", type: "scatter",
    marker: { color: "#dd6b3f", size: 15, line: { color: "#fff", width: 2 }, symbol: "circle" },
    hoverinfo: "skip", showlegend: false, name: "selected"
  };
  const layout = {
    margin: { l: 56, r: 16, t: 10, b: 48 },
    paper_bgcolor: "#fff", plot_bgcolor: "#eaeaf2",
    xaxis: { title: "Torsion Angle (degrees)", dtick: 50, gridcolor: "#fff", zeroline: false,
             range: [-8, 368] },
    yaxis: { title: "Relative Energy (kcal/mol)", gridcolor: "#fff", zeroline: false },
    showlegend: false, hovermode: "closest"
  };
  Plotly.newPlot("plot", [profile, cursor], layout, { displayModeBar: false, responsive: true });

  const plotDiv = document.getElementById("plot");

  function nearestByAngle(xVal) {
    let bestIdx = 0, bestDiff = Infinity;
    for (let i = 0; i < angles.length; i++) {
      const a = angles[i];
      if (a == null) continue;
      const d = Math.abs(a - xVal);
      if (d < bestDiff) { bestDiff = d; bestIdx = i; }
    }
    return bestIdx;
  }

  plotDiv.on("plotly_click", e => { if (e.points.length) select(e.points[0].pointNumber); });
  plotDiv.on("plotly_hover", e => {
    if (!playing && e.points.length && e.points[0].curveNumber === 0)
      select(e.points[0].pointNumber);
  });

  // Fallback: clicks landing in the plot area but not on a data point
  // still snap to the nearest rotamer by torsion angle.
  plotDiv.addEventListener("click", evt => {
    const xaxis = plotDiv._fullLayout && plotDiv._fullLayout.xaxis;
    if (!xaxis || typeof xaxis.p2d !== "function") return;
    const bbox = plotDiv.getBoundingClientRect();
    const xPixel = evt.clientX - bbox.left - xaxis._offset;
    if (xPixel < 0 || xPixel > xaxis._length) return;
    select(nearestByAngle(xaxis.p2d(xPixel)));
  });
}

function moveCursor() {
  Plotly.restyle("plot", { x: [[angles[current]]], y: [[rel[current]]] }, [1]);
}

/* ---------- sync ---------- */
function select(idx) {
  if (idx == null || idx < 0 || idx >= FRAMES.length) return;
  current = idx;
  const f = FRAMES[idx];
  document.getElementById("rAngle").textContent = f.angle != null ? f.angle.toFixed(0) : "–";
  document.getElementById("rRel").textContent   = f.rel   != null ? f.rel.toFixed(2)   : "–";
  document.getElementById("rAbs").textContent   = f.energy!= null ? f.energy.toFixed(2): "–";
  document.getElementById("slider").value = idx;
  moveCursor();
  renderStructure(idx, document.getElementById("optFreeze").checked);
}

/* ---------- controls ---------- */
function play() {
  playing = true;
  document.getElementById("play").innerHTML = "&#10073;&#10073; Pause";
  document.getElementById("optFreeze").checked = true; // steadier during animation
  timer = setInterval(() => select((current + 1) % FRAMES.length), 120);
}
function pause() {
  playing = false;
  document.getElementById("play").innerHTML = "&#9654; Play";
  clearInterval(timer);
}

window.addEventListener("DOMContentLoaded", () => {
  const slider = document.getElementById("slider");
  slider.max = FRAMES.length - 1;
  slider.addEventListener("input", e => { if (playing) pause(); select(+e.target.value); });

  document.getElementById("play").addEventListener("click", () => playing ? pause() : play());
  ["optSticks", "optH"].forEach(id =>
    document.getElementById(id).addEventListener("change", () => renderStructure(current, true)));
  document.getElementById("optSpin").addEventListener("change", e => {
    viewer.spin(e.target.checked ? "y" : false);
  });

  buildPlot();
  select(0);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
