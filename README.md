# Vibe Coding Projects

A collection of lightweight, browser-based tools for molecular modeling and cheminformatics.

## 🛠 Projects

### ⚗ Docking Viewer
A standalone web application for visualizing and triaging molecular docking results. It combines 2D chemical structure rendering with high-performance 3D protein-ligand visualization.

#### Features
- **3D Visualization:** Powered by [Mol*](https://molstar.org/), providing interactive viewing of proteins and ligands with support for non-covalent interaction detection (H-bonds, hydrophobic contacts, etc.).
- **2D Depiction:** Clean chemical structure rendering using [RDKit.js](https://github.com/rdkit/rdkit-js).
- **Data Triage:** Easily navigate through a set of docked ligands, flag interesting hits, and export them as a new SDF file.
- **Property Analysis:** Displays calculated properties (MW, LogP, TPSA, QED, etc.) and medicinal chemistry filters (Lead-like, NIH, BRENK) directly from the SDF data.
- **Binding Site Focus:** Intelligent camera positioning and binding-site residue isolation for quick inspection of ligand poses.

#### How to Use
1. **Launch the Viewer:**
   - * Open `docking_viewer/docking_viewer.html` directly in a browser.
2. **Load Protein:** Click "📂 Protein (PDB)" and select your receptor file (e.g., `1r4l.pdb`).
3. **Load Ligands:** Click "📂 Ligands (SDF)" and select your docking results (e.g., `ligands_H.sdf`).
4. **Triage:** Use the arrow keys or UI buttons to navigate. Press `F` to flag hits.
5. **Export:** Click "⬇ Export Flagged" to save your selections.

## ⚖ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
