#!/usr/bin/env python3
"""
A desktop viewer for Butina clustering of chemical structures.

Open a SMILES file (via the "Open SMILES File" button or by dropping the file
onto the window). The molecules are clustered with the Butina algorithm using
Tanimoto distances between Morgan fingerprints. Results are shown as two grids:

    Left  : the cluster centers (one centroid structure per cluster).
    Right : the members of the cluster selected on the left.

Clicking a cluster center on the left fills the right grid with that cluster's
members. Each grid has its own Cols/Rows controls that set how many structures
are shown across and down (and therefore how large each structure is); extra
structures scroll vertically. A progress bar tracks the clustering, which runs
on a background thread so the interface stays responsive.

SMILES file format: one molecule per line, "<SMILES> <name>" (whitespace
separated). Lines that fail to parse are skipped.

Usage:
    python cluster_viewer.py [optional_smiles_file]
"""

import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdDepictor, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Cluster import Butina

from PyQt5 import QtCore, QtGui, QtWidgets

# Distance cutoff below which two molecules join the same cluster. A Tanimoto
# distance of 0.35 corresponds to a Tanimoto similarity of 0.65, a common
# starting point for medicinal-chemistry-scale clustering.
DEFAULT_CUTOFF = 0.35

# Morgan fingerprint settings used for the similarity calculation.
FP_RADIUS = 2
FP_SIZE = 2048

# Default render size (pixels); the real size is derived from the grid layout.
IMG_W, IMG_H = 260, 200

# Card-grid geometry (pixels).
GAP = 12        # space between cards
MARGIN = 12     # space around the whole grid
PAD = 8         # padding inside a card, between its edge and contents
LINE_H = 16     # vertical room per caption line
CHECK_H = 24    # vertical room for a card's checkbox row
MIN_CARD = 80   # smallest a card is allowed to get

# Light, GitHub-ish palette mirroring the pl_viewer HTML styling.
CARD_QSS = """
QScrollArea { border: none; }
QWidget#cardbody { background: #f6f8fa; }
QFrame#card {
    border: 1px solid #d9dee3;
    border-radius: 10px;
    background: #ffffff;
}
QFrame#card:hover { border-color: #9aa4af; }
QFrame#card[selected="true"] {
    border-color: #2563eb;
    background: #eff5ff;
}
QLabel#cardtitle { color: #57606a; font-size: 11px; font-weight: 600; }
QLabel#cardname {
    color: #1f2328;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 11px;
}
"""

# Window-level styling for the toolbar / panel headers. Text colors are set
# explicitly so the light backgrounds stay legible under a dark system theme.
APP_QSS = """
QMainWindow, QWidget#central { background: #ffffff; }
QLabel { color: #1f2328; }
QLabel#paneltitle { font-size: 13px; color: #1f2328; }
QToolBar, QWidget#toolbar { background: #f6f8fa; }
QPushButton#openbtn {
    border: 1px solid #d9dee3; border-radius: 6px;
    padding: 5px 12px; background: #ffffff; color: #1f2328;
}
QPushButton#openbtn:hover { background: #f0f3f6; }
QPushButton#openbtn:disabled { color: #aeb6bf; }
QSpinBox, QDoubleSpinBox {
    border: 1px solid #d9dee3; border-radius: 6px; padding: 2px 4px;
    background: #ffffff; color: #1f2328;
}
QProgressBar {
    border: 1px solid #d9dee3; border-radius: 6px; text-align: center;
    background: #ffffff; color: #1f2328;
}
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
QTabWidget::pane { border: 1px solid #d9dee3; border-radius: 6px; top: -1px; }
QTabBar::tab {
    color: #57606a; background: #eef1f4;
    border: 1px solid #d9dee3; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    padding: 5px 14px; margin-right: 2px;
}
QTabBar::tab:selected { color: #1f2328; background: #ffffff; }
QTabBar::tab:hover { color: #1f2328; }
"""


def render_mol(mol, width=IMG_W, height=IMG_H):
    """Render an RDKit molecule to a QPixmap using the Cairo PNG backend."""
    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    opts = drawer.drawOptions()
    opts.clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    image = QtGui.QImage.fromData(png, "PNG")
    return QtGui.QPixmap.fromImage(image)


class ClusterWorker(QtCore.QThread):
    """Parse, fingerprint, and Butina-cluster molecules off the UI thread."""

    progress = QtCore.pyqtSignal(int)          # 0-100
    message = QtCore.pyqtSignal(str)           # status text
    # Emitted on success: (mols, names, clusters). clusters is a list of
    # index-tuples; the first index of each tuple is the cluster centroid.
    finished_ok = QtCore.pyqtSignal(object, object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path, cutoff, parent=None):
        super().__init__(parent)
        self.path = path
        self.cutoff = cutoff

    def run(self):
        try:
            self.message.emit("Reading SMILES file...")
            mols, names = self._read_smiles()
            if not mols:
                self.failed.emit("No valid molecules were found in the file.")
                return

            n = len(mols)
            self.message.emit(f"Generating fingerprints for {n} molecules...")
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=FP_RADIUS, fpSize=FP_SIZE
            )
            fps = []
            for i, mol in enumerate(mols):
                fps.append(gen.GetFingerprint(mol))
                if i % 50 == 0:
                    # Fingerprinting occupies the first 40% of the progress bar.
                    self.progress.emit(int(40 * (i + 1) / n))

            self.message.emit("Computing distance matrix...")
            dists = self._distance_matrix(fps)

            self.message.emit("Clustering...")
            clusters = Butina.ClusterData(
                dists, n, self.cutoff, isDistData=True
            )
            # Butina returns clusters largest-first with the centroid first;
            # keep that order so the left grid reads top-down by size.
            clusters = [tuple(c) for c in clusters]
            self.progress.emit(100)
            self.message.emit(
                f"{n} molecules in {len(clusters)} clusters "
                f"(cutoff {self.cutoff:.2f})."
            )
            self.finished_ok.emit(mols, names, clusters)
        except Exception as exc:  # surface any failure to the UI
            self.failed.emit(str(exc))

    def _read_smiles(self):
        mols, names = [], []
        with open(self.path) as fh:
            for line in fh:
                fields = line.split()
                if not fields:
                    continue
                smi = fields[0]
                name = fields[1] if len(fields) > 1 else f"mol_{len(mols) + 1}"
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mols.append(mol)
                names.append(name)
        return mols, names

    def _distance_matrix(self, fps):
        """Flat lower-triangle Tanimoto distance list for Butina."""
        n = len(fps)
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)
            if i % 50 == 0:
                # Distance matrix occupies progress 40% -> 95%.
                self.progress.emit(40 + int(55 * i / n))
        return dists


class StructureCard(QtWidgets.QFrame):
    """A single rounded card: a structure image above one or two caption lines.

    The last caption line is rendered as a monospace name; any earlier line is
    a muted bold title (used for the "Cluster N (n=...)" header on centers).
    """

    clicked = QtCore.pyqtSignal(object)       # emits this card's payload
    checkChanged = QtCore.pyqtSignal(int, bool)  # (render_idx, checked)

    def __init__(self, render_idx, caption_lines, payload=None,
                 selectable=False, checkable=False, checked=False, parent=None):
        super().__init__(parent)
        self.render_idx = render_idx
        self.caption_lines = list(caption_lines)
        self.payload = payload
        self._selectable = selectable
        self.checkbox = None

        self.setObjectName("card")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setProperty("selected", "false")
        if selectable:
            self.setCursor(QtCore.Qt.PointingHandCursor)

        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(PAD, PAD, PAD, PAD)
        box.setSpacing(2)

        if checkable:
            top = QtWidgets.QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            self.checkbox = QtWidgets.QCheckBox()
            self.checkbox.setChecked(checked)
            self.checkbox.toggled.connect(
                lambda on: self.checkChanged.emit(self.render_idx, on)
            )
            top.addWidget(self.checkbox)
            top.addStretch(1)
            box.addLayout(top)

        self.image = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        box.addWidget(self.image, 1)

        self._labels = []
        for j, _ in enumerate(self.caption_lines):
            is_name = j == len(self.caption_lines) - 1
            lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignHCenter)
            lbl.setObjectName("cardname" if is_name else "cardtitle")
            self._labels.append(lbl)
            box.addWidget(lbl)

    def caption_height(self):
        return len(self._labels) * LINE_H

    def mousePressEvent(self, event):
        if self._selectable:
            self.clicked.emit(self.payload)
        super().mousePressEvent(event)

    def set_selected(self, on):
        self.setProperty("selected", "true" if on else "false")
        # Re-apply the stylesheet so the [selected] state takes effect.
        self.style().unpolish(self)
        self.style().polish(self)

    def lay_out(self, card_w, card_h, render_fn):
        """Resize the card and (re)render its structure + captions to fit."""
        self.setFixedSize(card_w, card_h)
        top = CHECK_H if self.checkbox is not None else 0
        img_w = max(40, card_w - 2 * PAD)
        img_h = max(40, card_h - 2 * PAD - top - self.caption_height())
        self.image.setPixmap(render_fn(self.render_idx, img_w, img_h))
        for lbl, text in zip(self._labels, self.caption_lines):
            metrics = QtGui.QFontMetrics(lbl.font())
            lbl.setText(metrics.elidedText(text, QtCore.Qt.ElideRight, img_w))
            lbl.setToolTip(text)


class CardGrid(QtWidgets.QScrollArea):
    """A scrollable grid of StructureCards with a fixed column/row count.

    Cards are placed at explicit (row, col) positions in a QGridLayout, so the
    column count is exact. ``cols`` and ``rows`` set each card's size from the
    viewport; cards past ``cols`` x ``rows`` scroll vertically into view.
    """

    selected = QtCore.pyqtSignal(object)      # payload of the clicked card
    checkChanged = QtCore.pyqtSignal(int, bool)  # (render_idx, checked)

    def __init__(self, render_fn, cols=3, rows=3, selectable=False,
                 checkable=False, parent=None):
        super().__init__(parent)
        self._render = render_fn
        self._cols = max(1, cols)
        self._rows = max(1, rows)
        self._selectable = selectable
        self._checkable = checkable
        self._cards = []
        self._selected_card = None

        self.setWidgetResizable(True)  # body width tracks the viewport width
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(CARD_QSS)

        self._body = QtWidgets.QWidget()
        self._body.setObjectName("cardbody")
        self._grid = QtWidgets.QGridLayout(self._body)
        self._grid.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        self._grid.setSpacing(GAP)
        self._grid.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setWidget(self._body)

        # Coalesce the storm of resizeEvents from a window drag into one relayout.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._relayout)

    def set_dims(self, cols, rows):
        self._cols = max(1, cols)
        self._rows = max(1, rows)
        self._relayout()

    def clear(self):
        self.set_items([])

    def set_items(self, items):
        """Populate the grid. ``items`` is a list of
        (render_idx, caption_lines, payload[, checked]) tuples."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        self._selected_card = None
        for item in items:
            render_idx, caption_lines, payload = item[0], item[1], item[2]
            checked = item[3] if len(item) > 3 else False
            card = StructureCard(
                render_idx, caption_lines, payload,
                self._selectable, self._checkable, checked,
            )
            if self._selectable:
                card.clicked.connect(self._on_card_clicked)
            if self._checkable:
                card.checkChanged.connect(self.checkChanged)
            self._cards.append(card)
        self._relayout()

    def set_checked_silent(self, idx, checked):
        """Set a card's checkbox without emitting checkChanged (used to keep
        the member and selected grids in sync)."""
        for card in self._cards:
            if card.render_idx == idx and card.checkbox is not None:
                card.checkbox.blockSignals(True)
                card.checkbox.setChecked(checked)
                card.checkbox.blockSignals(False)
                return

    def select_payload(self, payload):
        """Programmatically select the first card matching ``payload``."""
        for card in self._cards:
            if card.payload == payload:
                self._select(card)
                return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._timer.start()  # relayout once the resize settles

    def _on_card_clicked(self, payload):
        for card in self._cards:
            if card.payload == payload:
                self._select(card)
                break

    def _select(self, card):
        if self._selected_card is card:
            return
        if self._selected_card is not None:
            self._selected_card.set_selected(False)
        self._selected_card = card
        card.set_selected(True)
        self.ensureWidgetVisible(card)
        self.selected.emit(card.payload)

    def _relayout(self):
        if not self._cards:
            return
        vp = self.viewport().size()
        card_w = max(
            MIN_CARD,
            (vp.width() - 2 * MARGIN - (self._cols - 1) * GAP) // self._cols,
        )
        card_h = max(
            MIN_CARD,
            (vp.height() - 2 * MARGIN - (self._rows - 1) * GAP) // self._rows,
        )
        # Detach every card, then re-place at the current column count.
        while self._grid.count():
            self._grid.takeAt(0)
        for i, card in enumerate(self._cards):
            card.lay_out(card_w, card_h, self._render)
            self._grid.addWidget(card, i // self._cols, i % self._cols)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Butina Cluster Viewer")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_QSS)

        self.mols = []
        self.names = []
        self.clusters = []
        self.cluster_of = {}        # mol index -> cluster id
        self.current_cluster = None  # cluster shown in the Members tab
        self.current_path = None     # SMILES file currently loaded
        self.worker = None
        # Structures the user has checked, kept in selection order.
        self._selected_order = []
        self._selected_set = set()
        # Cache of rendered structures keyed by (idx, w, h) so re-selecting a
        # cluster or returning to a previous size is instant.
        self._pixmap_cache = {}

        self._build_ui()

    # ---- UI construction -------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Toolbar row: open button, cutoff control, status text.
        toolbar = QtWidgets.QWidget()
        toolbar.setObjectName("toolbar")
        tb = QtWidgets.QHBoxLayout(toolbar)
        tb.setContentsMargins(8, 6, 8, 6)
        self.open_btn = QtWidgets.QPushButton("Open SMILES File...")
        self.open_btn.setObjectName("openbtn")
        self.open_btn.clicked.connect(self.open_file_dialog)
        tb.addWidget(self.open_btn)

        tb.addSpacing(8)
        tb.addWidget(QtWidgets.QLabel("Distance cutoff:"))
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.05, 0.95)
        self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setValue(DEFAULT_CUTOFF)
        self.cutoff_spin.setToolTip(
            "Tanimoto distance below which molecules join the same cluster. "
            "Changing this re-clusters the loaded file."
        )
        # Re-cluster when the threshold changes, debounced so dragging the
        # spinner doesn't kick off a run on every intermediate value.
        self._recluster_timer = QtCore.QTimer(self)
        self._recluster_timer.setSingleShot(True)
        self._recluster_timer.setInterval(300)
        self._recluster_timer.timeout.connect(self._recluster)
        self.cutoff_spin.valueChanged.connect(self._on_cutoff_changed)
        tb.addWidget(self.cutoff_spin)

        tb.addStretch(1)
        self.status_label = QtWidgets.QLabel("Open or drop a SMILES file to begin.")
        self.status_label.setStyleSheet("color: #57606a;")
        tb.addWidget(self.status_label)
        layout.addWidget(toolbar)

        # Progress bar (hidden until clustering starts).
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Two card grids side by side, each with a header + Cols/Rows controls.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(True)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #d9dee3; margin: 0 2px; }"
        )

        self.center_grid = CardGrid(self._pixmap_for, cols=2, rows=3,
                                    selectable=True)
        self.center_grid.selected.connect(self.on_center_selected)
        left_panel, self.centers_header = self._make_panel(
            "Cluster Centers", self.center_grid
        )
        splitter.addWidget(left_panel)

        # Right side: a tabbed panel with the current cluster's members and a
        # running list of the structures the user has selected.
        self.right_tabs = QtWidgets.QTabWidget()

        self.member_grid = CardGrid(self._pixmap_for, cols=3, rows=3,
                                    selectable=False, checkable=True)
        self.member_grid.checkChanged.connect(self.on_member_check)
        select_all_btn = QtWidgets.QPushButton("Select All in Cluster")
        select_all_btn.setObjectName("openbtn")
        select_all_btn.clicked.connect(self.select_all_in_cluster)
        unselect_all_btn = QtWidgets.QPushButton("Unselect All in Cluster")
        unselect_all_btn.setObjectName("openbtn")
        unselect_all_btn.clicked.connect(self.unselect_all_in_cluster)
        members_tab, self.members_header = self._make_panel(
            "Cluster Members", self.member_grid,
            extra_widgets=[select_all_btn, unselect_all_btn],
        )
        self.right_tabs.addTab(members_tab, "Members")

        self.selected_grid = CardGrid(self._pixmap_for, cols=3, rows=3,
                                      selectable=False, checkable=True)
        self.selected_grid.checkChanged.connect(self.on_selected_check)
        save_btn = QtWidgets.QPushButton("Save Selected...")
        save_btn.setObjectName("openbtn")
        save_btn.clicked.connect(self.save_selected)
        clear_btn = QtWidgets.QPushButton("Clear Selection")
        clear_btn.setObjectName("openbtn")
        clear_btn.clicked.connect(self.clear_selection)
        selected_tab, _ = self._make_panel(
            "Selected", self.selected_grid, extra_widgets=[save_btn, clear_btn]
        )
        self.selected_tab_index = self.right_tabs.addTab(selected_tab, "Selected (0)")

        splitter.addWidget(self.right_tabs)

        splitter.setSizes([620, 620])
        layout.addWidget(splitter, 1)

    def _make_panel(self, title, grid, extra_widgets=None):
        """Build a titled panel with Cols/Rows spinboxes wired to ``grid``.

        ``extra_widgets`` are inserted into the header between the title and the
        Cols/Rows spinboxes (e.g. a Save button). Returns (panel_widget,
        title_label); the caller keeps the label to update it later.
        """
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)

        # A small minimum so the splitter can drag this panel narrow; the grid
        # scrolls within whatever width it is given.
        grid.setMinimumWidth(0)
        panel.setMinimumWidth(0)

        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(f"<b>{title}</b>")
        label.setObjectName("paneltitle")
        # Ignored width lets the title shrink to nothing (and fill the gap when
        # there is room) so the header doesn't pin a large minimum panel width.
        label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                            QtWidgets.QSizePolicy.Preferred)
        header.addWidget(label, 1)

        for wdg in (extra_widgets or []):
            header.addWidget(wdg)

        cols_spin = QtWidgets.QSpinBox()
        cols_spin.setRange(1, 12)
        cols_spin.setValue(grid._cols)
        cols_spin.setPrefix("Cols: ")
        rows_spin = QtWidgets.QSpinBox()
        rows_spin.setRange(1, 12)
        rows_spin.setValue(grid._rows)
        rows_spin.setPrefix("Rows: ")

        def update_dims():
            grid.set_dims(cols_spin.value(), rows_spin.value())

        cols_spin.valueChanged.connect(update_dims)
        rows_spin.valueChanged.connect(update_dims)
        header.addWidget(cols_spin)
        header.addWidget(rows_spin)
        box.addLayout(header)
        box.addWidget(grid, 1)
        return panel, label

    # ---- File loading ----------------------------------------------------

    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SMILES File", "",
            "SMILES files (*.smi *.smiles *.txt);;All files (*)",
        )
        if path:
            self.start_clustering(path)

    # Drag-and-drop: accept a single dropped file anywhere on the window.
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.start_clustering(urls[0].toLocalFile())

    # ---- Clustering lifecycle -------------------------------------------

    def _on_cutoff_changed(self, _value):
        if self.current_path:
            self._recluster_timer.start()

    def _recluster(self):
        if not self.current_path:
            return
        if self.worker is not None and self.worker.isRunning():
            self._recluster_timer.start()  # a run is in flight; retry shortly
            return
        self.start_clustering(self.current_path)

    def start_clustering(self, path):
        if self.worker is not None and self.worker.isRunning():
            return  # ignore re-entry while a job is in flight

        self.current_path = path
        self.center_grid.clear()
        self.member_grid.clear()
        self.selected_grid.clear()
        self._selected_order = []
        self._selected_set = set()
        self.right_tabs.setTabText(self.selected_tab_index, "Selected (0)")
        self._pixmap_cache.clear()
        self.centers_header.setText("<b>Cluster Centers</b>")
        self.members_header.setText("<b>Cluster Members</b>")
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.open_btn.setEnabled(False)
        self.cutoff_spin.setEnabled(False)

        self.worker = ClusterWorker(path, self.cutoff_spin.value())
        self.worker.progress.connect(self.progress.setValue)
        self.worker.message.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.on_clustering_done)
        self.worker.failed.connect(self.on_clustering_failed)
        self.worker.start()

    def on_clustering_failed(self, msg):
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.cutoff_spin.setEnabled(True)
        self.status_label.setText("Error.")
        QtWidgets.QMessageBox.critical(self, "Clustering failed", msg)

    def on_clustering_done(self, mols, names, clusters):
        self.mols = mols
        self.names = names
        self.clusters = clusters
        self.cluster_of = {
            idx: cid for cid, members in enumerate(clusters) for idx in members
        }
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.cutoff_spin.setEnabled(True)
        self.populate_centers()

    # ---- Grid population -------------------------------------------------

    def _pixmap_for(self, idx, width, height):
        # Bucket the render size so resizing reuses cached images while staying
        # crisp (RDKit renders at the displayed resolution, not an upscale).
        w = max(40, round(width / 8) * 8)
        h = max(40, round(height / 8) * 8)
        key = (idx, w, h)
        pm = self._pixmap_cache.get(key)
        if pm is None:
            pm = render_mol(self.mols[idx], w, h)
            self._pixmap_cache[key] = pm
        return pm

    def populate_centers(self):
        items = []
        for cluster_id, members in enumerate(self.clusters):
            centroid = members[0]
            caption = [
                f"Cluster {cluster_id + 1}  ·  n={len(members)}",
                self.names[centroid],
            ]
            items.append((centroid, caption, cluster_id))
        self.center_grid.set_items(items)
        self.centers_header.setText(
            f"<b>Cluster Centers</b>  ({len(self.clusters)})"
        )
        if self.clusters:
            self.center_grid.select_payload(0)  # show the first cluster

    def on_center_selected(self, cluster_id):
        self.current_cluster = cluster_id
        members = self.clusters[cluster_id]
        self.members_header.setText(
            f"<b>Cluster {cluster_id + 1} Members</b>  ({len(members)})"
        )
        items = [
            (idx, [self.names[idx]], None, idx in self._selected_set)
            for idx in members
        ]
        self.member_grid.set_items(items)

    # ---- Selection handling ---------------------------------------------

    def on_member_check(self, idx, checked):
        """A checkbox in the Members tab toggled: add/remove from selection."""
        if checked and idx not in self._selected_set:
            self._selected_set.add(idx)
            self._selected_order.append(idx)
        elif not checked and idx in self._selected_set:
            self._selected_set.discard(idx)
            self._selected_order.remove(idx)
        self.refresh_selected_tab()

    def on_selected_check(self, idx, checked):
        """A checkbox in the Selected tab toggled: unchecking drops it (and
        syncs the matching Members card)."""
        if checked or idx not in self._selected_set:
            return
        self._selected_set.discard(idx)
        self._selected_order.remove(idx)
        self.member_grid.set_checked_silent(idx, False)
        # Rebuild after this signal unwinds, since the toggled card is removed.
        QtCore.QTimer.singleShot(0, self.refresh_selected_tab)

    def refresh_selected_tab(self):
        items = []
        for idx in self._selected_order:
            cid = self.cluster_of.get(idx, -1)
            caption = [f"Cluster {cid + 1}", self.names[idx]]
            items.append((idx, caption, None, True))
        self.selected_grid.set_items(items)
        self.right_tabs.setTabText(
            self.selected_tab_index, f"Selected ({len(items)})"
        )

    def select_all_in_cluster(self):
        """Add every member of the currently shown cluster to the selection."""
        if self.current_cluster is None:
            return
        for idx in self.clusters[self.current_cluster]:
            if idx not in self._selected_set:
                self._selected_set.add(idx)
                self._selected_order.append(idx)
            self.member_grid.set_checked_silent(idx, True)
        self.refresh_selected_tab()

    def unselect_all_in_cluster(self):
        """Remove every member of the currently shown cluster from selection."""
        if self.current_cluster is None:
            return
        for idx in self.clusters[self.current_cluster]:
            if idx in self._selected_set:
                self._selected_set.discard(idx)
                self._selected_order.remove(idx)
            self.member_grid.set_checked_silent(idx, False)
        self.refresh_selected_tab()

    def clear_selection(self):
        """Remove every structure from the selection."""
        if not self._selected_order:
            return
        for idx in list(self._selected_order):
            self.member_grid.set_checked_silent(idx, False)
        self._selected_order = []
        self._selected_set = set()
        self.refresh_selected_tab()

    def save_selected(self):
        if not self._selected_order:
            QtWidgets.QMessageBox.information(
                self, "Save Selected", "No structures are selected."
            )
            return
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Selected Structures", "selected.smi",
            "SMILES files (*.smi *.smiles);;SDF files (*.sdf)",
        )
        if not path:
            return

        # Decide the format from the explicit extension, else the chosen filter.
        low = path.lower()
        if low.endswith(".sdf"):
            fmt = "sdf"
        elif low.endswith((".smi", ".smiles")):
            fmt = "smi"
        else:
            fmt = "sdf" if "sdf" in selected_filter.lower() else "smi"
            path += ".sdf" if fmt == "sdf" else ".smi"

        if fmt == "sdf":
            self._write_sdf(path)
        else:
            self._write_smi(path)
        self.status_label.setText(
            f"Saved {len(self._selected_order)} structures to {path}"
        )

    def _write_smi(self, path):
        """SMILES file: one '<SMILES> <name> <cluster>' line per structure."""
        with open(path, "w") as fh:
            for idx in self._selected_order:
                smiles = Chem.MolToSmiles(self.mols[idx])
                cluster = self.cluster_of.get(idx, -1) + 1
                fh.write(f"{smiles} {self.names[idx]} {cluster}\n")

    def _write_sdf(self, path):
        """SDF with a 2D depiction, the name, and a Cluster property each."""
        writer = Chem.SDWriter(path)
        try:
            for idx in self._selected_order:
                mol = Chem.Mol(self.mols[idx])
                rdDepictor.Compute2DCoords(mol)
                mol.SetProp("_Name", self.names[idx])
                mol.SetProp("Cluster", str(self.cluster_of.get(idx, -1) + 1))
                writer.write(mol)
        finally:
            writer.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.start_clustering(sys.argv[1])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
