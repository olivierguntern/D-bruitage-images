"""
Fenêtre principale du logiciel de débruitage d'image.

Layout :
  ┌────────────────────────────────────────────────────┐
  │  Barre de menus + barre d'outils                  │
  ├──────────────────────────┬─────────────────────────┤
  │                          │  Panneau de contrôle    │
  │   SplitImageViewer       │  ─ Sélection algo       │
  │   (avant / après)        │  ─ Paramètres           │
  │                          │  ─ Métriques            │
  ├──────────────────────────┴─────────────────────────┤
  │  Histogramme                                       │
  ├────────────────────────────────────────────────────┤
  │  Barre de statut                                   │
  └────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QIcon, QKeySequence, QFont, QColor, QPalette
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QAction, QFileDialog, QLabel, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QGroupBox, QPushButton, QProgressBar, QStatusBar,
    QCheckBox, QApplication, QSizePolicy, QFrame, QScrollArea,
    QMessageBox, QTabWidget, QFormLayout, QToolBar, QStyleFactory,
)

from denoiser.algorithms import DenoiseAlgorithm, DenoiseResult, denoise, get_algorithm_names
from denoiser.deep_model import DnCNNDenoiser
from .image_viewer import SplitImageViewer, HistogramWidget


# ──────────────────────────────────────────────────────────────
# Worker thread pour le débruitage asynchrone
# ──────────────────────────────────────────────────────────────

class DenoiseWorker(QObject):
    finished = pyqtSignal(object)   # DenoiseResult
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)      # 0-100

    def __init__(self, image, algorithm, params, dncnn):
        super().__init__()
        self.image = image
        self.algorithm = algorithm
        self.params = params
        self.dncnn = dncnn

    def run(self):
        try:
            result = denoise(self.image, self.algorithm, self.params, self.dncnn)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────
# Fenêtre principale
# ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    APP_TITLE   = "D-Bruitage Images"
    APP_VERSION = "1.0.0"
    SUPPORTED_EXTENSIONS = (
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.ppm *.pgm);;"
        "Tous les fichiers (*)"
    )

    def __init__(self):
        super().__init__()

        self._orig_image: Optional[np.ndarray] = None    # uint8 RGB
        self._proc_image: Optional[np.ndarray] = None
        self._current_path: Optional[str] = None
        self._dncnn: Optional[DnCNNDenoiser] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[DenoiseWorker] = None

        self._setup_style()
        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()

        self.setWindowTitle(f"{self.APP_TITLE} — v{self.APP_VERSION}")
        self.resize(1280, 780)
        self.statusBar().showMessage("Prêt.")

    # ── Style ──────────────────────────────────────────────────

    def _setup_style(self):
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        palette = QPalette()
        dark = QColor(35, 35, 40)
        mid  = QColor(50, 50, 58)
        text = QColor(220, 220, 225)
        accent = QColor(70, 130, 200)
        palette.setColor(QPalette.Window,          dark)
        palette.setColor(QPalette.WindowText,      text)
        palette.setColor(QPalette.Base,            QColor(25, 25, 30))
        palette.setColor(QPalette.AlternateBase,   mid)
        palette.setColor(QPalette.ToolTipBase,     dark)
        palette.setColor(QPalette.ToolTipText,     text)
        palette.setColor(QPalette.Text,            text)
        palette.setColor(QPalette.Button,          mid)
        palette.setColor(QPalette.ButtonText,      text)
        palette.setColor(QPalette.BrightText,      Qt.red)
        palette.setColor(QPalette.Link,            accent)
        palette.setColor(QPalette.Highlight,       accent)
        palette.setColor(QPalette.HighlightedText, Qt.white)
        QApplication.setPalette(palette)

    # ── Widgets ────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Splitter horizontal : viewer | panneau droit
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # ── Viewer ────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self.viewer = SplitImageViewer()
        left_layout.addWidget(self.viewer, stretch=1)

        self.histogram = HistogramWidget()
        left_layout.addWidget(self.histogram)

        splitter.addWidget(left)

        # ── Panneau de contrôle ───────────────────────────────
        right_panel = self._build_control_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 340])
        splitter.setCollapsible(1, False)

        # ── Barre de progression ──────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ─ Algorithme ─────────────────────────────────────────
        algo_group = QGroupBox("Algorithme")
        algo_layout = QVBoxLayout(algo_group)

        self.algo_combo = QComboBox()
        for name in get_algorithm_names():
            self.algo_combo.addItem(name)
        self.algo_combo.setCurrentText(DenoiseAlgorithm.BILATERAL.value)
        algo_layout.addWidget(self.algo_combo)

        self.algo_desc = QLabel()
        self.algo_desc.setWordWrap(True)
        self.algo_desc.setStyleSheet("color: #8ab; font-size: 11px;")
        algo_layout.addWidget(self.algo_desc)

        layout.addWidget(algo_group)

        # ─ Paramètres (onglets) ───────────────────────────────
        params_group = QGroupBox("Paramètres")
        params_layout = QVBoxLayout(params_group)

        self.params_tabs = QTabWidget()
        self.params_tabs.setTabPosition(QTabWidget.North)

        # Construit un onglet par algorithme
        self._param_widgets: dict[str, dict] = {}
        for algo in DenoiseAlgorithm:
            tab, widgets = self._build_param_tab(algo)
            self.params_tabs.addTab(tab, "")   # libellé mis à jour plus bas
            self._param_widgets[algo.value] = widgets

        params_layout.addWidget(self.params_tabs)
        layout.addWidget(params_group)

        # ─ Bruit de test ──────────────────────────────────────
        noise_group = QGroupBox("Ajouter du bruit (test)")
        noise_layout = QHBoxLayout(noise_group)

        self.noise_type = QComboBox()
        self.noise_type.addItems(["Gaussien", "Sel & Poivre", "Speckle", "Poisson"])
        noise_layout.addWidget(self.noise_type)

        self.noise_level = QSlider(Qt.Horizontal)
        self.noise_level.setRange(1, 100)
        self.noise_level.setValue(25)
        self.noise_level.setToolTip("Intensité du bruit")
        noise_layout.addWidget(self.noise_level)

        self.noise_level_label = QLabel("25")
        self.noise_level_label.setFixedWidth(28)
        noise_layout.addWidget(self.noise_level_label)

        self.btn_add_noise = QPushButton("Ajouter")
        self.btn_add_noise.setFixedWidth(70)
        noise_layout.addWidget(self.btn_add_noise)

        layout.addWidget(noise_group)

        # ─ Boutons d'action ───────────────────────────────────
        self.btn_denoise = QPushButton("Débruiter")
        self.btn_denoise.setMinimumHeight(38)
        self.btn_denoise.setStyleSheet(
            "QPushButton { background-color: #2a6cb3; font-size: 14px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a80cc; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.btn_denoise.setEnabled(False)
        layout.addWidget(self.btn_denoise)

        self.btn_reset = QPushButton("Réinitialiser")
        self.btn_reset.setEnabled(False)
        layout.addWidget(self.btn_reset)

        # ─ Métriques ──────────────────────────────────────────
        metrics_group = QGroupBox("Métriques qualité")
        metrics_layout = QFormLayout(metrics_group)

        self.lbl_psnr = QLabel("—")
        self.lbl_ssim = QLabel("—")
        self.lbl_time = QLabel("—")
        self.lbl_size = QLabel("—")

        for label_text, widget in [
            ("PSNR :", self.lbl_psnr),
            ("SSIM :", self.lbl_ssim),
            ("Temps :", self.lbl_time),
            ("Taille :", self.lbl_size),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #aaa;")
            metrics_layout.addRow(lbl, widget)

        layout.addWidget(metrics_group)
        layout.addStretch()

        # ─ Export ─────────────────────────────────────────────
        self.btn_export = QPushButton("Exporter l'image traitée…")
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)

        # Synchronise l'onglet de paramètres avec l'algo sélectionné
        self._on_algo_changed(self.algo_combo.currentText())

        return panel

    def _build_param_tab(self, algo: DenoiseAlgorithm):
        """Retourne (widget_tab, dict_of_param_widgets) pour un algorithme."""
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(4, 4, 4, 4)
        widgets = {}

        def add_slider(key, label, lo, hi, val, step=1, decimals=0, tooltip=""):
            if decimals > 0:
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)
                sb.setSingleStep(step)
                sb.setValue(val)
                sb.setDecimals(decimals)
                sb.setToolTip(tooltip)
                widgets[key] = sb
                form.addRow(label, sb)
            else:
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                sl = QSlider(Qt.Horizontal)
                sl.setRange(int(lo), int(hi))
                sl.setValue(int(val))
                sl.setToolTip(tooltip)
                lbl = QLabel(str(int(val)))
                lbl.setFixedWidth(36)
                sl.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
                rl.addWidget(sl)
                rl.addWidget(lbl)
                widgets[key] = sl
                form.addRow(label, row)

        if algo == DenoiseAlgorithm.GAUSSIAN:
            add_slider("sigma", "Sigma :", 0.1, 10.0, 1.5, 0.1, 2,
                       "Écart-type du noyau gaussien")

        elif algo == DenoiseAlgorithm.MEDIAN:
            add_slider("radius", "Rayon :", 1, 10, 3, tooltip="Rayon du voisinage (1→3×3…)")

        elif algo == DenoiseAlgorithm.BILATERAL:
            add_slider("d", "Diamètre :", 3, 25, 9,
                       tooltip="Diamètre du voisinage (impair)")
            add_slider("sigma_color", "σ couleur :", 1.0, 200.0, 75.0, 1.0, 1,
                       tooltip="Filtre en intensité")
            add_slider("sigma_space", "σ espace :", 1.0, 200.0, 75.0, 1.0, 1,
                       tooltip="Filtre spatial")

        elif algo == DenoiseAlgorithm.NLM:
            add_slider("h", "h :", 0.1, 3.0, 0.6, 0.05, 2,
                       tooltip="Paramètre de filtrage (0.4–1.2)")
            add_slider("patch_size", "Patch :", 3, 15, 7,
                       tooltip="Taille du patch (pixels)")
            add_slider("patch_distance", "Distance :", 5, 31, 11,
                       tooltip="Rayon de recherche")
            cb = QCheckBox("Mode rapide")
            cb.setChecked(True)
            widgets["fast_mode"] = cb
            form.addRow("", cb)

        elif algo == DenoiseAlgorithm.WAVELET:
            wt_combo = QComboBox()
            wt_combo.addItems(["db1", "db2", "db4", "sym4", "sym6", "bior1.3", "coif1"])
            widgets["wavelet"] = wt_combo
            form.addRow("Ondelette :", wt_combo)

            mode_combo = QComboBox()
            mode_combo.addItems(["soft", "hard"])
            widgets["mode"] = mode_combo
            form.addRow("Mode seuil :", mode_combo)

            add_slider("wavelet_levels", "Niveaux :", 1, 8, 4,
                       tooltip="Nombre de niveaux de décomposition")

        elif algo == DenoiseAlgorithm.TV:
            add_slider("weight", "Poids :", 0.01, 0.5, 0.1, 0.01, 3,
                       tooltip="Plus grand → plus lisse (0.05–0.3)")
            add_slider("max_iter", "Itérations :", 50, 500, 200,
                       tooltip="Nombre max. d'itérations")

        elif algo == DenoiseAlgorithm.BM3D_LIKE:
            add_slider("sigma", "Sigma bruit :", 1.0, 100.0, 25.0, 0.5, 1,
                       tooltip="Niveau de bruit estimé (0–255)")
            add_slider("patch_size", "Patch :", 4, 16, 8,
                       tooltip="Taille des blocs")
            add_slider("patch_distance", "Distance :", 5, 25, 13,
                       tooltip="Rayon de recherche de blocs similaires")

        elif algo == DenoiseAlgorithm.DNCNN:
            add_slider("tile_size", "Taille tuile :", 64, 512, 128,
                       tooltip="Taille des tuiles (mémoire)")
            add_slider("overlap", "Recouvrement :", 8, 64, 16,
                       tooltip="Recouvrement entre tuiles")
            lbl_warn = QLabel(
                "⚠ Sans poids pré-entraînés, le résultat\n"
                "est représentatif mais non optimal."
            )
            lbl_warn.setStyleSheet("color: #e8a020; font-size: 11px;")
            lbl_warn.setWordWrap(True)
            form.addRow(lbl_warn)

        return tab, widgets

    # ── Menus ──────────────────────────────────────────────────

    def _build_menus(self):
        mb = self.menuBar()

        # Fichier
        file_menu = mb.addMenu("&Fichier")

        act_open = QAction("&Ouvrir…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_image)
        file_menu.addAction(act_open)

        act_save = QAction("&Exporter le résultat…", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self.export_result)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_batch = QAction("Traitement par &lots…", self)
        act_batch.triggered.connect(self.batch_process)
        file_menu.addAction(act_batch)

        file_menu.addSeparator()
        act_quit = QAction("&Quitter", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Vue
        view_menu = mb.addMenu("&Vue")

        act_fit = QAction("Adapter à la fenêtre", self)
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(self.viewer.fit_to_window)
        view_menu.addAction(act_fit)

        act_reset_view = QAction("Vue 100%", self)
        act_reset_view.setShortcut("Ctrl+1")
        act_reset_view.triggered.connect(self.viewer.reset_view)
        view_menu.addAction(act_reset_view)

        # Aide
        help_menu = mb.addMenu("&Aide")
        act_about = QAction("À &propos…", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_toolbar(self):
        tb = QToolBar("Outils")
        tb.setMovable(False)
        self.addToolBar(tb)

        btn_open = QPushButton("Ouvrir")
        btn_open.clicked.connect(self.open_image)
        tb.addWidget(btn_open)

        tb.addSeparator()

        btn_fit = QPushButton("Adapter")
        btn_fit.clicked.connect(self.viewer.fit_to_window)
        tb.addWidget(btn_fit)

        tb.addSeparator()

        tb.addWidget(QLabel("  Algorithme : "))
        self.toolbar_algo = QComboBox()
        for name in get_algorithm_names():
            self.toolbar_algo.addItem(name)
        self.toolbar_algo.setCurrentText(DenoiseAlgorithm.BILATERAL.value)
        self.toolbar_algo.setMinimumWidth(180)
        tb.addWidget(self.toolbar_algo)

        tb.addSeparator()
        btn_run = QPushButton("▶  Débruiter")
        btn_run.setStyleSheet("background-color: #2a6cb3; font-weight: bold; padding: 4px 12px;")
        btn_run.clicked.connect(self.run_denoise)
        tb.addWidget(btn_run)

    # ── Connexions de signaux ──────────────────────────────────

    def _connect_signals(self):
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        self.toolbar_algo.currentTextChanged.connect(self._on_toolbar_algo_changed)
        self.btn_denoise.clicked.connect(self.run_denoise)
        self.btn_reset.clicked.connect(self.reset_result)
        self.btn_export.clicked.connect(self.export_result)
        self.btn_add_noise.clicked.connect(self.add_noise)
        self.noise_level.valueChanged.connect(
            lambda v: self.noise_level_label.setText(str(v))
        )

    # ── Slots ──────────────────────────────────────────────────

    def _on_algo_changed(self, name: str):
        """Synchronise l'onglet de paramètres et la description."""
        try:
            algo = DenoiseAlgorithm(name)
            idx = list(DenoiseAlgorithm).index(algo)
            self.params_tabs.setCurrentIndex(idx)
        except ValueError:
            pass
        self.algo_desc.setText(self._algo_description(name))
        self.toolbar_algo.blockSignals(True)
        self.toolbar_algo.setCurrentText(name)
        self.toolbar_algo.blockSignals(False)

    def _on_toolbar_algo_changed(self, name: str):
        self.algo_combo.blockSignals(True)
        self.algo_combo.setCurrentText(name)
        self.algo_combo.blockSignals(False)
        self._on_algo_changed(name)

    def _algo_description(self, name: str) -> str:
        descriptions = {
            DenoiseAlgorithm.GAUSSIAN.value:
                "Filtre passe-bas rapide. Efficace pour bruit gaussien léger.",
            DenoiseAlgorithm.MEDIAN.value:
                "Filtre médian. Excellent pour le bruit impulsionnel (sel & poivre).",
            DenoiseAlgorithm.BILATERAL.value:
                "Préserve les contours en filtrant spatialement ET radiométriquement.",
            DenoiseAlgorithm.NLM.value:
                "Non-Local Means : exploite la redondance globale de l'image. Haute qualité.",
            DenoiseAlgorithm.WAVELET.value:
                "Seuillage par ondelettes. Bon compromis vitesse / qualité.",
            DenoiseAlgorithm.TV.value:
                "Variation totale. Préserve les contours, effet légèrement cartoon.",
            DenoiseAlgorithm.BM3D_LIKE.value:
                "Double passe NLM avec pondération Wiener. Proche BM3D sans dépendance externe.",
            DenoiseAlgorithm.DNCNN.value:
                "Réseau de neurones convolutionnel profond (DnCNN). Meilleure qualité avec poids pré-entraînés.",
        }
        return descriptions.get(name, "")

    def _get_current_params(self) -> dict:
        """Lit les paramètres de l'onglet courant."""
        name = self.algo_combo.currentText()
        widgets = self._param_widgets.get(name, {})
        params = {}
        for key, widget in widgets.items():
            if isinstance(widget, (QSlider, QSpinBox)):
                params[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentText()
        return params

    # ── Ouverture image ────────────────────────────────────────

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "", self.SUPPORTED_EXTENSIONS
        )
        if not path:
            return
        self._load_image(path)

    def _load_image(self, path: str):
        import cv2
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            self.statusBar().showMessage(f"Impossible d'ouvrir : {path}")
            return
        self._orig_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._proc_image = None
        self._current_path = path

        self.viewer.set_original(self._orig_image)
        self.viewer.set_processed(None)
        self.viewer.fit_to_window()
        self.histogram.set_image(self._orig_image)

        h, w = self._orig_image.shape[:2]
        self.lbl_size.setText(f"{w}×{h} px")
        self.lbl_psnr.setText("—")
        self.lbl_ssim.setText("—")
        self.lbl_time.setText("—")

        self.btn_denoise.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_export.setEnabled(False)

        filename = Path(path).name
        self.setWindowTitle(f"{self.APP_TITLE} — {filename}")
        self.statusBar().showMessage(f"Image chargée : {filename}  ({w}×{h})")

    # ── Débruitage ─────────────────────────────────────────────

    def run_denoise(self):
        if self._orig_image is None:
            self.open_image()
            return
        if self._thread and self._thread.isRunning():
            return

        algo = self.algo_combo.currentText()
        params = self._get_current_params()

        # Initialise DnCNN à la demande
        if algo == DenoiseAlgorithm.DNCNN.value and self._dncnn is None:
            self.statusBar().showMessage("Initialisation du modèle DnCNN…")
            QApplication.processEvents()
            self._dncnn = DnCNNDenoiser()

        self.btn_denoise.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.statusBar().showMessage(f"Débruitage en cours ({algo})…")

        self._worker = DenoiseWorker(self._orig_image, algo, params, self._dncnn)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_denoise_done)
        self._worker.error.connect(self._on_denoise_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_denoise_done(self, result: DenoiseResult):
        self._proc_image = result.image
        self.viewer.set_processed(self._proc_image)
        self.histogram.set_image(self._proc_image)

        # Métriques
        result.compute_metrics(self._orig_image)
        self.lbl_time.setText(f"{result.elapsed_ms:.0f} ms")
        if result.psnr_value:
            self.lbl_psnr.setText(f"{result.psnr_value:.2f} dB")
        if result.ssim_value:
            self.lbl_ssim.setText(f"{result.ssim_value:.4f}")

        self.btn_denoise.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress_bar.hide()
        self.statusBar().showMessage(
            f"Terminé — {result.algorithm}  |  "
            f"PSNR : {result.psnr_value:.2f} dB  |  "
            f"SSIM : {result.ssim_value:.4f}  |  "
            f"{result.elapsed_ms:.0f} ms"
        )

    def _on_denoise_error(self, msg: str):
        self.btn_denoise.setEnabled(True)
        self.progress_bar.hide()
        self.statusBar().showMessage(f"Erreur : {msg}")
        QMessageBox.critical(self, "Erreur de débruitage", msg)

    # ── Bruit de test ──────────────────────────────────────────

    def add_noise(self):
        if self._orig_image is None:
            return
        level = self.noise_level.value()
        noise_type = self.noise_type.currentText()

        img = self._orig_image.astype(np.float32) / 255.0

        if noise_type == "Gaussien":
            sigma = level / 255.0
            noisy = img + np.random.normal(0, sigma, img.shape).astype(np.float32)
        elif noise_type == "Sel & Poivre":
            noisy = img.copy()
            mask = np.random.rand(*img.shape[:2])
            prob = level / 200.0
            noisy[mask < prob] = 0.0
            noisy[mask > 1 - prob] = 1.0
        elif noise_type == "Speckle":
            noisy = img + img * np.random.normal(0, level / 255.0, img.shape).astype(np.float32)
        elif noise_type == "Poisson":
            peak = max(1, 255 - level * 2)
            noisy = np.random.poisson(img * peak) / peak

        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        self._orig_image = noisy
        self._proc_image = None
        self.viewer.set_original(self._orig_image)
        self.viewer.set_processed(None)
        self.histogram.set_image(self._orig_image)
        self.lbl_psnr.setText("—")
        self.lbl_ssim.setText("—")
        self.statusBar().showMessage(f"Bruit {noise_type} ajouté (niveau {level})")

    # ── Réinitialisation ───────────────────────────────────────

    def reset_result(self):
        if self._current_path:
            self._load_image(self._current_path)
        else:
            self._proc_image = None
            self.viewer.set_processed(None)
        self.statusBar().showMessage("Réinitialisé.")

    # ── Export ─────────────────────────────────────────────────

    def export_result(self):
        if self._proc_image is None:
            return
        default_name = ""
        if self._current_path:
            p = Path(self._current_path)
            default_name = str(p.parent / (p.stem + "_debruite" + p.suffix))

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter l'image traitée", default_name,
            "Images PNG (*.png);;Images JPEG (*.jpg *.jpeg);;Tous les fichiers (*)"
        )
        if not path:
            return

        import cv2
        bgr = cv2.cvtColor(self._proc_image, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, bgr)
        if ok:
            self.statusBar().showMessage(f"Image exportée : {path}")
        else:
            QMessageBox.critical(self, "Erreur", f"Impossible d'écrire : {path}")

    # ── Traitement par lots ────────────────────────────────────

    def batch_process(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner des images", "", self.SUPPORTED_EXTENSIONS
        )
        if not files:
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Dossier de sortie"
        )
        if not out_dir:
            return

        algo = self.algo_combo.currentText()
        params = self._get_current_params()

        import cv2
        total = len(files)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        errors = []
        for i, fpath in enumerate(files):
            self.statusBar().showMessage(f"Traitement {i+1}/{total} : {Path(fpath).name}")
            QApplication.processEvents()

            img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img_bgr is None:
                errors.append(fpath)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            try:
                result = denoise(img_rgb, algo, params,
                                 self._dncnn if algo == DenoiseAlgorithm.DNCNN.value else None)
                out_path = Path(out_dir) / (Path(fpath).stem + "_debruite.png")
                out_bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), out_bgr)
            except Exception as exc:
                errors.append(f"{fpath}: {exc}")

            self.progress_bar.setValue(i + 1)

        self.progress_bar.hide()
        msg = f"{total - len(errors)}/{total} images traitées."
        if errors:
            msg += f"\nErreurs : {len(errors)}"
        self.statusBar().showMessage(msg)
        QMessageBox.information(self, "Traitement par lots terminé", msg)

    # ── À propos ───────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(
            self,
            f"À propos — {self.APP_TITLE}",
            f"<h2>{self.APP_TITLE} v{self.APP_VERSION}</h2>"
            "<p>Logiciel de débruitage d'image haute performance.</p>"
            "<p><b>Algorithmes disponibles :</b><br>"
            "Gaussien · Médian · Bilatéral · Non-Local Means<br>"
            "Ondelettes · Total Variation · BM3D-like · DnCNN</p>"
            "<p><b>Technologies :</b> Python · PyQt5 · OpenCV · scikit-image · NumPy</p>"
        )
