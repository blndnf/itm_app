"""Main window for Trilumin Process application."""

import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QProgressBar,
    QDialog,
    QTextEdit,
    QLabel,
    QTabWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QDialogButtonBox,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import QToolTip

from gui.widgets import (
    ImagePreview,
    SettingsPanel,
    ResultPanel,
    AbstractionSettingsPanel,
    OutlineSource,
)
from processing.outlines import OutlineExtractor
from processing.shades import ShadeQuantizer
from processing.palette import PaletteExtractor, PaletteSettings, SortMethod, PaletteMethod
from processing.abstraction import ImageAbstractor, AbstractionSettings, apply_color_boost
from utils.image_io import ImageIO


class DebugDialog(QDialog):
    """Dialog to display palette analysis debug information."""

    def __init__(self, debug_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Farbanalyse Debug")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Farbanalyse - Debug Informationen")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Text area with monospace font
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Courier New", 10))
        self._text_edit.setPlainText(debug_text)
        layout.addWidget(self._text_edit)

        # Close button
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class AdvancedSettingsDialog(QDialog):
    """Dialog for advanced color analysis settings."""

    METHOD_INFO = {
        "kmeans": ("Mini-Batch K-Means (Default)", "Schneller Clustering-Algorithmus, guter Allrounder"),
        "median_cut": ("Median Cut", "Deterministisch, gut für gleichmäßige Farbverteilung"),
        "mean_shift": ("Mean Shift", "Findet Clusteranzahl automatisch, langsamer"),
        "dbscan": ("DBSCAN", "Dichtebasiert, erkennt Ausreißer als Rauschen"),
        "octree": ("Octree Quantization", "Extrem schnell, für Echtzeit-Vorschau"),
        "hybrid": ("Hybrid (Octree → K-Means)", "Vorfilterung + Feinanalyse, beste Qualität"),
        "gmm": ("Gaussian Mixture Model", "Weiche Cluster-Zuordnung, gut für Farbverläufe"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Erweiterte Farbanalyse-Einstellungen")
        self.setMinimumSize(500, 450)

        self._settings = QSettings("Trilumin", "TrialuminProcess")
        self._setup_ui()
        self._load_settings()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Tab 1: Method Selection
        method_tab = QWidget()
        method_layout = QVBoxLayout(method_tab)

        method_group = QGroupBox("Clustering-Methode")
        method_form = QFormLayout(method_group)

        self._method_combo = QComboBox()
        for method_id, (name, tooltip) in self.METHOD_INFO.items():
            self._method_combo.addItem(name, method_id)
            idx = self._method_combo.count() - 1
            self._method_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)

        method_form.addRow("Methode:", self._method_combo)

        self._method_description = QLabel()
        self._method_description.setWordWrap(True)
        self._method_description.setStyleSheet("color: #666; font-style: italic;")
        method_form.addRow(self._method_description)

        method_layout.addWidget(method_group)

        # Cascading vs Sequential toggle
        balance_group = QGroupBox("Balancierungs-Modus")
        balance_layout = QVBoxLayout(balance_group)

        self._cascading_check = QCheckBox("Cascading Balancing (Standard)")
        self._cascading_check.setToolTip(
            "Durchkaskadieren: Nach dem Balancieren einer Familie "
            "wird geprüft ob dadurch neue Ungleichgewichte entstanden sind.\n"
            "Sequential: Erst eine Familie vollständig balancieren, dann die nächste."
        )
        self._cascading_check.setChecked(True)
        balance_layout.addWidget(self._cascading_check)

        method_layout.addWidget(balance_group)
        method_layout.addStretch()

        self._tabs.addTab(method_tab, "Methodenauswahl")

        # Tab 2: Method Parameters
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)

        # K-Means parameters
        self._kmeans_group = QGroupBox("K-Means Parameter")
        kmeans_form = QFormLayout(self._kmeans_group)

        self._kmeans_batch = QSpinBox()
        self._kmeans_batch.setRange(100, 10000)
        self._kmeans_batch.setValue(1024)
        self._kmeans_batch.setToolTip(
            "Batch Size: Anzahl der Pixel pro Trainings-Iteration.\n"
            "Höher = schneller, aber weniger präzise.\n"
            "Niedriger = langsamer, aber genauere Cluster.\n"
            "Empfohlen: 1024 für gute Balance."
        )
        kmeans_form.addRow("Batch Size:", self._kmeans_batch)

        self._kmeans_iter = QSpinBox()
        self._kmeans_iter.setRange(10, 500)
        self._kmeans_iter.setValue(100)
        self._kmeans_iter.setToolTip(
            "Maximale Anzahl der Trainings-Durchläufe.\n"
            "Mehr Iterationen = stabilere Cluster, aber langsamer.\n"
            "Bei Konvergenz stoppt der Algorithmus früher.\n"
            "Empfohlen: 100 für die meisten Bilder."
        )
        kmeans_form.addRow("Max Iterations:", self._kmeans_iter)

        self._use_fixed_seed = QCheckBox("Festen Seed verwenden")
        self._use_fixed_seed.setChecked(True)
        self._use_fixed_seed.setToolTip(
            "Mit festem Seed: Gleiche Ergebnisse bei gleichem Bild.\n"
            "Ohne: Leicht unterschiedliche Ergebnisse bei jedem Durchlauf.\n"
            "Empfohlen: AN für reproduzierbare Resultate."
        )
        kmeans_form.addRow(self._use_fixed_seed)

        self._random_seed = QSpinBox()
        self._random_seed.setRange(0, 9999)
        self._random_seed.setValue(42)
        self._random_seed.setToolTip(
            "Startwert für den Zufallsgenerator.\n"
            "Verschiedene Seeds ergeben verschiedene initiale Cluster.\n"
            "Bei ungewöhnlichen Ergebnissen: anderen Seed probieren."
        )
        kmeans_form.addRow("Random Seed:", self._random_seed)

        params_layout.addWidget(self._kmeans_group)

        # Mean Shift parameters
        self._meanshift_group = QGroupBox("Mean Shift Parameter")
        ms_form = QFormLayout(self._meanshift_group)

        self._ms_auto_bandwidth = QCheckBox("Bandwidth automatisch schätzen")
        self._ms_auto_bandwidth.setChecked(True)
        self._ms_auto_bandwidth.setToolTip(
            "Automatisch: Schätzt optimale Bandwidth aus den Daten.\n"
            "Manuell: Eigenen Wert eingeben für mehr Kontrolle.\n"
            "Empfohlen: Automatisch für die meisten Fälle."
        )
        ms_form.addRow(self._ms_auto_bandwidth)

        self._ms_bandwidth = QDoubleSpinBox()
        self._ms_bandwidth.setRange(0.1, 50.0)
        self._ms_bandwidth.setValue(30.0)
        self._ms_bandwidth.setEnabled(False)
        self._ms_bandwidth.setToolTip(
            "Radius der Farbsuche im RGB-Raum.\n"
            "Kleiner = mehr, feinere Cluster.\n"
            "Größer = weniger, gröbere Cluster.\n"
            "Typische Werte: 20-40 für Fotos."
        )
        ms_form.addRow("Bandwidth:", self._ms_bandwidth)

        self._ms_bin_seeding = QCheckBox("Bin Seeding (schneller)")
        self._ms_bin_seeding.setChecked(True)
        self._ms_bin_seeding.setToolTip(
            "Verwendet diskretisierte Startpunkte statt aller Pixel.\n"
            "AN = deutlich schneller, minimal weniger genau.\n"
            "AUS = sehr langsam, aber maximale Genauigkeit."
        )
        ms_form.addRow(self._ms_bin_seeding)

        ms_warning = QLabel("⚠️ Kann bei großen Bildern sehr langsam sein!")
        ms_warning.setStyleSheet("color: orange;")
        ms_form.addRow(ms_warning)

        params_layout.addWidget(self._meanshift_group)
        self._meanshift_group.hide()

        # DBSCAN parameters
        self._dbscan_group = QGroupBox("DBSCAN Parameter")
        db_form = QFormLayout(self._dbscan_group)

        self._db_eps = QDoubleSpinBox()
        self._db_eps.setRange(0.1, 50.0)
        self._db_eps.setValue(10.0)
        self._db_eps.setToolTip(
            "Epsilon: Maximaler Abstand zwischen Punkten im selben Cluster.\n"
            "Kleiner = mehr, kleinere Cluster (gut für Details).\n"
            "Größer = weniger, größere Cluster.\n"
            "Im Lab-Farbraum: 5-15 typisch für Fotos."
        )
        db_form.addRow("Epsilon (max Abstand):", self._db_eps)

        self._db_min_samples = QSpinBox()
        self._db_min_samples.setRange(1, 100)
        self._db_min_samples.setValue(50)
        self._db_min_samples.setToolTip(
            "Minimale Pixel für einen gültigen Cluster.\n"
            "Niedrig = auch kleine Farbflächen werden erfasst.\n"
            "Hoch = nur große, dominante Farbbereiche.\n"
            "Tipp: Niedrig (10-30) für Details, hoch (50-100) für Hauptfarben."
        )
        db_form.addRow("Min Samples:", self._db_min_samples)

        self._db_colorspace = QComboBox()
        self._db_colorspace.addItems(["Lab", "RGB"])
        self._db_colorspace.setToolTip(
            "Lab: Perzeptuell gleichmäßig (wie Menschen Farben sehen).\n"
            "RGB: Technischer Farbraum, schneller aber weniger akkurat.\n"
            "Empfohlen: Lab für bessere Farbtrennung."
        )
        db_form.addRow("Farbraum:", self._db_colorspace)

        db_info = QLabel("Clusteranzahl wird automatisch bestimmt")
        db_info.setStyleSheet("color: #666; font-style: italic;")
        db_form.addRow(db_info)

        params_layout.addWidget(self._dbscan_group)
        self._dbscan_group.hide()

        # Hybrid parameters
        self._hybrid_group = QGroupBox("Hybrid Parameter")
        hy_form = QFormLayout(self._hybrid_group)

        self._hy_prefilter = QSpinBox()
        self._hy_prefilter.setRange(32, 256)
        self._hy_prefilter.setValue(128)
        self._hy_prefilter.setToolTip(
            "Erster Schritt: Octree reduziert auf N Farben (schnell).\n"
            "Mehr = feinere Vorauswahl, aber langsamer.\n"
            "Weniger = gröbere Vorauswahl, aber schneller.\n"
            "Empfohlen: 128 für gute Balance."
        )
        hy_form.addRow("Octree Vorfilter:", self._hy_prefilter)

        self._hy_final = QSpinBox()
        self._hy_final.setRange(2, 32)
        self._hy_final.setValue(12)
        self._hy_final.setToolTip(
            "Zweiter Schritt: K-Means verfeinert auf N finale Cluster.\n"
            "Sollte der gewünschten Palettengröße entsprechen.\n"
            "Wird automatisch aus Haupteinstellung übernommen."
        )
        hy_form.addRow("Finale Cluster:", self._hy_final)

        self._hy_colorspace = QComboBox()
        self._hy_colorspace.addItems(["Lab", "RGB"])
        self._hy_colorspace.setToolTip(
            "Farbraum für den finalen K-Means Schritt.\n"
            "Lab: Bessere perzeptuelle Farbtrennung.\n"
            "RGB: Schneller, aber weniger natürliche Gruppierung."
        )
        hy_form.addRow("Farbraum (final):", self._hy_colorspace)

        params_layout.addWidget(self._hybrid_group)
        self._hybrid_group.hide()

        # GMM parameters
        self._gmm_group = QGroupBox("GMM Parameter")
        gmm_form = QFormLayout(self._gmm_group)

        self._gmm_cov = QComboBox()
        self._gmm_cov.addItems(["full", "tied", "diag", "spherical"])
        self._gmm_cov.setToolTip(
            "Art der Kovarianzmatrix für Cluster-Form:\n"
            "• full: Jeder Cluster eigene Form (am flexibelsten)\n"
            "• tied: Alle Cluster gleiche Form\n"
            "• diag: Nur achsenparallele Ellipsen\n"
            "• spherical: Nur Kugeln (am schnellsten)\n"
            "Empfohlen: 'full' für beste Qualität."
        )
        gmm_form.addRow("Covariance Type:", self._gmm_cov)

        self._gmm_iter = QSpinBox()
        self._gmm_iter.setRange(10, 500)
        self._gmm_iter.setValue(100)
        self._gmm_iter.setToolTip(
            "Maximale EM-Iterationen für Konvergenz.\n"
            "Mehr = bessere Konvergenz, aber langsamer.\n"
            "100 reicht meist für gute Ergebnisse."
        )
        gmm_form.addRow("Max Iterations:", self._gmm_iter)

        params_layout.addWidget(self._gmm_group)
        self._gmm_group.hide()

        # Median Cut / Octree info
        self._simple_group = QGroupBox("Info")
        simple_form = QFormLayout(self._simple_group)
        self._simple_info = QLabel("Verwendet PIL/Pillow Quantize")
        self._simple_info.setStyleSheet("color: #666;")
        simple_form.addRow(self._simple_info)
        params_layout.addWidget(self._simple_group)
        self._simple_group.hide()

        params_layout.addStretch()
        self._tabs.addTab(params_tab, "Parameter")

        # Time estimate label
        self._time_estimate = QLabel("Geschätzte Zeit: ~0.5 sec")
        self._time_estimate.setStyleSheet("color: #666;")
        layout.addWidget(self._time_estimate)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Zurücksetzen")
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        button_layout.addWidget(self._button_box)

        layout.addLayout(button_layout)

    def _connect_signals(self):
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._ms_auto_bandwidth.toggled.connect(
            lambda checked: self._ms_bandwidth.setEnabled(not checked)
        )
        self._use_fixed_seed.toggled.connect(self._random_seed.setEnabled)

    def _on_method_changed(self, index):
        method = self._method_combo.currentData()
        _, description = self.METHOD_INFO.get(method, ("", ""))
        self._method_description.setText(description)

        # Show/hide parameter groups
        self._kmeans_group.setVisible(method == "kmeans")
        self._meanshift_group.setVisible(method == "mean_shift")
        self._dbscan_group.setVisible(method == "dbscan")
        self._hybrid_group.setVisible(method == "hybrid")
        self._gmm_group.setVisible(method == "gmm")
        self._simple_group.setVisible(method in ("median_cut", "octree"))

        if method == "octree":
            self._simple_info.setText("Nur RGB-Farbraum, keine Lab-Unterstützung")
        else:
            self._simple_info.setText("Verwendet PIL/Pillow Quantize")

    def _load_settings(self):
        method = self._settings.value("advanced/method", "kmeans")
        idx = self._method_combo.findData(method)
        if idx >= 0:
            self._method_combo.setCurrentIndex(idx)

        self._cascading_check.setChecked(
            self._settings.value("advanced/cascading", True, type=bool)
        )

        # K-Means
        self._kmeans_batch.setValue(
            self._settings.value("advanced/kmeans/batch_size", 1024, type=int)
        )
        self._kmeans_iter.setValue(
            self._settings.value("advanced/kmeans/max_iter", 100, type=int)
        )
        self._use_fixed_seed.setChecked(
            self._settings.value("advanced/use_fixed_seed", True, type=bool)
        )
        self._random_seed.setValue(
            self._settings.value("advanced/random_seed", 42, type=int)
        )

        # Mean Shift
        self._ms_auto_bandwidth.setChecked(
            self._settings.value("advanced/mean_shift/auto_bandwidth", True, type=bool)
        )
        self._ms_bandwidth.setValue(
            self._settings.value("advanced/mean_shift/bandwidth", 30.0, type=float)
        )

        # Trigger method change to show correct panels
        self._on_method_changed(self._method_combo.currentIndex())

    def _save_settings(self):
        self._settings.setValue("advanced/method", self._method_combo.currentData())
        self._settings.setValue("advanced/cascading", self._cascading_check.isChecked())

        self._settings.setValue("advanced/kmeans/batch_size", self._kmeans_batch.value())
        self._settings.setValue("advanced/kmeans/max_iter", self._kmeans_iter.value())
        self._settings.setValue("advanced/use_fixed_seed", self._use_fixed_seed.isChecked())
        self._settings.setValue("advanced/random_seed", self._random_seed.value())

        self._settings.setValue("advanced/mean_shift/auto_bandwidth", self._ms_auto_bandwidth.isChecked())
        self._settings.setValue("advanced/mean_shift/bandwidth", self._ms_bandwidth.value())

        self._settings.setValue("advanced/dbscan/eps", self._db_eps.value())
        self._settings.setValue("advanced/dbscan/min_samples", self._db_min_samples.value())
        self._settings.setValue("advanced/dbscan/colorspace", self._db_colorspace.currentText())

        self._settings.setValue("advanced/hybrid/prefilter", self._hy_prefilter.value())
        self._settings.setValue("advanced/hybrid/final", self._hy_final.value())
        self._settings.setValue("advanced/hybrid/colorspace", self._hy_colorspace.currentText())

        self._settings.setValue("advanced/gmm/covariance", self._gmm_cov.currentText())
        self._settings.setValue("advanced/gmm/max_iter", self._gmm_iter.value())

    def _reset_defaults(self):
        self._method_combo.setCurrentIndex(0)
        self._cascading_check.setChecked(True)
        self._kmeans_batch.setValue(1024)
        self._kmeans_iter.setValue(100)
        self._use_fixed_seed.setChecked(True)
        self._random_seed.setValue(42)
        self._ms_auto_bandwidth.setChecked(True)
        self._ms_bandwidth.setValue(30.0)
        self._ms_bin_seeding.setChecked(True)
        self._db_eps.setValue(10.0)
        self._db_min_samples.setValue(50)
        self._db_colorspace.setCurrentIndex(0)
        self._hy_prefilter.setValue(128)
        self._hy_final.setValue(12)
        self._hy_colorspace.setCurrentIndex(0)
        self._gmm_cov.setCurrentIndex(0)
        self._gmm_iter.setValue(100)

    def accept(self):
        self._save_settings()
        super().accept()

    def get_method(self) -> str:
        return self._method_combo.currentData()

    def get_cascading_mode(self) -> bool:
        return self._cascading_check.isChecked()

    def get_parameters(self) -> dict:
        return {
            "method": self._method_combo.currentData(),
            "cascading": self._cascading_check.isChecked(),
            "kmeans_batch_size": self._kmeans_batch.value(),
            "kmeans_max_iter": self._kmeans_iter.value(),
            "use_fixed_seed": self._use_fixed_seed.isChecked(),
            "random_state": self._random_seed.value() if self._use_fixed_seed.isChecked() else None,
            "mean_shift_auto_bandwidth": self._ms_auto_bandwidth.isChecked(),
            "mean_shift_bandwidth": self._ms_bandwidth.value(),
            "mean_shift_bin_seeding": self._ms_bin_seeding.isChecked(),
            "dbscan_eps": self._db_eps.value(),
            "dbscan_min_samples": self._db_min_samples.value(),
            "dbscan_colorspace": self._db_colorspace.currentText(),
            "hybrid_octree_prefilter": self._hy_prefilter.value(),
            "hybrid_final_clusters": self._hy_final.value(),
            "hybrid_colorspace": self._hy_colorspace.currentText(),
            "gmm_covariance_type": self._gmm_cov.currentText(),
            "gmm_max_iter": self._gmm_iter.value(),
        }


@dataclass
class ProcessingOptions:
    """Options for the processing worker."""

    num_values: int = 5
    num_colors: int = 9
    edge_sensitivity: float = 0.5
    palette_method: PaletteMethod = PaletteMethod.DIVERSE
    balance_mode: str = "balanced"  # "balanced" or "pronounced"
    sort_method: SortMethod = SortMethod.HUE
    grays_position: str = "end"
    add_numbers: bool = True
    outline_source: OutlineSource = OutlineSource.POSTERIZED
    min_contour_length: int = 0
    max_curvature: float = 1.0
    min_contrast: int = 0
    abstraction_settings: Optional[AbstractionSettings] = None
    color_boost: bool = False
    palette_source: str = "original"  # "original" or "preprocessed"


class ProcessingWorker(QThread):
    """Worker thread for image processing including abstraction."""

    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, options: ProcessingOptions):
        super().__init__()
        self.image = image
        self.options = options

    def run(self) -> None:
        try:
            results = {}
            opts = self.options

            # Start with original image
            working_image = self.image

            # Apply color boost if enabled (before abstraction)
            if opts.color_boost:
                self.progress.emit("Wende Color Boost an...")
                working_image = apply_color_boost(working_image)

            # Apply abstraction if enabled
            if opts.abstraction_settings and opts.abstraction_settings.enabled:
                self.progress.emit("Wende Vorverarbeitung an...")
                abstractor = ImageAbstractor(opts.abstraction_settings)
                working_image = abstractor.abstract(working_image)
                results["abstracted"] = working_image
            else:
                results["abstracted"] = working_image if opts.color_boost else None

            # Process palette FIRST (needed for outlines from posterized)
            # Palette extraction source is based on user setting:
            # - "original": Uses original image (consistent palette regardless of abstraction)
            # - "preprocessed": Uses working_image (abstraction affects palette)
            self.progress.emit("Extrahiere Farbpalette...")
            palette_settings = PaletteSettings(
                num_colors=opts.num_colors,
                sort_method=opts.sort_method,
                grays_position=opts.grays_position,
                palette_method=opts.palette_method,
                balance_mode=opts.balance_mode,
            )
            palette_extractor = PaletteExtractor(palette_settings)

            # Select palette source based on user setting
            if opts.palette_source == "preprocessed":
                palette_source_image = working_image
            else:
                # Default: use original image (color boost is always applied internally)
                palette_source_image = self.image
                # Apply color boost to palette source if enabled
                if opts.color_boost:
                    palette_source_image = apply_color_boost(self.image)

            results["colors"] = palette_extractor.extract_palette(palette_source_image)
            # Store debug log from palette extraction
            results["debug_log"] = palette_extractor.get_debug_log()
            # Use working_image for posterization (shows abstraction effect)
            results["posterized"] = palette_extractor.create_posterized_image(
                working_image, add_numbers=opts.add_numbers
            )

            # Process shades
            self.progress.emit("Quantisiere Graustufen...")
            shade_quantizer = ShadeQuantizer()
            shade_quantizer.set_num_values(opts.num_values)
            results["shades"] = shade_quantizer.quantize(
                working_image, add_numbers=opts.add_numbers
            )
            results["grayscale_levels"] = shade_quantizer.get_value_levels()

            # Process outlines based on selected source
            self.progress.emit("Extrahiere Konturen...")
            outline_extractor = OutlineExtractor()
            outline_extractor.set_sensitivity(opts.edge_sensitivity)
            outline_extractor.set_min_contour_length(opts.min_contour_length)
            outline_extractor.set_max_curvature(opts.max_curvature)
            outline_extractor.set_min_contrast(opts.min_contrast)

            # Get outline source image(s)
            posterized_for_outline = palette_extractor.create_posterized_image(
                working_image, add_numbers=False
            )

            if opts.outline_source == OutlineSource.ORIGINAL:
                results["outlines"] = outline_extractor.extract(working_image)
            elif opts.outline_source == OutlineSource.POSTERIZED:
                results["outlines"] = outline_extractor.extract(
                    working_image, posterized_image=posterized_for_outline
                )
            elif opts.outline_source == OutlineSource.SHADES:
                # Use grayscale shades image for outlines
                shades_for_outline = shade_quantizer.quantize(
                    working_image, add_numbers=False
                )
                results["outlines"] = outline_extractor.extract(
                    working_image, posterized_image=shades_for_outline
                )
            elif opts.outline_source == OutlineSource.COMBINED:
                # Combine posterized and shades outlines
                outlines_poster = outline_extractor.extract(
                    working_image, posterized_image=posterized_for_outline
                )
                shades_for_outline = shade_quantizer.quantize(
                    working_image, add_numbers=False
                )
                outlines_shades = outline_extractor.extract(
                    working_image, posterized_image=shades_for_outline
                )
                # Combine: take minimum (darker = edge) of both
                results["outlines"] = np.minimum(outlines_poster, outlines_shades)

            # Create combined palette image with grayscales
            # High resolution with auto-calculated DIN A4 layout
            self.progress.emit("Erstelle Palettenbild...")
            results["palette"] = palette_extractor.create_palette_image(
                results["colors"],
                swatch_size=400,
                cols=0,  # Auto-calculate for DIN A4 ratio
                show_name=True,
                show_number=opts.add_numbers,
                grayscales=results["grayscale_levels"],
            )

            # Create composite overlay: Colors with Shades luminosity applied
            # Uses soft light blending to preserve colors in dark areas
            self.progress.emit("Erstelle Überlagerung...")
            shades_no_numbers = shade_quantizer.quantize(working_image, add_numbers=False)
            colors_no_numbers = palette_extractor.create_posterized_image(
                working_image, add_numbers=False
            )

            import cv2

            # Convert shades to single channel luminosity (0-1 range)
            if len(shades_no_numbers.shape) == 2:
                luminosity = shades_no_numbers.astype(np.float32) / 255.0
            else:
                luminosity = cv2.cvtColor(shades_no_numbers, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            # Convert colors to HSV to preserve hue and saturation
            colors_hsv = cv2.cvtColor(colors_no_numbers, cv2.COLOR_BGR2HSV).astype(np.float32)
            original_v = colors_hsv[:, :, 2] / 255.0  # Normalize to 0-1

            # IMPROVED BLEND: Soft light formula with floor
            # Soft light: if lum < 0.5: result = original - (1-2*lum) * original * (1-original)
            #             if lum >= 0.5: result = original + (2*lum-1) * (D(original) - original)
            # where D(x) = sqrt(x) for simplicity

            # Apply minimum floor to prevent pure black (lift shadows)
            min_floor = 0.15  # Minimum value - prevents pure black
            luminosity_lifted = min_floor + luminosity * (1.0 - min_floor)

            # Soft light blend between original V and shades
            mask_dark = luminosity_lifted < 0.5
            mask_light = ~mask_dark

            blended_v = np.zeros_like(original_v)

            # Dark areas: darken gently
            dark_lum = luminosity_lifted[mask_dark]
            dark_orig = original_v[mask_dark]
            blended_v[mask_dark] = dark_orig - (1 - 2 * dark_lum) * dark_orig * (1 - dark_orig)

            # Light areas: lighten gently
            light_lum = luminosity_lifted[mask_light]
            light_orig = original_v[mask_light]
            # Using sqrt for dodge function
            d_orig = np.sqrt(light_orig)
            blended_v[mask_light] = light_orig + (2 * light_lum - 1) * (d_orig - light_orig)

            # Mix: 70% soft light result, 30% direct luminosity (for structure)
            final_v = 0.7 * blended_v + 0.3 * luminosity_lifted

            # Apply to HSV
            colors_hsv[:, :, 2] = np.clip(final_v * 255.0, 0, 255)

            # Convert back to BGR
            colors_hsv = np.clip(colors_hsv, 0, 255).astype(np.uint8)
            results["composite"] = cv2.cvtColor(colors_hsv, cv2.COLOR_HSV2BGR)

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for Trilumin Process."""

    def __init__(self):
        super().__init__()
        self._source_image: Optional[np.ndarray] = None
        self._abstracted_image: Optional[np.ndarray] = None
        self._show_abstracted: bool = False
        self._results: dict = {}
        self._worker: Optional[ProcessingWorker] = None
        self._current_file_path: Optional[str] = None
        self._saved_display_index: int = 0
        self._is_new_image_load: bool = False
        self._settings_changed_since_process: bool = False
        self._debug_log: str = ""  # Store debug log from palette extraction

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Trilumin Process - Bildanalyse für Ölmalerei")
        self.setMinimumSize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Toolbar
        toolbar_layout = QHBoxLayout()

        self._open_button = QPushButton("Bild öffnen")
        self._open_button.setMinimumWidth(120)
        toolbar_layout.addWidget(self._open_button)

        # Display mode dropdown (left side)
        from PyQt6.QtWidgets import QComboBox
        self._display_combo = QComboBox()
        self._display_combo.addItem("Zeige: Quelle", "source")
        self._display_combo.addItem("Zeige: Vorverarbeitet", "abstracted")
        self._display_combo.addItem("Zeige: Composite", "composite")
        self._display_combo.setMinimumWidth(160)
        self._display_combo.setEnabled(False)
        self._display_combo.setToolTip(
            "Quelle: Originalbild\n"
            "Vorverarbeitet: Nach Abstraktion/Color Boost\n"
            "Composite: Überlagerung von Shades und Colors"
        )
        toolbar_layout.addWidget(self._display_combo)

        # Debug button
        self._debug_button = QPushButton("[debug]")
        self._debug_button.setMaximumWidth(60)
        self._debug_button.setEnabled(False)
        self._debug_button.setToolTip("Zeigt detaillierte Farbanalyse-Informationen")
        toolbar_layout.addWidget(self._debug_button)

        # Advanced settings button
        self._advanced_button = QPushButton("⚙️")
        self._advanced_button.setMaximumWidth(40)
        self._advanced_button.setToolTip("Erweiterte Farbanalyse-Einstellungen")
        toolbar_layout.addWidget(self._advanced_button)

        toolbar_layout.addStretch()

        # Update button (right side)
        self._update_button = QPushButton("Aktualisieren")
        self._update_button.setMinimumWidth(120)
        self._update_button.setEnabled(False)
        self._update_button.setToolTip(
            "Wendet geänderte Einstellungen auf das Bild an"
        )
        toolbar_layout.addWidget(self._update_button)

        self._save_all_button = QPushButton("Alle speichern")
        self._save_all_button.setMinimumWidth(120)
        self._save_all_button.setEnabled(False)
        toolbar_layout.addWidget(self._save_all_button)

        main_layout.addLayout(toolbar_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Source image and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self._source_preview = ImagePreview("Quellbild")
        left_layout.addWidget(self._source_preview, stretch=2)

        # Abstraction settings panel
        self._abstraction_panel = AbstractionSettingsPanel()
        left_layout.addWidget(self._abstraction_panel)

        # Analysis settings panel
        self._settings_panel = SettingsPanel()
        left_layout.addWidget(self._settings_panel)

        splitter.addWidget(left_panel)

        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Top row: Outlines and Shades
        top_results = QSplitter(Qt.Orientation.Horizontal)

        self._outlines_panel = ResultPanel("Konturen (Outlines)", "outlines")
        top_results.addWidget(self._outlines_panel)

        self._shades_panel = ResultPanel("Graustufen (Shades)", "shades")
        top_results.addWidget(self._shades_panel)

        right_layout.addWidget(top_results, stretch=1)

        # Bottom row: Posterized and Palette
        bottom_results = QSplitter(Qt.Orientation.Horizontal)

        self._posterized_panel = ResultPanel("Posterisiert (Colors)", "posterized")
        bottom_results.addWidget(self._posterized_panel)

        self._palette_panel = ResultPanel("Farbpalette", "palette")
        bottom_results.addWidget(self._palette_panel)

        right_layout.addWidget(bottom_results, stretch=1)

        splitter.addWidget(right_panel)

        # Set splitter sizes (1:2 ratio)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.hide()
        self._status_bar.addPermanentWidget(self._progress_bar)

        self._status_bar.showMessage("Bereit. Bitte öffnen Sie ein Bild.")

    def _connect_signals(self) -> None:
        self._open_button.clicked.connect(self._open_image)
        self._update_button.clicked.connect(self._process_image)
        self._save_all_button.clicked.connect(self._save_all)
        self._display_combo.currentIndexChanged.connect(self._on_display_changed)
        self._debug_button.clicked.connect(self._show_debug_dialog)
        self._advanced_button.clicked.connect(self._show_advanced_settings)

        self._outlines_panel.save_requested.connect(self._save_result)
        self._shades_panel.save_requested.connect(self._save_result)
        self._posterized_panel.save_requested.connect(self._save_result)
        self._palette_panel.save_requested.connect(self._save_result)

        # Settings changed signals
        self._settings_panel.settings_changed.connect(self._on_settings_changed)
        self._abstraction_panel.settings_changed.connect(self._on_settings_changed)

    def _show_debug_dialog(self) -> None:
        """Show the debug dialog with palette analysis information."""
        if self._debug_log:
            dialog = DebugDialog(self._debug_log, self)
            dialog.exec()
        else:
            QMessageBox.information(
                self,
                "Debug",
                "Keine Debug-Informationen verfügbar.\n"
                "Bitte zuerst ein Bild verarbeiten."
            )

    def _show_advanced_settings(self) -> None:
        """Show the advanced color analysis settings dialog."""
        dialog = AdvancedSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Settings were saved, trigger reprocessing if image loaded
            if self._source_image is not None:
                self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        """Handle settings changes."""
        self._settings_changed_since_process = True
        if self._results:
            self._update_button.setEnabled(True)

    def _on_display_changed(self, index: int) -> None:
        """Handle display mode dropdown change."""
        mode = self._display_combo.currentData()

        if mode == "source":
            if self._source_image is not None:
                self._source_preview.set_image(self._source_image)
        elif mode == "abstracted":
            if self._abstracted_image is not None:
                self._source_preview.set_image(self._abstracted_image)
            elif self._source_image is not None:
                self._source_preview.set_image(self._source_image)
        elif mode == "composite":
            if "composite" in self._results and self._results["composite"] is not None:
                self._source_preview.set_image(self._results["composite"])
            elif self._source_image is not None:
                self._source_preview.set_image(self._source_image)

    def _open_image(self) -> None:
        # Use source folder from settings if available
        start_folder = self._settings_panel.get_source_folder() or ""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild öffnen",
            start_folder,
            ImageIO.get_file_filter("load"),
        )

        if file_path:
            image = ImageIO.load_image(file_path)
            if image is not None:
                self._source_image = image
                self._abstracted_image = None
                self._current_file_path = file_path
                self._source_preview.set_image(image)
                self._display_combo.setCurrentIndex(0)  # Reset to "Quelle"
                self._display_combo.setEnabled(False)
                self._clear_results()

                info = ImageIO.get_image_info(file_path)
                self._status_bar.showMessage(
                    f"Geladen: {os.path.basename(file_path)} "
                    f"({info['width']}x{info['height']} px)"
                )

                # Auto-process immediately after loading (new image)
                self._process_image(is_new_image=True)
            else:
                QMessageBox.critical(
                    self,
                    "Fehler",
                    "Das Bild konnte nicht geladen werden.",
                )

    def _get_processing_options(self) -> ProcessingOptions:
        """Build processing options from current UI settings."""
        abstraction_settings = None
        if self._abstraction_panel.is_enabled():
            abstraction_settings = AbstractionSettings(
                enabled=True,
                method=self._abstraction_panel.get_method(),
                detail_level=self._abstraction_panel.get_detail_level(),
            )

        return ProcessingOptions(
            num_values=self._settings_panel.get_values(),
            num_colors=self._settings_panel.get_steps(),
            edge_sensitivity=self._settings_panel.get_edge_sensitivity(),
            palette_method=self._settings_panel.get_palette_method(),
            balance_mode=self._settings_panel.get_balance_mode(),
            sort_method=self._settings_panel.get_sort_method(),
            grays_position=self._settings_panel.get_grays_position(),
            add_numbers=self._settings_panel.should_add_numbers(),
            outline_source=self._settings_panel.get_outline_source(),
            min_contour_length=self._settings_panel.get_min_contour_length(),
            max_curvature=self._settings_panel.get_max_curvature(),
            min_contrast=self._settings_panel.get_min_contrast(),
            abstraction_settings=abstraction_settings,
            color_boost=self._abstraction_panel.is_color_boost_enabled(),
            palette_source=self._abstraction_panel.get_palette_source(),
        )

    def _process_image(self, is_new_image: bool = False) -> None:
        if self._source_image is None:
            return

        # Save current display mode index (to restore after processing)
        self._saved_display_index = self._display_combo.currentIndex()
        self._is_new_image_load = is_new_image

        # Disable UI during processing
        self._update_button.setEnabled(False)
        self._open_button.setEnabled(False)
        self._display_combo.setEnabled(False)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.show()

        # Create and start worker thread
        self._worker = ProcessingWorker(
            self._source_image,
            self._get_processing_options(),
        )
        self._worker.progress.connect(self._on_processing_progress)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.error.connect(self._on_processing_error)
        self._worker.start()

    def _on_processing_progress(self, message: str) -> None:
        self._status_bar.showMessage(message)

    def _on_processing_finished(self, results: dict) -> None:
        self._results = results
        self._settings_changed_since_process = False

        # Store debug log and enable debug button
        self._debug_log = results.get("debug_log", "")
        self._debug_button.setEnabled(bool(self._debug_log))

        # Store abstracted image if available
        if results.get("abstracted") is not None:
            self._abstracted_image = results["abstracted"]
            self._show_abstracted = True
        else:
            self._abstracted_image = None
            self._show_abstracted = False

        # Enable display dropdown (composite is now available)
        self._display_combo.setEnabled(True)

        # Determine which display mode to show
        if self._is_new_image_load:
            # New image: show "Vorverarbeitet" if available, else "Quelle"
            if self._abstracted_image is not None:
                self._display_combo.setCurrentIndex(1)  # "Vorverarbeitet"
                self._source_preview.set_image(self._abstracted_image)
            else:
                self._display_combo.setCurrentIndex(0)  # "Quelle"
                if self._source_image is not None:
                    self._source_preview.set_image(self._source_image)
        else:
            # Settings update: restore previous display mode
            target_index = self._saved_display_index

            # Validate the saved index is still valid
            if target_index == 1 and self._abstracted_image is None:
                target_index = 0  # Fall back to source if no abstracted
            elif target_index == 2 and "composite" not in self._results:
                target_index = 1 if self._abstracted_image else 0

            self._display_combo.setCurrentIndex(target_index)
            # Trigger display update
            self._on_display_changed(target_index)

        # Display results
        self._outlines_panel.set_image(results["outlines"])
        self._shades_panel.set_image(results["shades"])
        self._posterized_panel.set_image(results["posterized"])
        self._palette_panel.set_image(results["palette"], is_rgb=True)

        # Re-enable UI
        self._open_button.setEnabled(True)
        self._save_all_button.setEnabled(True)
        self._update_button.setEnabled(False)
        self._progress_bar.hide()

        # Build status message
        opts = self._get_processing_options()
        abstraction_info = ""
        if opts.abstraction_settings and opts.abstraction_settings.enabled:
            method_name = {
                "pixelate": "Pixelierung",
                "bilateral": "Bilateral",
                "mean_shift": "Mean Shift",
                "kmeans": "K-Means",
            }.get(opts.abstraction_settings.method.value, "")
            abstraction_info = f" | Vorverarbeitung: {method_name}"

        self._status_bar.showMessage(
            f"Verarbeitung abgeschlossen. "
            f"{opts.num_values} Graustufen, "
            f"{opts.num_colors} Farben"
            f"{abstraction_info}"
        )

    def _on_processing_error(self, error_message: str) -> None:
        self._open_button.setEnabled(True)
        self._update_button.setEnabled(self._settings_changed_since_process)
        self._progress_bar.hide()

        QMessageBox.critical(
            self,
            "Verarbeitungsfehler",
            f"Ein Fehler ist aufgetreten:\n{error_message}",
        )
        self._status_bar.showMessage("Verarbeitung fehlgeschlagen.")

    def _clear_results(self) -> None:
        self._results = {}
        self._abstracted_image = None
        self._show_abstracted = False
        self._settings_changed_since_process = False
        self._outlines_panel.clear()
        self._shades_panel.clear()
        self._posterized_panel.clear()
        self._palette_panel.clear()
        self._save_all_button.setEnabled(False)
        self._update_button.setEnabled(False)

    def _save_result(self, result_type: str) -> None:
        if result_type not in self._results and result_type != "posterized":
            return

        if result_type == "outlines":
            image = self._outlines_panel.get_image()
        elif result_type == "shades":
            image = self._shades_panel.get_image()
        elif result_type == "posterized":
            image = self._posterized_panel.get_image()
        elif result_type == "palette":
            image = self._palette_panel.get_image()
        else:
            return

        if image is None:
            return

        base_name = ""
        if self._current_file_path:
            base_name = os.path.splitext(os.path.basename(self._current_file_path))[0]
            base_name = f"{base_name}_"

        default_name = f"{base_name}{result_type}.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Speichern: {result_type}",
            default_name,
            ImageIO.get_file_filter("save"),
        )

        if file_path:
            is_rgb = result_type == "palette"
            if ImageIO.save_image(image, file_path, is_rgb=is_rgb):
                self._status_bar.showMessage(f"Gespeichert: {file_path}")
            else:
                QMessageBox.critical(
                    self,
                    "Speicherfehler",
                    "Das Bild konnte nicht gespeichert werden.",
                )

    def _save_all(self) -> None:
        if not self._results:
            return

        # Use export folder from settings - if set and valid, export directly
        export_folder = self._settings_panel.get_export_folder()
        if export_folder and os.path.isdir(export_folder):
            directory = export_folder
        else:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Speicherort auswählen",
                export_folder or "",
            )
            if not directory:
                return

        base_name = "trilumin"
        if self._current_file_path:
            base_name = os.path.splitext(os.path.basename(self._current_file_path))[0]

        saved_count = 0
        errors = []

        result_map = [
            ("outlines", self._outlines_panel.get_image(), False),
            ("shades", self._shades_panel.get_image(), False),
            ("posterized", self._posterized_panel.get_image(), False),
            ("palette", self._palette_panel.get_image(), True),
        ]

        # Add abstracted image if available (stored in BGR format like source)
        if self._abstracted_image is not None:
            result_map.insert(0, ("abstracted", self._abstracted_image, False))

        # Add composite overlay (shades + colors blend)
        if self._results.get("composite") is not None:
            result_map.append(("composite", self._results["composite"], False))

        for name, image, is_rgb in result_map:
            if image is not None:
                file_path = os.path.join(directory, f"{base_name}_{name}.png")
                if ImageIO.save_image(image, file_path, is_rgb=is_rgb):
                    saved_count += 1
                else:
                    errors.append(name)

        if errors:
            QMessageBox.warning(
                self,
                "Teilweise gespeichert",
                f"{saved_count} Bilder gespeichert.\n"
                f"Fehler bei: {', '.join(errors)}",
            )
        else:
            self._status_bar.showMessage(
                f"Alle {saved_count} Bilder gespeichert in: {directory}"
            )
            QMessageBox.information(
                self,
                "Gespeichert",
                f"Alle {saved_count} Bilder wurden erfolgreich gespeichert.",
            )

    def closeEvent(self, event) -> None:
        """Save settings when closing the window."""
        self._settings_panel.save_settings()
        super().closeEvent(event)
