"""Main window for Trilumin Process application."""

import os
from typing import Optional

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
    QApplication,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from .widgets import ImagePreview, SettingsPanel, ResultPanel
from ..processing.outlines import OutlineExtractor
from ..processing.shades import ShadeQuantizer
from ..processing.palette import PaletteExtractor
from ..utils.image_io import ImageIO


class ProcessingWorker(QThread):
    """Worker thread for image processing."""

    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        image: np.ndarray,
        num_values: int,
        num_colors: int,
        edge_sensitivity: float,
    ):
        super().__init__()
        self.image = image
        self.num_values = num_values
        self.num_colors = num_colors
        self.edge_sensitivity = edge_sensitivity

    def run(self) -> None:
        try:
            results = {}

            # Process outlines
            self.progress.emit("Extrahiere Konturen...")
            outline_extractor = OutlineExtractor()
            outline_extractor.set_sensitivity(self.edge_sensitivity)
            results["outlines"] = outline_extractor.extract(self.image)

            # Process shades
            self.progress.emit("Quantisiere Graustufen...")
            shade_quantizer = ShadeQuantizer()
            shade_quantizer.set_num_values(self.num_values)
            results["shades"] = shade_quantizer.quantize(self.image)

            # Process palette
            self.progress.emit("Extrahiere Farbpalette...")
            palette_extractor = PaletteExtractor()
            palette_extractor.set_num_colors(self.num_colors)
            results["colors"] = palette_extractor.extract_palette(self.image)
            results["posterized"] = palette_extractor.create_posterized_image(self.image)
            results["palette"] = palette_extractor.create_palette_image(results["colors"])

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for Trilumin Process."""

    def __init__(self):
        super().__init__()
        self._source_image: Optional[np.ndarray] = None
        self._results: dict = {}
        self._worker: Optional[ProcessingWorker] = None
        self._current_file_path: Optional[str] = None

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

        self._process_button = QPushButton("Verarbeiten")
        self._process_button.setMinimumWidth(120)
        self._process_button.setEnabled(False)
        toolbar_layout.addWidget(self._process_button)

        toolbar_layout.addStretch()

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
        self._process_button.clicked.connect(self._process_image)
        self._save_all_button.clicked.connect(self._save_all)

        self._outlines_panel.save_requested.connect(self._save_result)
        self._shades_panel.save_requested.connect(self._save_result)
        self._posterized_panel.save_requested.connect(self._save_result)
        self._palette_panel.save_requested.connect(self._save_result)

    def _open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild öffnen",
            "",
            ImageIO.get_file_filter("load"),
        )

        if file_path:
            image = ImageIO.load_image(file_path)
            if image is not None:
                self._source_image = image
                self._current_file_path = file_path
                self._source_preview.set_image(image)
                self._process_button.setEnabled(True)
                self._clear_results()

                # Get image info
                info = ImageIO.get_image_info(file_path)
                self._status_bar.showMessage(
                    f"Geladen: {os.path.basename(file_path)} "
                    f"({info['width']}x{info['height']} px)"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Fehler",
                    "Das Bild konnte nicht geladen werden.",
                )

    def _process_image(self) -> None:
        if self._source_image is None:
            return

        # Disable UI during processing
        self._process_button.setEnabled(False)
        self._open_button.setEnabled(False)
        self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.show()

        # Create and start worker thread
        self._worker = ProcessingWorker(
            self._source_image,
            self._settings_panel.get_values(),
            self._settings_panel.get_steps(),
            self._settings_panel.get_edge_sensitivity(),
        )
        self._worker.progress.connect(self._on_processing_progress)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.error.connect(self._on_processing_error)
        self._worker.start()

    def _on_processing_progress(self, message: str) -> None:
        self._status_bar.showMessage(message)

    def _on_processing_finished(self, results: dict) -> None:
        self._results = results

        # Display results
        self._outlines_panel.set_image(results["outlines"])
        self._shades_panel.set_image(results["shades"])
        self._posterized_panel.set_image(results["posterized"])
        self._palette_panel.set_image(results["palette"], is_rgb=True)

        # Re-enable UI
        self._process_button.setEnabled(True)
        self._open_button.setEnabled(True)
        self._save_all_button.setEnabled(True)
        self._progress_bar.hide()

        self._status_bar.showMessage(
            f"Verarbeitung abgeschlossen. "
            f"{self._settings_panel.get_values()} Graustufen, "
            f"{self._settings_panel.get_steps()} Farben."
        )

    def _on_processing_error(self, error_message: str) -> None:
        self._process_button.setEnabled(True)
        self._open_button.setEnabled(True)
        self._progress_bar.hide()

        QMessageBox.critical(
            self,
            "Verarbeitungsfehler",
            f"Ein Fehler ist aufgetreten:\n{error_message}",
        )
        self._status_bar.showMessage("Verarbeitung fehlgeschlagen.")

    def _clear_results(self) -> None:
        self._results = {}
        self._outlines_panel.clear()
        self._shades_panel.clear()
        self._posterized_panel.clear()
        self._palette_panel.clear()
        self._save_all_button.setEnabled(False)

    def _save_result(self, result_type: str) -> None:
        if result_type not in self._results and result_type != "posterized":
            return

        # Get the image to save
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

        # Generate default filename
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
            # Palette image is RGB, others are BGR or grayscale
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

        # Ask for directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Speicherort auswählen",
            "",
        )

        if not directory:
            return

        # Generate base filename
        base_name = "trilumin"
        if self._current_file_path:
            base_name = os.path.splitext(os.path.basename(self._current_file_path))[0]

        saved_count = 0
        errors = []

        # Save each result
        result_map = [
            ("outlines", self._outlines_panel.get_image(), False),
            ("shades", self._shades_panel.get_image(), False),
            ("posterized", self._posterized_panel.get_image(), False),
            ("palette", self._palette_panel.get_image(), True),
        ]

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
