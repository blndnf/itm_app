"""Custom widgets for Trilumin Process GUI."""

from enum import Enum
from typing import Optional, Callable

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QImage, QPixmap

from processing.abstraction import AbstractionMethod
from processing.palette import SortMethod, PaletteMethod


class OutlineSource(Enum):
    """Source for outline extraction."""
    ORIGINAL = "original"
    POSTERIZED = "posterized"
    SHADES = "shades"
    COMBINED = "combined"


class ZoomWindow(QWidget):
    """Floating window for enlarged image preview."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title or "Vergrößerung")
        self.setMinimumSize(600, 400)
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for large images
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet("background-color: #2a2a2a;")

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setWidget(self._image_label)

        layout.addWidget(self._scroll_area)

    def set_image(self, image: np.ndarray, is_rgb: bool = False) -> None:
        """Set the image to display."""
        if image is None:
            return

        # Convert to RGB if needed
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif not is_rgb:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image

        # Create QImage and QPixmap
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(
            display_image.data.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Show at actual size or scaled to fit window
        pixmap = QPixmap.fromImage(q_image)
        self._image_label.setPixmap(pixmap)
        self._image_label.adjustSize()


class ImagePreview(QWidget):
    """Widget for displaying image previews with automatic scaling."""

    clicked = pyqtSignal()

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._title = title
        self._zoom_window: Optional[ZoomWindow] = None
        self._is_rgb: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title label
        if self._title:
            title_label = QLabel(self._title)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
            layout.addWidget(title_label)

        # Image label in scroll area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setMinimumSize(200, 200)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "background-color: #383838; border: 1px solid #555; color: #888;"
        )
        self._image_label.setText("Kein Bild")
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._scroll_area.setWidget(self._image_label)
        layout.addWidget(self._scroll_area)

    def set_image(self, image: np.ndarray, is_rgb: bool = False) -> None:
        """
        Set the image to display.

        Args:
            image: Image as numpy array.
            is_rgb: True if image is RGB, False if BGR or grayscale.
        """
        self._image = image.copy()
        self._is_rgb = is_rgb

        # Convert to RGB if needed
        if len(image.shape) == 2:
            # Grayscale - convert to RGB
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif not is_rgb:
            # BGR to RGB
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image

        # Create QImage
        height, width = display_image.shape[:2]
        if len(display_image.shape) == 3:
            bytes_per_line = 3 * width
            qimage = QImage(
                display_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
        else:
            bytes_per_line = width
            qimage = QImage(
                display_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8,
            )

        # Scale to fit while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimage)
        margin = self._scroll_area.contentsMargins().left() * 2
        available_size = self._scroll_area.size()
        target_width = max(100, available_size.width() - margin)
        target_height = max(100, available_size.height() - margin)
        scaled_pixmap = pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._image_label.setPixmap(scaled_pixmap)

    def get_image(self) -> Optional[np.ndarray]:
        """Get the current image."""
        return self._image

    def clear(self) -> None:
        """Clear the displayed image."""
        self._image = None
        self._is_rgb = False
        self._image_label.clear()
        self._image_label.setText("Kein Bild")
        if self._zoom_window:
            self._zoom_window.close()
            self._zoom_window = None

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events - open zoom window."""
        if self._image is not None:
            self._show_zoom_window()
        self.clicked.emit()
        super().mousePressEvent(event)

    def _show_zoom_window(self) -> None:
        """Show or update the zoom window."""
        if self._zoom_window is None:
            self._zoom_window = ZoomWindow(self._title)
        self._zoom_window.set_image(self._image, self._is_rgb)
        self._zoom_window.show()
        self._zoom_window.raise_()
        self._zoom_window.activateWindow()


class SettingsPanel(QWidget):
    """Panel containing all processing settings."""

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Values (Grayscale levels) group - min 2, max 9
        values_group = QGroupBox("Graustufen (Values)")
        values_layout = QHBoxLayout(values_group)

        self._values_slider = QSlider(Qt.Orientation.Horizontal)
        self._values_slider.setRange(2, 9)
        self._values_slider.setValue(5)
        self._values_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._values_slider.setTickInterval(1)

        self._values_spinbox = QSpinBox()
        self._values_spinbox.setRange(2, 9)
        self._values_spinbox.setValue(5)
        self._values_spinbox.setToolTip("Direkte Eingabe möglich (2-9)")

        values_layout.addWidget(QLabel("2"))
        values_layout.addWidget(self._values_slider)
        values_layout.addWidget(QLabel("9"))
        values_layout.addWidget(self._values_spinbox)

        layout.addWidget(values_group)

        # Steps (Color levels) group - min 2, max 24, increments of 1
        steps_group = QGroupBox("Farbstufen (Steps)")
        steps_layout = QHBoxLayout(steps_group)

        self._steps_slider = QSlider(Qt.Orientation.Horizontal)
        self._steps_slider.setRange(2, 24)
        self._steps_slider.setValue(9)
        self._steps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._steps_slider.setTickInterval(1)

        self._steps_spinbox = QSpinBox()
        self._steps_spinbox.setRange(2, 24)
        self._steps_spinbox.setSingleStep(1)
        self._steps_spinbox.setValue(9)
        self._steps_spinbox.setToolTip("Direkte Eingabe möglich (2-24)")

        steps_layout.addWidget(QLabel("2"))
        steps_layout.addWidget(self._steps_slider)
        steps_layout.addWidget(QLabel("24"))
        steps_layout.addWidget(self._steps_spinbox)

        layout.addWidget(steps_group)


        # Palette sorting options
        sort_group = QGroupBox("Paletten-Optionen")
        sort_layout = QVBoxLayout(sort_group)

        # Palette extraction method
        extract_row = QHBoxLayout()
        extract_row.addWidget(QLabel("Methode:"))
        self._palette_method_combo = QComboBox()
        self._palette_method_combo.addItem("Intensify (empfohlen)", PaletteMethod.INTENSIFY)
        self._palette_method_combo.addItem("Divers", PaletteMethod.DIVERSE)
        self._palette_method_combo.addItem("Gesättigt", PaletteMethod.SATURATED)
        self._palette_method_combo.addItem("Standard", PaletteMethod.STANDARD)
        self._palette_method_combo.setToolTip(
            "Intensify: Garantiert leuchtende Farben aus jeder Farbfamilie\n"
            "Divers: Maximiert Farbkontraste, gute Abdeckung\n"
            "Gesättigt: Bevorzugt kräftige Farben\n"
            "Standard: Nach Häufigkeit (K-Means Original)"
        )
        extract_row.addWidget(self._palette_method_combo)
        sort_layout.addLayout(extract_row)

        # Sort method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Sortierung:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("Nach Farbton", SortMethod.HUE)
        self._sort_combo.addItem("Nach Helligkeit", SortMethod.LIGHTNESS)
        self._sort_combo.addItem("Nach Sättigung", SortMethod.SATURATION)
        self._sort_combo.addItem("Nach Fläche", SortMethod.PERCENTAGE)
        method_row.addWidget(self._sort_combo)
        sort_layout.addLayout(method_row)

        # Add numbers checkbox
        self._add_numbers_checkbox = QCheckBox("Nummern anzeigen")
        self._add_numbers_checkbox.setChecked(True)
        sort_layout.addWidget(self._add_numbers_checkbox)

        layout.addWidget(sort_group)

        # Outline options (including edge sensitivity)
        outline_group = QGroupBox("Konturenerkennung")
        outline_layout = QVBoxLayout(outline_group)

        # Outline source dropdown
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Quelle:"))
        self._outline_source_combo = QComboBox()
        self._outline_source_combo.addItem("Original", OutlineSource.ORIGINAL)
        self._outline_source_combo.addItem("Posterisiert", OutlineSource.POSTERIZED)
        self._outline_source_combo.addItem("Graustufen", OutlineSource.SHADES)
        self._outline_source_combo.addItem("Kombiniert", OutlineSource.COMBINED)
        self._outline_source_combo.setCurrentIndex(1)  # Default: Posterized
        self._outline_source_combo.setToolTip(
            "Original: Konturen aus Originalbild\n"
            "Posterisiert: Klarere Konturen aus Farbposterisierung\n"
            "Graustufen: Konturen aus Graustufenbild\n"
            "Kombiniert: Überlagerung von Posterisiert und Graustufen"
        )
        source_row.addWidget(self._outline_source_combo)
        outline_layout.addLayout(source_row)

        # Edge sensitivity slider (moved here, labels fixed: 0%=Grob, 100%=Fein)
        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Sensitivität:"))
        self._edge_slider = QSlider(Qt.Orientation.Horizontal)
        self._edge_slider.setRange(0, 100)
        self._edge_slider.setValue(50)
        self._edge_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._edge_slider.setTickInterval(10)
        self._edge_slider.setToolTip(
            "0% = Grob (wenige Kanten)\n100% = Fein (viele Kanten)"
        )
        self._edge_label = QLabel("50%")
        self._edge_label.setMinimumWidth(40)
        edge_row.addWidget(QLabel("Grob"))
        edge_row.addWidget(self._edge_slider)
        edge_row.addWidget(QLabel("Fein"))
        edge_row.addWidget(self._edge_label)
        outline_layout.addLayout(edge_row)

        # Minimum contour length slider
        min_len_row = QHBoxLayout()
        min_len_row.addWidget(QLabel("Min. Länge:"))
        self._min_length_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_length_slider.setRange(0, 200)
        self._min_length_slider.setValue(0)
        self._min_length_slider.setToolTip("Minimale Konturlänge (0 = kein Filter)")
        self._min_length_label = QLabel("0")
        self._min_length_label.setMinimumWidth(30)
        min_len_row.addWidget(self._min_length_slider)
        min_len_row.addWidget(self._min_length_label)
        outline_layout.addLayout(min_len_row)

        # Maximum curvature slider (filters out small circles/squiggles)
        max_curv_row = QHBoxLayout()
        max_curv_row.addWidget(QLabel("Max. Krümmung:"))
        self._max_curvature_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_curvature_slider.setRange(0, 100)
        self._max_curvature_slider.setValue(100)  # 100 = allow all
        self._max_curvature_slider.setToolTip(
            "Filtert kreisförmige Konturen (100 = alle erlauben, 0 = nur gerade Linien)"
        )
        self._max_curvature_label = QLabel("100%")
        self._max_curvature_label.setMinimumWidth(40)
        max_curv_row.addWidget(self._max_curvature_slider)
        max_curv_row.addWidget(self._max_curvature_label)
        outline_layout.addLayout(max_curv_row)

        # Minimum contrast slider
        min_contrast_row = QHBoxLayout()
        min_contrast_row.addWidget(QLabel("Min. Kontrast:"))
        self._min_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_contrast_slider.setRange(0, 100)
        self._min_contrast_slider.setValue(0)
        self._min_contrast_slider.setToolTip(
            "Erhöht Schwellwert für Kantenerkennung (0 = Standard)"
        )
        self._min_contrast_label = QLabel("0")
        self._min_contrast_label.setMinimumWidth(30)
        min_contrast_row.addWidget(self._min_contrast_slider)
        min_contrast_row.addWidget(self._min_contrast_label)
        outline_layout.addLayout(min_contrast_row)

        layout.addWidget(outline_group)

        # Folder settings
        folder_group = QGroupBox("Ordner")
        folder_layout = QVBoxLayout(folder_group)

        # Source folder
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Quellordner:"))
        self._source_folder_edit = QLineEdit()
        self._source_folder_edit.setPlaceholderText("Standard-Ordner zum Öffnen")
        src_row.addWidget(self._source_folder_edit)
        self._source_folder_btn = QPushButton("...")
        self._source_folder_btn.setMaximumWidth(30)
        src_row.addWidget(self._source_folder_btn)
        folder_layout.addLayout(src_row)

        # Export folder
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exportordner:"))
        self._export_folder_edit = QLineEdit()
        self._export_folder_edit.setPlaceholderText("Standard-Ordner zum Speichern")
        exp_row.addWidget(self._export_folder_edit)
        self._export_folder_btn = QPushButton("...")
        self._export_folder_btn.setMaximumWidth(30)
        exp_row.addWidget(self._export_folder_btn)
        folder_layout.addLayout(exp_row)

        layout.addWidget(folder_group)

        layout.addStretch()

        # Load saved settings
        self._load_settings()

    def _connect_signals(self) -> None:
        # Sync sliders and spinboxes
        self._values_slider.valueChanged.connect(self._values_spinbox.setValue)
        self._values_spinbox.valueChanged.connect(self._values_slider.setValue)
        self._values_spinbox.valueChanged.connect(lambda: self.settings_changed.emit())

        self._steps_slider.valueChanged.connect(self._steps_spinbox.setValue)
        self._steps_spinbox.valueChanged.connect(self._steps_slider.setValue)
        self._steps_spinbox.valueChanged.connect(lambda: self.settings_changed.emit())

        self._edge_slider.valueChanged.connect(self._on_edge_changed)

        self._palette_method_combo.currentIndexChanged.connect(lambda: self.settings_changed.emit())
        self._sort_combo.currentIndexChanged.connect(lambda: self.settings_changed.emit())
        self._add_numbers_checkbox.toggled.connect(lambda: self.settings_changed.emit())
        self._outline_source_combo.currentIndexChanged.connect(lambda: self.settings_changed.emit())

        # Outline filter sliders
        self._min_length_slider.valueChanged.connect(self._on_min_length_changed)
        self._max_curvature_slider.valueChanged.connect(self._on_max_curvature_changed)
        self._min_contrast_slider.valueChanged.connect(self._on_min_contrast_changed)

        # Folder browser buttons
        self._source_folder_btn.clicked.connect(self._browse_source_folder)
        self._export_folder_btn.clicked.connect(self._browse_export_folder)

    def _on_edge_changed(self, value: int) -> None:
        self._edge_label.setText(f"{value}%")
        self.settings_changed.emit()

    def _on_min_length_changed(self, value: int) -> None:
        self._min_length_label.setText(str(value))
        self.settings_changed.emit()

    def _on_max_curvature_changed(self, value: int) -> None:
        self._max_curvature_label.setText(f"{value}%")
        self.settings_changed.emit()

    def _on_min_contrast_changed(self, value: int) -> None:
        self._min_contrast_label.setText(str(value))
        self.settings_changed.emit()

    def get_values(self) -> int:
        """Get the number of grayscale values."""
        return self._values_spinbox.value()

    def get_steps(self) -> int:
        """Get the number of color steps."""
        return self._steps_spinbox.value()

    def get_edge_sensitivity(self) -> float:
        """Get edge sensitivity (0.0 to 1.0)."""
        return self._edge_slider.value() / 100.0

    def get_palette_method(self) -> PaletteMethod:
        """Get the selected palette extraction method."""
        return self._palette_method_combo.currentData()

    def get_sort_method(self) -> SortMethod:
        """Get the selected palette sort method."""
        return self._sort_combo.currentData()

    def get_grays_position(self) -> str:
        """Get grays position (always 'end')."""
        return "end"  # Hardcoded as per user request

    def should_add_numbers(self) -> bool:
        """Check if numbers should be added to images."""
        return self._add_numbers_checkbox.isChecked()

    def get_outline_source(self) -> OutlineSource:
        """Get the outline extraction source."""
        return self._outline_source_combo.currentData()

    def get_min_contour_length(self) -> int:
        """Get minimum contour length filter value."""
        return self._min_length_slider.value()

    def get_max_curvature(self) -> float:
        """Get maximum curvature filter (0.0-1.0)."""
        return self._max_curvature_slider.value() / 100.0

    def get_min_contrast(self) -> int:
        """Get minimum contrast threshold."""
        return self._min_contrast_slider.value()

    def get_source_folder(self) -> str:
        """Get the source folder path."""
        return self._source_folder_edit.text()

    def get_export_folder(self) -> str:
        """Get the export folder path."""
        return self._export_folder_edit.text()

    def _browse_source_folder(self) -> None:
        """Open folder browser for source folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Quellordner auswählen", self._source_folder_edit.text()
        )
        if folder:
            self._source_folder_edit.setText(folder)
            self._save_settings()

    def _browse_export_folder(self) -> None:
        """Open folder browser for export folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Exportordner auswählen", self._export_folder_edit.text()
        )
        if folder:
            self._export_folder_edit.setText(folder)
            self._save_settings()

    def _load_settings(self) -> None:
        """Load settings from QSettings."""
        settings = QSettings("Trilumin", "TriluMinProcess")

        # Load values
        self._values_spinbox.setValue(settings.value("values", 5, type=int))
        self._steps_spinbox.setValue(settings.value("steps", 9, type=int))
        self._edge_slider.setValue(settings.value("edge_sensitivity", 50, type=int))

        # Load palette method (default: DIVERSE = 0)
        palette_method_index = settings.value("palette_method", 0, type=int)
        self._palette_method_combo.setCurrentIndex(palette_method_index)

        # Load sort method
        sort_index = settings.value("sort_method", 0, type=int)
        self._sort_combo.setCurrentIndex(sort_index)

        # Load add numbers
        self._add_numbers_checkbox.setChecked(
            settings.value("add_numbers", True, type=bool)
        )

        # Load outline source
        outline_index = settings.value("outline_source", 1, type=int)
        self._outline_source_combo.setCurrentIndex(outline_index)

        # Load outline filters
        self._min_length_slider.setValue(settings.value("min_contour_length", 0, type=int))
        self._max_curvature_slider.setValue(settings.value("max_curvature", 100, type=int))
        self._min_contrast_slider.setValue(settings.value("min_contrast", 0, type=int))

        # Load folders
        self._source_folder_edit.setText(
            settings.value("source_folder", "", type=str)
        )
        self._export_folder_edit.setText(
            settings.value("export_folder", "", type=str)
        )

    def save_settings(self) -> None:
        """Save current settings to QSettings."""
        self._save_settings()

    def _save_settings(self) -> None:
        """Save current settings to QSettings (internal)."""
        settings = QSettings("Trilumin", "TriluMinProcess")

        settings.setValue("values", self._values_spinbox.value())
        settings.setValue("steps", self._steps_spinbox.value())
        settings.setValue("edge_sensitivity", self._edge_slider.value())
        settings.setValue("palette_method", self._palette_method_combo.currentIndex())
        settings.setValue("sort_method", self._sort_combo.currentIndex())
        settings.setValue("add_numbers", self._add_numbers_checkbox.isChecked())
        settings.setValue("outline_source", self._outline_source_combo.currentIndex())
        settings.setValue("min_contour_length", self._min_length_slider.value())
        settings.setValue("max_curvature", self._max_curvature_slider.value())
        settings.setValue("min_contrast", self._min_contrast_slider.value())
        settings.setValue("source_folder", self._source_folder_edit.text())
        settings.setValue("export_folder", self._export_folder_edit.text())


class ResultPanel(QWidget):
    """Panel for displaying a processing result with save button."""

    save_requested = pyqtSignal(str)  # Emits the result type

    def __init__(
        self,
        title: str,
        result_type: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._title = title
        self._result_type = result_type
        self._image: Optional[np.ndarray] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with title and save button
        header_layout = QHBoxLayout()

        title_label = QLabel(self._title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self._save_button = QPushButton("Speichern")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(
            lambda: self.save_requested.emit(self._result_type)
        )
        header_layout.addWidget(self._save_button)

        layout.addLayout(header_layout)

        # Image preview
        self._preview = ImagePreview()
        layout.addWidget(self._preview)

    def set_image(self, image: np.ndarray, is_rgb: bool = False) -> None:
        """Set the result image."""
        self._image = image.copy()
        self._preview.set_image(image, is_rgb)
        self._save_button.setEnabled(True)

    def get_image(self) -> Optional[np.ndarray]:
        """Get the current result image."""
        return self._image

    def clear(self) -> None:
        """Clear the result."""
        self._image = None
        self._preview.clear()
        self._save_button.setEnabled(False)

    def get_result_type(self) -> str:
        """Get the result type identifier."""
        return self._result_type


class AbstractionSettingsPanel(QWidget):
    """Panel for image abstraction/pre-processing settings."""

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Abstraction group
        group = QGroupBox("Vorverarbeitung (Abstraktion)")
        group_layout = QVBoxLayout(group)

        # Enable checkbox
        self._enabled_checkbox = QCheckBox("Vorverarbeitung aktivieren")
        self._enabled_checkbox.setChecked(True)  # Enabled by default
        group_layout.addWidget(self._enabled_checkbox)

        # Color Boost checkbox
        self._color_boost_checkbox = QCheckBox("Color Boost")
        self._color_boost_checkbox.setChecked(False)
        self._color_boost_checkbox.setToolTip(
            "Erhöht Farbintensität:\n"
            "+15% Sättigung, +5% Kontrast, +5% Klarheit,\n"
            "-5% Textur, +3% Dunstentfernung, +1% Belichtung"
        )
        group_layout.addWidget(self._color_boost_checkbox)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Methode:"))

        self._method_combo = QComboBox()
        self._method_combo.addItem("Bilateral Filter", AbstractionMethod.BILATERAL)
        self._method_combo.addItem("Pixelierung", AbstractionMethod.PIXELATE)
        self._method_combo.addItem("Mean Shift", AbstractionMethod.MEAN_SHIFT)
        self._method_combo.addItem("K-Means Farben", AbstractionMethod.KMEANS)
        self._method_combo.setEnabled(True)  # Enabled by default

        method_layout.addWidget(self._method_combo)
        group_layout.addLayout(method_layout)

        # Detail level slider
        detail_layout = QHBoxLayout()
        detail_layout.addWidget(QLabel("Detail:"))

        self._detail_slider = QSlider(Qt.Orientation.Horizontal)
        self._detail_slider.setRange(1, 10)
        self._detail_slider.setValue(5)
        self._detail_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._detail_slider.setTickInterval(1)
        self._detail_slider.setEnabled(True)  # Enabled by default

        self._detail_label = QLabel("5")
        self._detail_label.setMinimumWidth(20)

        detail_layout.addWidget(QLabel("1"))
        detail_layout.addWidget(self._detail_slider)
        detail_layout.addWidget(QLabel("10"))
        detail_layout.addWidget(self._detail_label)

        group_layout.addLayout(detail_layout)

        # Help text
        help_label = QLabel("1 = stark abstrahiert, 10 = fast original")
        help_label.setStyleSheet("color: #888; font-size: 10px;")
        group_layout.addWidget(help_label)

        layout.addWidget(group)

    def _connect_signals(self) -> None:
        self._enabled_checkbox.toggled.connect(self._on_enabled_changed)
        self._color_boost_checkbox.toggled.connect(lambda: self.settings_changed.emit())
        self._method_combo.currentIndexChanged.connect(
            lambda: self.settings_changed.emit()
        )
        self._detail_slider.valueChanged.connect(self._on_detail_changed)

    def _on_enabled_changed(self, enabled: bool) -> None:
        self._method_combo.setEnabled(enabled)
        self._detail_slider.setEnabled(enabled)
        self.settings_changed.emit()

    def _on_detail_changed(self, value: int) -> None:
        self._detail_label.setText(str(value))
        self.settings_changed.emit()

    def is_enabled(self) -> bool:
        """Check if abstraction is enabled."""
        return self._enabled_checkbox.isChecked()

    def is_color_boost_enabled(self) -> bool:
        """Check if color boost is enabled."""
        return self._color_boost_checkbox.isChecked()

    def get_method(self) -> AbstractionMethod:
        """Get the selected abstraction method."""
        return self._method_combo.currentData()

    def get_detail_level(self) -> int:
        """Get the detail level (1-10)."""
        return self._detail_slider.value()
