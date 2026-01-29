"""Custom widgets for Trilumin Process GUI."""

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
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from processing.abstraction import AbstractionMethod


class ImagePreview(QWidget):
    """Widget for displaying image previews with automatic scaling."""

    clicked = pyqtSignal()

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._title = title
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
        self._image_label.clear()
        self._image_label.setText("Kein Bild")

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events."""
        self.clicked.emit()
        super().mousePressEvent(event)


class SettingsPanel(QWidget):
    """Panel containing all processing settings."""

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Values (Grayscale levels) group
        values_group = QGroupBox("Graustufen (Values)")
        values_layout = QHBoxLayout(values_group)

        self._values_slider = QSlider(Qt.Orientation.Horizontal)
        self._values_slider.setRange(3, 12)
        self._values_slider.setValue(5)
        self._values_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._values_slider.setTickInterval(1)

        self._values_spinbox = QSpinBox()
        self._values_spinbox.setRange(3, 12)
        self._values_spinbox.setValue(5)

        values_layout.addWidget(QLabel("3"))
        values_layout.addWidget(self._values_slider)
        values_layout.addWidget(QLabel("12"))
        values_layout.addWidget(self._values_spinbox)

        layout.addWidget(values_group)

        # Steps (Color levels) group
        steps_group = QGroupBox("Farbstufen (Steps)")
        steps_layout = QHBoxLayout(steps_group)

        self._steps_slider = QSlider(Qt.Orientation.Horizontal)
        self._steps_slider.setRange(2, 8)  # 6, 9, 12, 15, 18, 21, 24
        self._steps_slider.setValue(3)  # Default 9
        self._steps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._steps_slider.setTickInterval(1)

        self._steps_spinbox = QSpinBox()
        self._steps_spinbox.setRange(6, 24)
        self._steps_spinbox.setSingleStep(3)
        self._steps_spinbox.setValue(9)

        steps_layout.addWidget(QLabel("6"))
        steps_layout.addWidget(self._steps_slider)
        steps_layout.addWidget(QLabel("24"))
        steps_layout.addWidget(self._steps_spinbox)

        layout.addWidget(steps_group)

        # Edge Sensitivity group
        edge_group = QGroupBox("Kantensensitivität")
        edge_layout = QHBoxLayout(edge_group)

        self._edge_slider = QSlider(Qt.Orientation.Horizontal)
        self._edge_slider.setRange(0, 100)
        self._edge_slider.setValue(50)
        self._edge_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._edge_slider.setTickInterval(10)

        self._edge_label = QLabel("50%")
        self._edge_label.setMinimumWidth(40)

        edge_layout.addWidget(QLabel("Fein"))
        edge_layout.addWidget(self._edge_slider)
        edge_layout.addWidget(QLabel("Grob"))
        edge_layout.addWidget(self._edge_label)

        layout.addWidget(edge_group)

        layout.addStretch()

    def _connect_signals(self) -> None:
        # Sync sliders and spinboxes
        self._values_slider.valueChanged.connect(self._values_spinbox.setValue)
        self._values_spinbox.valueChanged.connect(self._values_slider.setValue)
        self._values_spinbox.valueChanged.connect(lambda: self.settings_changed.emit())

        self._steps_slider.valueChanged.connect(self._on_steps_slider_changed)
        self._steps_spinbox.valueChanged.connect(self._on_steps_spinbox_changed)

        self._edge_slider.valueChanged.connect(self._on_edge_changed)

    def _on_steps_slider_changed(self, value: int) -> None:
        steps_value = value * 3  # Convert to multiple of 3
        self._steps_spinbox.blockSignals(True)
        self._steps_spinbox.setValue(steps_value)
        self._steps_spinbox.blockSignals(False)
        self.settings_changed.emit()

    def _on_steps_spinbox_changed(self, value: int) -> None:
        # Round to nearest multiple of 3
        value = round(value / 3) * 3
        value = max(6, min(24, value))
        self._steps_spinbox.blockSignals(True)
        self._steps_spinbox.setValue(value)
        self._steps_spinbox.blockSignals(False)

        slider_value = value // 3
        self._steps_slider.blockSignals(True)
        self._steps_slider.setValue(slider_value)
        self._steps_slider.blockSignals(False)
        self.settings_changed.emit()

    def _on_edge_changed(self, value: int) -> None:
        self._edge_label.setText(f"{value}%")
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
        self._enabled_checkbox.setChecked(False)
        group_layout.addWidget(self._enabled_checkbox)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Methode:"))

        self._method_combo = QComboBox()
        self._method_combo.addItem("Bilateral Filter", AbstractionMethod.BILATERAL)
        self._method_combo.addItem("Pixelierung", AbstractionMethod.PIXELATE)
        self._method_combo.addItem("Mean Shift", AbstractionMethod.MEAN_SHIFT)
        self._method_combo.addItem("K-Means Farben", AbstractionMethod.KMEANS)
        self._method_combo.setEnabled(False)

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
        self._detail_slider.setEnabled(False)

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

    def get_method(self) -> AbstractionMethod:
        """Get the selected abstraction method."""
        return self._method_combo.currentData()

    def get_detail_level(self) -> int:
        """Get the detail level (1-10)."""
        return self._detail_slider.value()
