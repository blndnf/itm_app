#!/usr/bin/env python3
"""
Trilumin Process - Image Analysis Tool for Oil Painting Preparation.

This application decomposes photographs into three separate components:
1. Outlines (Contours) - Edge detection for line drawings
2. Shades (Grayscale) - Quantized value studies
3. Color Palette - Reduced color palette with posterization

Usage:
    python src/main.py

Requirements:
    Python 3.11+
    PyQt6, OpenCV, NumPy, scikit-learn
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from gui.main_window import MainWindow


def main() -> int:
    """
    Main entry point for Trilumin Process application.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Trilumin Process")
    app.setOrganizationName("Algovia Labs")
    app.setApplicationVersion("1.0.0")

    # Set application style
    app.setStyle("Fusion")

    # Apply dark-friendly stylesheet
    stylesheet = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #4a90d9;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #5a9fe9;
        }
        QPushButton:pressed {
            background-color: #3a80c9;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #ddd;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            margin: -5px 0;
            background: #4a90d9;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #5a9fe9;
        }
        QSpinBox {
            padding: 4px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        QStatusBar {
            background-color: #e0e0e0;
        }
        QScrollArea {
            border: 1px solid #ccc;
            border-radius: 3px;
        }
    """
    app.setStyleSheet(stylesheet)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
