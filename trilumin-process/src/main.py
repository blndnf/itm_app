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


def check_dependencies() -> list[str]:
    """Check if all required dependencies are installed."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    try:
        from PyQt6 import QtWidgets
    except ImportError:
        missing.append("PyQt6")

    return missing


def main() -> int:
    """
    Main entry point for Trilumin Process application.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Check dependencies first
    missing = check_dependencies()
    if missing:
        print("=" * 60)
        print("FEHLER: Fehlende Abhängigkeiten!")
        print("=" * 60)
        print("\nBitte installieren Sie die folgenden Pakete:\n")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nFühren Sie diesen Befehl aus:")
        print(f"\n  pip install {' '.join(missing)}")
        print("\nOder installieren Sie alle Abhängigkeiten:")
        print("\n  pip install -r requirements.txt")
        print("\n" + "=" * 60)
        input("\nDrücken Sie Enter zum Beenden...")
        return 1

    # Import after dependency check
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from gui.main_window import MainWindow

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

    # Apply dark theme stylesheet
    stylesheet = """
        * {
            color: #e0e0e0;
        }
        QMainWindow {
            background-color: #2d2d2d;
        }
        QWidget {
            background-color: #2d2d2d;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #383838;
            color: #e0e0e0;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #e0e0e0;
        }
        QLabel {
            color: #e0e0e0;
            background-color: transparent;
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
            background-color: #555555;
            color: #888888;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #555;
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
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #383838;
            color: #e0e0e0;
        }
        QComboBox {
            padding: 4px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #383838;
            color: #e0e0e0;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #383838;
            color: #e0e0e0;
            selection-background-color: #4a90d9;
        }
        QCheckBox {
            color: #e0e0e0;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #383838;
        }
        QCheckBox::indicator:checked {
            background-color: #4a90d9;
        }
        QStatusBar {
            background-color: #252525;
            color: #e0e0e0;
        }
        QScrollArea {
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #383838;
        }
        QSplitter::handle {
            background-color: #555;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #383838;
            text-align: center;
            color: #e0e0e0;
        }
        QProgressBar::chunk {
            background-color: #4a90d9;
            border-radius: 2px;
        }
    """
    app.setStyleSheet(stylesheet)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n" + "=" * 60)
        print("FEHLER beim Starten der Anwendung:")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}")
        print("\n" + "=" * 60)
        input("\nDrücken Sie Enter zum Beenden...")
        sys.exit(1)
