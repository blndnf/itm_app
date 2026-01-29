# Trilumin Process

Ein Python-basiertes Bildanalyse-Tool, das Fotografien in drei separate, speicherbare Komponenten für Ölmalerei zerlegt.

## Features

### Drei Ausgabe-Module

1. **Outlines (Konturen)**
   - Canny Edge Detection
   - Schwarze Linien auf weißem Hintergrund
   - Einstellbare Kantensensitivität

2. **Shades (Graustufen)**
   - Konvertierung zu Graustufen
   - Quantisierung auf einstellbare Anzahl Values (3-12)
   - Posterisiertes Graustufenbild

3. **Color Palette (Farbpalette)**
   - K-Means Clustering basierend auf Steps-Einstellung
   - Farbpaletten-Übersicht mit Hex-Codes
   - Posterisiertes Farbbild mit reduzierten Farben

## Installation

### Voraussetzungen
- Python 3.11 oder höher
- Windows 10/11 (primäre Zielplattform)

### Setup

```bash
# Repository klonen
git clone https://github.com/Frando247/algovia-labs.git
cd algovia-labs/trilumin-process

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
venv\Scripts\activate  # Windows
# oder: source venv/bin/activate  # Linux/Mac

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Verwendung

```bash
# Anwendung starten
python src/main.py
```

### Workflow

1. **Bild laden**: Klicken Sie auf "Bild öffnen" und wählen Sie ein Bild (JPG, PNG, TIFF)
2. **Einstellungen anpassen**:
   - **Values**: Anzahl der Graustufen (3-12)
   - **Steps**: Anzahl der Farbstufen (Vielfaches von 3: 6, 9, 12, 15)
   - **Edge Sensitivity**: Empfindlichkeit der Kantenerkennung
3. **Verarbeitung**: Klicken Sie auf "Verarbeiten" um die drei Ausgaben zu generieren
4. **Export**: Speichern Sie einzelne Ergebnisse oder alle auf einmal

## Projektstruktur

```
trilumin-process/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py              # Einstiegspunkt
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py   # Hauptfenster
│   │   └── widgets.py       # Custom Widgets
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── outlines.py      # Canny Edge Detection
│   │   ├── shades.py        # Graustufenquantisierung
│   │   └── palette.py       # K-Means Farbextraktion
│   └── utils/
│       ├── __init__.py
│       └── image_io.py      # Laden/Speichern
└── tests/
    ├── __init__.py
    └── test_processing.py
```

## Technologie-Stack

- **Python 3.11+** - Programmiersprache
- **PyQt6** - GUI Framework
- **OpenCV** - Bildverarbeitung
- **NumPy** - Numerische Berechnungen
- **scikit-learn** - K-Means Clustering

## Lizenz

MIT License

## Autor

Entwickelt für Algovia Labs
