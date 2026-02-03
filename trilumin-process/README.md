# Trilumin Process - Handbuch

Ein Python-basiertes Bildanalyse-Tool, das Fotografien in drei separate, speicherbare Komponenten fuer Oelmalerei zerlegt.

## Schnellstart

```bash
# Abhaengigkeiten installieren
pip install -r requirements.txt

# Anwendung starten
python src/main.py
```

## Uebersicht

Trilumin Process analysiert Bilder und extrahiert drei Komponenten:

1. **Outlines (Konturen)** - Kantenerkennung fuer Vorzeichnungen
2. **Shades (Graustufen)** - Tonwertanalyse fuer Helligkeitsverteilung
3. **Color Palette (Farbpalette)** - Intelligente Farbextraktion mit Balance-Garantie

---

## Palette-Methoden

### STANDARD
Urspruengliche K-Means-Extraktion nach Flaeche. Groessere Bildbereiche werden bevorzugt.
- Gut fuer: Bilder, bei denen dominante Farben wichtig sind
- Nachteil: Kann kleine, aber wichtige Farben ignorieren

### DIVERSE
Maximiert Farbkontraste ueber das Farbrad. Waehlt Farben mit maximalem Abstand zueinander.
- Gut fuer: Ausgewogene Paletten mit breiter Farbverteilung
- Algorithmus: Greedy-Auswahl mit Abstandsmaximierung

### SATURATED (Gesaettigt)
Priorisiert die gesaettigsten Farben jeder Familie.
- Gut fuer: Leuchtende, intensive Farbpaletten
- Sortiert nach: Saettigung (hoechste zuerst)

### GLOW (Leuchtend)
Waehlt die hellsten Farben der haeufigsten Farbfamilien.
- Gut fuer: Helle, luftige Paletten
- Sortiert nach: Helligkeit gewichtet mit Familienflaeche

### LUMINOUS (Strahlend)
Kompromiss zwischen DIVERSE und GLOW - kontrastreich UND hell.
- Gut fuer: Paletten, die beides brauchen
- Algorithmus: Gewichtete Kombination aus Abstand und Helligkeit

### INTENSIFY (Intensivieren) - EMPFOHLEN
Maximiert Farbvielfalt durch komplementaere Paare. Garantiert Farben aus jeder vorhandenen Familie.
- Gut fuer: Komplexe Bilder mit vielen Farbfamilien
- Algorithmus: Fuellt komplementaere Paare abwechselnd:
  - Blau + Orange
  - Rot + Gruen
  - Gelb + Violett
  - Braun (als Akzent)

---

## Farbfamilien

Die Analyse kategorisiert Farben in 7 Kuenstler-Familien:

| Familie  | Farbton (Hue)  | Beschreibung |
|----------|----------------|--------------|
| Rot      | 345-15 Grad    | Warme Roettoene |
| Orange   | 15-45 Grad     | Orange bis Pfirsich |
| Gelb     | 45-70 Grad     | Gelb bis Goldtoene |
| Gruen    | 70-160 Grad    | Alle Gruentoene |
| Blau     | 160-260 Grad   | Tuerkis bis Kobalt |
| Violett  | 260-345 Grad   | Violett bis Magenta |
| Braun    | 15-50 Grad     | Niedrige Saettigung (<45%) |

---

## Balance-Regel (2:1 Maximum)

**KRITISCH:** Die Palette garantiert ein maximaloes Verhaeltnis von 2:1 zwischen Farbfamilien.

Beispiel mit 12 Farben und 4 Familien:
- Maximum pro Familie: 4 Slots
- Minimum pro Familie: 2 Slots
- NICHT erlaubt: 8 Blau + 2 Rot + 1 Gelb + 1 Gruen

Wenn eine Familie blockiert wird:
1. Generiere Variationen der unterrepraesentierten Familien
2. Dunklere/Hellere/Gesaettigtere Versionen der vorhandenen Farben
3. Garantiert faire Verteilung

---

## Erweiterte Einstellungen

Klicke auf das Zahnrad-Symbol (Einstellungen) fuer:

### Clustering-Methoden

| Methode | Beschreibung | Beste fuer |
|---------|--------------|------------|
| K-Means | Standard-Clustering | Allgemein |
| Median Cut | Historisch (GIF) | Gleichmaessige Verteilung |
| Mean Shift | Automatische Cluster | Unbekannte Strukturen |
| DBSCAN | Dichtebasiert | Unregelmaessige Cluster |
| Octree | Baumstruktur | Schnell, deterministisch |
| Hybrid | K-Means + Octree | Balance aus beiden |
| GMM | Probabilistisch | Ueberlappende Farben |

### Parameter

- **Cluster-Anzahl**: Wie viele Farben K-Means initial findet (Standard: 48)
- **Bandwidth**: Kernelgroesse fuer Mean Shift
- **Epsilon/Min Samples**: DBSCAN-Dichteparameter
- **Octree-Tiefe**: Aufloesung des Farbbaums (4-8)

---

## Debug-Modus

Klicke auf **[debug]** nach der Verarbeitung um zu sehen:

1. **Cluster-Analyse**: Wie viele Farben gefunden wurden
2. **Familien-Zuordnung**: Welche Farbe zu welcher Familie gehoert
3. **Balance-Entscheidungen**: Warum bestimmte Farben gewaehlt/abgelehnt wurden
4. **Variations-Generierung**: Wie fehlende Slots gefuellt wurden

---

## Composite-Bild

Das Composite kombiniert alle drei Ausgaben:
- Outlines (Konturen) oben links
- Shades (Graustufen) oben rechts
- Palette (Farbpalette) unten links
- Original unten rechts

---

## Tastenkuerzel

| Taste | Funktion |
|-------|----------|
| Strg+O | Bild oeffnen |
| Strg+S | Bild speichern |
| Strg+Q | Beenden |

---

## Workflow fuer Oelmaler

1. **Bild laden** - Referenzfoto auswaehlen
2. **Outlines exportieren** - Fuer die Vorzeichnung auf Leinwand
3. **Shades analysieren** - Verstehen, wo Licht und Schatten liegen
4. **Palette mit INTENSIFY** - Garantiert ausgewogene Farbmischung
5. **Farben anmischen** - Mit den Hex-Codes als Referenz

---

## Technische Details

### Farbabstand-Berechnung

Die Aehnlichkeit zwischen Farben wird im HSL-Raum berechnet:
- Farbton-Unterschied wird staerker gewichtet bei gesaettigten Farben
- Minimum-Abstand: 0.08 (wird bei Variationen progressiv gelockert)

### Variations-Typen

Je nach Methode werden verschiedene Variationen generiert:
- **lighter**: Hellere Version (L erhoeht)
- **darker**: Dunklere Version (L reduziert)
- **saturated**: Gesaettigtere Version (S erhoeht)
- **desaturated**: Entsaettigte Version (S reduziert)

---

## Installation

### Voraussetzungen
- Python 3.11 oder hoeher
- Windows 10/11, Linux oder macOS

### Setup

```bash
# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Abhaengigkeiten installieren
pip install -r requirements.txt
```

### Abhaengigkeiten
- PyQt6 - GUI Framework
- OpenCV - Bildverarbeitung
- NumPy - Numerische Berechnungen
- scikit-learn - Machine Learning (K-Means, DBSCAN, GMM, Mean Shift)
- scipy - Wissenschaftliche Berechnungen

---

## Fehlerbehebung

### "Fehlende Farbe" (z.B. 11 statt 12)
- Ursache: Variations-Generierung konnte keine eindeutigen Farben finden
- Loesung: Nutze DIVERSE oder INTENSIFY Methode

### Zu viele blaue/einer Farbe
- Ursache: Bild hat dominante Farbfamilie
- Loesung: Balance-Regel sorgt automatisch fuer max 2:1 Verhaeltnis

### Grautoene in der Palette
- Ursache: Niedrige Saettigung im Originalbild
- Loesung: Nutze SATURATED Methode fuer intensivere Farben

---

## Lizenz

MIT License

## Autor

Entwickelt fuer Algovia Labs
