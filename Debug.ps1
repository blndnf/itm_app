# PDF Visual Debugger V3 - Super detailliert
# Zeigt JEDEN Schritt der Verarbeitung

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Add-Type -Path (Join-Path $ScriptPath "itextsharp.dll")

$customStrategyCode = @"
using System;
using System.Collections.Generic;
using iTextSharp.text.pdf.parser;
public class CoordinateTextExtractionStrategy : ITextExtractionStrategy
{
    public List<TextChunk> TextChunks = new List<TextChunk>();
    public void BeginTextBlock() { }
    public void EndTextBlock() { }
    public void RenderText(TextRenderInfo renderInfo)
    {
        string text = renderInfo.GetText();
        var baseline = renderInfo.GetBaseline();
        float x = baseline.GetStartPoint()[0];
        float y = baseline.GetStartPoint()[1];
        TextChunks.Add(new TextChunk(text, x, y));
    }
    public string GetResultantText() { return ""; }
    public void RenderImage(ImageRenderInfo renderInfo) { }
}
public class TextChunk
{
    public string Text { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public TextChunk(string text, float x, float y) { Text = text; X = x; Y = y; }
}
"@

Add-Type -TypeDefinition $customStrategyCode -ReferencedAssemblies (Join-Path $ScriptPath "itextsharp.dll")

Write-Host "=============================================="
Write-Host " PDF Visual Debugger V3 - Super Detail"
Write-Host "=============================================="
Write-Host ""

Add-Type -AssemblyName System.Windows.Forms
$ofd = New-Object System.Windows.Forms.OpenFileDialog
$ofd.Filter = "PDF Dateien (*.pdf)|*.pdf"
if ($ofd.ShowDialog() -ne "OK") { return }

$pdfPath = $ofd.FileName
$pdfName = [System.IO.Path]::GetFileNameWithoutExtension($pdfPath)

Write-Host "Analysiere: $($ofd.SafeFileName)"

$reader = New-Object iTextSharp.text.pdf.PdfReader -ArgumentList $pdfPath
$totalPages = $reader.NumberOfPages
Write-Host "Seiten gesamt: $totalPages"

# Extrahiere Chunks von ALLEN Seiten
$chunks = @()
for ($pageNum = 1; $pageNum -le $totalPages; $pageNum++) {
    $strategy = New-Object CoordinateTextExtractionStrategy
    [void][iTextSharp.text.pdf.parser.PdfTextExtractor]::GetTextFromPage($reader, $pageNum, $strategy)

    foreach ($chunk in $strategy.TextChunks) {
        $chunks += [pscustomobject]@{ Page=$pageNum; X=$chunk.X; Y=$chunk.Y; Text=$chunk.Text }
    }
}
$reader.Close()

Write-Host "Chunks gesamt (alle Seiten): $($chunks.Count)"
Write-Host ""

$outputPath = Join-Path $ScriptPath "$pdfName`_debug_super_detail.txt"
$output = @()
$output += "=========================================="
$output += " SUPER DETAIL DEBUG"
$output += "=========================================="
$output += "Chunks: $($chunks.Count)"
$output += ""

# === SCHRITT 1: Y-GRUPPIERUNG ===
$output += "=== SCHRITT 1: Y-GRUPPIERUNG ==="
$output += ""

function Group-ChunksByY {
    param($chunks, $yTolerance = 0.5)
    $yGroups = @{}
    foreach ($chunk in $chunks) {
        $y = $chunk.Y
        $foundGroup = $false
        foreach ($existingY in $yGroups.Keys) {
            if ([Math]::Abs($y - $existingY) -lt $yTolerance) {
                $yGroups[$existingY] += $chunk
                $foundGroup = $true
                break
            }
        }
        if (-not $foundGroup) { $yGroups[$y] = @($chunk) }
    }
    return $yGroups
}

$yGroups = Group-ChunksByY -chunks $chunks
$output += "Y-Gruppen: $($yGroups.Count)"
$output += ""

# Zeige erste 10 Y-Gruppen MIT ALLEN CHUNKS
$yKeys = $yGroups.Keys | Sort-Object -Descending | Select-Object -First 10
foreach ($y in $yKeys) {
    $output += "Y=$([Math]::Round($y, 2))"
    $yChunks = @($yGroups[$y]) | Sort-Object X
    foreach ($chunk in $yChunks) {
        $output += "  X=$([Math]::Round($chunk.X, 2).ToString().PadLeft(7)) | '$($chunk.Text)'"
    }
    $output += ""
}

# === SCHRITT 2: HEADER FINDEN (NUR SEITE 1) ===
$output += "=== SCHRITT 2: HEADER FINDEN (NUR SEITE 1) ==="
$output += ""

# Suche Header nur auf Seite 1
$page1Chunks = $chunks | Where-Object { $_.Page -eq 1 }
$headerY = $null
foreach ($y in ($yGroups.Keys | Sort-Object -Descending)) {
    $yChunks = @($yGroups[$y]) | Where-Object { $_.Page -eq 1 }
    if ($yChunks.Count -eq 0) { continue }

    $texts = $yChunks | Select-Object -ExpandProperty Text
    if (($texts -contains "User") -and ($texts -contains "Group") -and ($texts -contains "Category")) {
        $headerY = $y
        $output += "Header gefunden bei Y=$([Math]::Round($headerY, 2)) (Seite 1)"
        break
    }
}

if (-not $headerY) {
    $output += "FEHLER: Kein Header auf Seite 1!"
    $output | Out-File -FilePath $outputPath -Encoding UTF8
    return
}

# === SCHRITT 3: SPALTEN DEFINIEREN (VON SEITE 1) ===
$output += ""
$output += "=== SCHRITT 3: SPALTEN DEFINIEREN (VON SEITE 1) ==="
$output += "Spaltenbreiten werden fuer ALLE Seiten verwendet"
$output += ""

$headerChunks = $page1Chunks | Where-Object {
    [Math]::Abs($_.Y - $headerY) -lt 2.0 -and
    $_.Text -match "^(User|Group|Time|Succeeded|Category|Reason|Action|Error)"
} | Sort-Object X

$columns = @()
foreach ($hc in $headerChunks) {
    $columns += [pscustomobject]@{ Name=$hc.Text; XStart=$hc.X; XEnd=999 }
}

# Setze XEnd fuer jede Spalte
for ($i = 0; $i -lt $columns.Count; $i++) {
    if ($i+1 -lt $columns.Count) {
        $columns[$i].XEnd = $columns[$i+1].XStart
    } else {
        $columns[$i].XEnd = 1000
    }
    $output += "$($columns[$i].Name.PadRight(15)) X=$([Math]::Round($columns[$i].XStart, 2)) bis X=$([Math]::Round($columns[$i].XEnd, 2))"
}

$output += ""
$output += "Spalten: $($columns.Count)"
$output += ""

# === SCHRITT 4: TABELLENZEILEN FINDEN ===
$output += "=== SCHRITT 4: TABELLENZEILEN FINDEN ==="
$output += "(Y-Gruppen wo User-Spalte UND Group-Spalte gefuellt)"
$output += ""

# ZUERST: Finde Footer Y-Position (enthaelt "Creation" UND "Page")
$footerY = $null
foreach ($y in ($yGroups.Keys)) {
    $yChunks = @($yGroups[$y])
    $yText = ($yChunks | Select-Object -ExpandProperty Text) -join " "
    if ($yText -match "Creation" -and $yText -match "Page") {
        $footerY = $y
        $output += "Footer-Zeile gefunden bei Y=$([Math]::Round($footerY, 2))"
        break
    }
}

if ($footerY -eq $null) {
    $output += "WARNUNG: Keine separate Footer-Zeile gefunden - Footer-Chunks koennten inline sein"
}

# Funktion: Bereinigt Footer-Bestandteile aus fertigen Zelleninhalten (retrospektiv)
function Clean-FooterFromCell {
    param($cellText, $columnName)

    $cleaned = $cellText
    $original = $cellText

    # User-Spalte: Entferne Footer-Datum am Ende (mit Trennzeichen)
    if ($columnName -eq "User") {
        # Entferne " ion Date : YYYY-MM-DD" (ion nicht Teil eines Wortes)
        $cleaned = $cleaned -replace "(?<!\w)ion\s+Date\s*:\s*\d{4}-\d{2}-\d{2}.*$", ""
        # Entferne " Date : YYYY-MM-DD" oder " Date:YYYY-MM-DD"
        $cleaned = $cleaned -replace "\s+Date\s*:\s*\d{4}-\d{2}-\d{2}.*$", ""
        # Fallback: Entferne Datum-Pattern am Ende (wenn mit Leerzeichen getrennt)
        $cleaned = $cleaned -replace "\s+\d{4}-\d{2}-\d{2}\s*$", ""
    }

    # Group-Spalte: Entferne Zeitstempel-Fragmente (HH:MM oder HH:MM:SS)
    if ($columnName -eq "Group") {
        # Entferne " HH:MM:SS" am Ende
        $cleaned = $cleaned -replace "\s+\d{1,2}:\d{2}:\d{2}\s*$", ""
        # Entferne " HH:MM:" am Ende (mit abschliessendem Doppelpunkt)
        $cleaned = $cleaned -replace "\s+\d{1,2}:\d{2}:\s*$", ""
        # Entferne " HH:MM" am Ende
        $cleaned = $cleaned -replace "\s+\d{1,2}:\d{2}\s*$", ""
    }

    # Time-Spalte: Entferne alles NACH dem ersten GMT±HH:MM
    if ($columnName -match "Time") {
        # Finde Position des ersten GMT und schneide ab dort ab
        if ($cleaned -match "(.*?GMT[+-]\d{2}:\d{2})") {
            $cleaned = $matches[1]
        }
    }

    # Error-Spalte: Entferne "Page X/Y" (mit oder ohne Leerzeichen)
    if ($columnName -match "Error") {
        # Entferne "Page X/Y" oder "PageX/Y"
        $cleaned = $cleaned -replace "Page\s*\d+/\d+", ""
        # Entferne nur "X/Y" falls übrig
        $cleaned = $cleaned -replace "^\s*\d+/\d+\s*$", ""
    }

    $result = $cleaned.Trim()

    # Debug: Zeige nur wenn etwas geändert wurde
    if ($original -ne $result) {
        $script:cleaningLog += "  [$columnName] '$original' -> '$result'`n"
    }

    return $result
}

# === SCHRITT 4: TABELLENZEILEN FINDEN (UEBER ALLE SEITEN) ===
$output += ""
$output += "=== SCHRITT 4: TABELLENZEILEN FINDEN (UEBER ALLE SEITEN) ==="
$output += "Ein User-Eintrag markiert eine neue Tabellenzeile"
$output += ""

$userCol = $columns[0]
$groupCol = $columns[1]
$tableRowStarts = @()  # Array of [pscustomobject]@{Page=X; Y=Y}

# Durchlaufe jede Seite
for ($pageNum = 1; $pageNum -le $totalPages; $pageNum++) {
    $output += "--- Seite $pageNum ---"

    $pageChunks = $chunks | Where-Object { $_.Page -eq $pageNum }
    $pageYGroups = Group-ChunksByY -chunks $pageChunks

    $foundOnPage = 0
    foreach ($y in ($pageYGroups.Keys | Sort-Object -Descending)) {
        # Auf Seite 1: Ueberspringe Header-Bereich
        if ($pageNum -eq 1 -and $y -ge ($headerY - 5)) { continue }

        # Ueberspringe sehr niedrige Y (Footer-Bereich)
        if ($y -lt 50) { continue }

        $yChunks = @($pageYGroups[$y])

        # Suche in User-Spalte
        $userChunks = $yChunks | Where-Object { $_.X -ge $userCol.XStart -and $_.X -lt $userCol.XEnd }

        # Wenn User-Spalte gefuellt: Neue Tabellenzeile!
        if ($userChunks) {
            # Optional: Pruefe auch Group-Spalte zur Validierung
            $groupChunks = $yChunks | Where-Object { $_.X -ge $groupCol.XStart -and $_.X -lt $groupCol.XEnd }
            $groupText = ($groupChunks | Select-Object -ExpandProperty Text) -join ""

            if ($groupText -match "ICPMH") {
                $userText = ($userChunks | Select-Object -ExpandProperty Text) -join ""
                $userText = $userText -replace "\s+", " "
                $userText = $userText.Trim()

                $tableRowStarts += [pscustomobject]@{ Page=$pageNum; Y=$y }
                $foundOnPage++
                $output += "  >>> User-Eintrag bei Y=$([Math]::Round($y, 2)) | User:'$userText'"
            }
        }
    }

    $output += "  Tabellenzeilen auf Seite $pageNum`: $foundOnPage"
    $output += ""
}

$output += "Tabellenzeilen gesamt: $($tableRowStarts.Count)"
$output += ""

# === SCHRITT 5: ERSTE ZEILE EXTRAHIEREN (DETAIL MIT MULTI-PAGE) ===
if ($tableRowStarts.Count -gt 0) {
    $output += "=== SCHRITT 5: ERSTE TABELLENZEILE EXTRAHIEREN (MULTI-PAGE) ==="
    $output += ""

    # Sortiere: Erst nach Page, dann nach Y (absteigend)
    $tableRowStarts = $tableRowStarts | Sort-Object Page, @{Expression="Y"; Descending=$true}

    $firstRow = $tableRowStarts[0]
    $secondRow = if ($tableRowStarts.Count -gt 1) { $tableRowStarts[1] } else { $null }

    $output += "Zeile 1: Seite $($firstRow.Page), Y=$([Math]::Round($firstRow.Y, 2))"
    if ($secondRow) {
        $output += "  bis: Seite $($secondRow.Page), Y=$([Math]::Round($secondRow.Y, 2))"
    } else {
        $output += "  bis: Ende des Dokuments"
    }
    $output += ""

    # Sammle Chunks: Von firstRow bis secondRow (über Seiten hinweg)
    $rowChunks = @()
    if ($secondRow) {
        # Fall 1: Es gibt eine nächste Zeile
        if ($firstRow.Page -eq $secondRow.Page) {
            # Gleiche Seite: Einfacher Y-Bereich
            $rowChunks = $chunks | Where-Object {
                $_.Page -eq $firstRow.Page -and
                $_.Y -le $firstRow.Y -and $_.Y -gt $secondRow.Y
            }
        } else {
            # Mehrere Seiten: Sammle von Start-Seite bis Ende, dann alle Zwischenseiten, dann End-Seite
            for ($p = $firstRow.Page; $p -le $secondRow.Page; $p++) {
                if ($p -eq $firstRow.Page) {
                    # Start-Seite: Ab firstRow.Y nach unten
                    $rowChunks += $chunks | Where-Object {
                        $_.Page -eq $p -and $_.Y -le $firstRow.Y -and $_.Y -gt 50  # bis footer
                    }
                } elseif ($p -eq $secondRow.Page) {
                    # End-Seite: Von oben bis secondRow.Y
                    $rowChunks += $chunks | Where-Object {
                        $_.Page -eq $p -and $_.Y -gt $secondRow.Y
                    }
                } else {
                    # Zwischenseite: Alles außer Header/Footer
                    $rowChunks += $chunks | Where-Object {
                        $_.Page -eq $p -and $_.Y -lt 500 -and $_.Y -gt 50
                    }
                }
            }
        }
    } else {
        # Fall 2: Letzte Zeile im Dokument
        $rowChunks = $chunks | Where-Object {
            ($_.Page -eq $firstRow.Page -and $_.Y -le $firstRow.Y) -or
            ($_.Page -gt $firstRow.Page -and $_.Y -gt 50 -and $_.Y -lt 500)
        }
    }

    $output += "Chunks in Zeile (über alle Seiten): $($rowChunks.Count)"
    $output += ""

    # Fuer jede Spalte
    foreach ($col in $columns) {
        $colChunks = $rowChunks | Where-Object {
            $_.X -ge $col.XStart -and $_.X -lt $col.XEnd
        } | Sort-Object Page, @{Expression="Y"; Descending=$true}, X

        # Join ohne Leerzeichen, dann normalisieren
        $cellText = ($colChunks | Select-Object -ExpandProperty Text) -join ""
        $cellText = $cellText -replace "\s+", " "
        $cellText = $cellText.Trim()

        $output += "$($col.Name.PadRight(15)): '$cellText'"

        if ($colChunks) {
            foreach ($chunk in ($colChunks | Select-Object -First 10)) {
                $output += "    Seite$($chunk.Page) X=$([Math]::Round($chunk.X, 2).ToString().PadLeft(6)) Y=$([Math]::Round($chunk.Y, 2).ToString().PadLeft(6)) | '$($chunk.Text)'"
            }
            if ($colChunks.Count -gt 10) {
                $output += "    ... und $($colChunks.Count - 10) weitere Chunks"
            }
        }
        $output += ""
    }
}

# === SCHRITT 6: TABELLE ===
$output += "=== SCHRITT 6: REKONSTRUIERTE TABELLE (ALLE ZEILEN) ==="
$output += ""

# Log für Bereinigungen
$script:cleaningLog = ""

# Berechne Spaltenbreiten basierend auf tatsaechlicher Pixel-Breite
$colWidths = @()
foreach ($col in $columns) {
    $pixelWidth = $col.XEnd - $col.XStart
    $charWidth = [Math]::Max(12, [Math]::Min(40, [Math]::Floor($pixelWidth / 5)))  # ~5 Pixel pro Zeichen
    $colWidths += $charWidth
}

# Header
$headerLine = "+"
for ($i = 0; $i -lt [Math]::Min(8, $columns.Count); $i++) {
    $headerLine += ("-" * ($colWidths[$i] + 2)) + "+"
}
$output += $headerLine

$headerText = "|"
for ($i = 0; $i -lt [Math]::Min(8, $columns.Count); $i++) {
    $colName = $columns[$i].Name
    if ($colName.Length -gt $colWidths[$i]) { $colName = $colName.Substring(0, $colWidths[$i]) }
    $headerText += " " + $colName.PadRight($colWidths[$i]) + " |"
}
$output += $headerText
$output += $headerLine

# ALLE Zeilen (Multi-Page Support!)
for ($i = 0; $i -lt $tableRowStarts.Count; $i++) {
    $currentRow = $tableRowStarts[$i]
    $nextRow = if ($i+1 -lt $tableRowStarts.Count) { $tableRowStarts[$i+1] } else { $null }

    # Sammle Chunks: Von currentRow bis nextRow (über Seiten hinweg)
    $rowChunks = @()
    if ($nextRow) {
        # Fall 1: Es gibt eine nächste Zeile
        if ($currentRow.Page -eq $nextRow.Page) {
            # Gleiche Seite: Einfacher Y-Bereich
            $rowChunks = $chunks | Where-Object {
                $_.Page -eq $currentRow.Page -and
                $_.Y -le $currentRow.Y -and $_.Y -gt $nextRow.Y
            }
        } else {
            # Mehrere Seiten
            for ($p = $currentRow.Page; $p -le $nextRow.Page; $p++) {
                if ($p -eq $currentRow.Page) {
                    $rowChunks += $chunks | Where-Object {
                        $_.Page -eq $p -and $_.Y -le $currentRow.Y -and $_.Y -gt 50
                    }
                } elseif ($p -eq $nextRow.Page) {
                    $rowChunks += $chunks | Where-Object {
                        $_.Page -eq $p -and $_.Y -gt $nextRow.Y
                    }
                } else {
                    $rowChunks += $chunks | Where-Object {
                        $_.Page -eq $p -and $_.Y -lt 500 -and $_.Y -gt 50
                    }
                }
            }
        }
    } else {
        # Fall 2: Letzte Zeile
        $rowChunks = $chunks | Where-Object {
            ($_.Page -eq $currentRow.Page -and $_.Y -le $currentRow.Y) -or
            ($_.Page -gt $currentRow.Page -and $_.Y -gt 50 -and $_.Y -lt 500)
        }
    }

    # Sammle Zellinhalte
    $cells = @()
    for ($c = 0; $c -lt [Math]::Min(8, $columns.Count); $c++) {
        $col = $columns[$c]
        $colChunks = $rowChunks | Where-Object {
            $_.X -ge $col.XStart -and $_.X -lt $col.XEnd
        } | Sort-Object Page, @{Expression="Y"; Descending=$true}, X

        # Join ohne Leerzeichen, dann normalisieren
        $cellText = ($colChunks | Select-Object -ExpandProperty Text) -join ""
        $cellText = $cellText -replace "\s+", " "

        # RETROSPEKTIVE BEREINIGUNG: Nur bei der LETZTEN Tabellenzeile
        $finalCellText = $cellText.Trim()
        if ($i -eq ($tableRowStarts.Count - 1)) {
            $finalCellText = Clean-FooterFromCell -cellText $finalCellText -columnName $col.Name
        }

        $cells += $finalCellText
    }
    
    # Umbrechen: Text in Zeilen aufteilen wenn zu lang
    $cellLines = @()
    $maxLines = 1
    for ($c = 0; $c -lt $cells.Count; $c++) {
        $text = $cells[$c]
        $width = $colWidths[$c]
        $lines = @()
        
        while ($text.Length -gt $width) {
            $lines += $text.Substring(0, $width)
            $text = $text.Substring($width)
        }
        if ($text.Length -gt 0) { $lines += $text }
        if ($lines.Count -eq 0) { $lines += "" }
        
        $cellLines += ,@($lines)
        if ($lines.Count -gt $maxLines) { $maxLines = $lines.Count }
    }
    
    # Ausgabe: Mehrere Zeilen wenn noetig
    for ($lineNum = 0; $lineNum -lt $maxLines; $lineNum++) {
        $rowLine = "|"
        for ($c = 0; $c -lt $cells.Count; $c++) {
            $lines = $cellLines[$c]
            $text = if ($lineNum -lt $lines.Count) { $lines[$lineNum] } else { "" }
            $rowLine += " " + $text.PadRight($colWidths[$c]) + " |"
        }
        $output += $rowLine
    }
    
    $output += $headerLine
}

# Zeige Bereinigungslog falls vorhanden
if ($script:cleaningLog -ne "") {
    $output += ""
    $output += "=== FOOTER-BEREINIGUNG (Letzte Zeile) ==="
    $output += $script:cleaningLog
}

$output | Out-File -FilePath $outputPath -Encoding UTF8

Write-Host "Fertig! -> $outputPath"
Read-Host "Enter"