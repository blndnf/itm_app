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
$strategy = New-Object CoordinateTextExtractionStrategy
[void][iTextSharp.text.pdf.parser.PdfTextExtractor]::GetTextFromPage($reader, 1, $strategy)
$chunks = @()
foreach ($chunk in $strategy.TextChunks) {
    $chunks += [pscustomobject]@{ Page=1; X=$chunk.X; Y=$chunk.Y; Text=$chunk.Text }
}
$reader.Close()

Write-Host "Chunks: $($chunks.Count)"
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

# === SCHRITT 2: HEADER FINDEN ===
$output += "=== SCHRITT 2: HEADER FINDEN ==="
$output += ""

$headerY = $null
foreach ($y in ($yGroups.Keys | Sort-Object -Descending)) {
    $yChunks = @($yGroups[$y])
    $texts = $yChunks | Select-Object -ExpandProperty Text
    if (($texts -contains "User") -and ($texts -contains "Group") -and ($texts -contains "Category")) {
        $headerY = $y
        $output += "Header gefunden bei Y=$([Math]::Round($headerY, 2))"
        break
    }
}

if (-not $headerY) {
    $output += "FEHLER: Kein Header!"
    $output | Out-File -FilePath $outputPath -Encoding UTF8
    return
}

# === SCHRITT 3: SPALTEN DEFINIEREN ===
$output += ""
$output += "=== SCHRITT 3: SPALTEN DEFINIEREN ==="
$output += ""

$headerChunks = $chunks | Where-Object { 
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

    # User-Spalte: Entferne "ion Date : YYYY-MM-DD" oder Varianten
    if ($columnName -eq "User") {
        $cleaned = $cleaned -replace "ion Date\s*:\s*\d{4}-\d{2}-\d{2}", ""
        $cleaned = $cleaned -replace "Date\s*:\s*\d{4}-\d{2}-\d{2}", ""
    }

    # Group-Spalte: Entferne Zeitstempel-Fragmente (HH:MM oder HH:MM:SS)
    if ($columnName -eq "Group") {
        $cleaned = $cleaned -replace "\s*\d{1,2}:\d{2}(?::\d{2})?\s*$", ""
    }

    # Time-Spalte: Entferne alles NACH dem ersten GMT±HH:MM
    if ($columnName -match "Time") {
        $cleaned = $cleaned -replace "(GMT[+-]\d{2}:\d{2}).*$", '$1'
    }

    # Error-Spalte: Entferne "Page X/Y" (mit oder ohne Leerzeichen)
    if ($columnName -match "Error") {
        $cleaned = $cleaned -replace "Page\s*\d+/\d+", ""
    }

    return $cleaned.Trim()
}

# Setze lowestValidY: Untere Grenze fuer gueltige Tabellenzeilen
$lowestValidY = if ($footerY -ne $null) { $footerY + 5.0 } else { -100 }
$output += "Untere Grenze fuer Tabellenzeilen: Y=$([Math]::Round($lowestValidY, 2))"

$output += ""
$output += "DEBUG: Pruefe jede Y-Gruppe unterhalb Header..."
$output += ""

$userCol = $columns[0]
$groupCol = $columns[1]
$tableRowStarts = @()

$checkedCount = 0
foreach ($y in ($yGroups.Keys | Sort-Object -Descending)) {
    # Ueberspringe Header-Bereich
    if ($y -ge ($headerY - 5)) { continue }
    
    # Ueberspringe Footer-Zeile
    if ($footerY -ne $null -and [Math]::Abs($y - $footerY) -lt 1.0) { continue }
    
    $checkedCount++
    $yChunks = @($yGroups[$y])
    
    # Suche in User-Spalte
    $userChunks = $yChunks | Where-Object { $_.X -ge $userCol.XStart -and $_.X -lt $userCol.XEnd }
    
    # Suche in Group-Spalte - ERST zusammensetzen, DANN pruefen
    $groupChunks = $yChunks | Where-Object { $_.X -ge $groupCol.XStart -and $_.X -lt $groupCol.XEnd }
    $groupText = ($groupChunks | Sort-Object X | Select-Object -ExpandProperty Text) -join ""
    
    # DEBUG erste 5 Y-Gruppen
    if ($checkedCount -le 5) {
        $userText = ($userChunks | Sort-Object X | Select-Object -ExpandProperty Text) -join ""
        $hasUser = if ($userChunks) { "JA" } else { "NEIN" }
        $hasGroup = if ($groupText -match "ICPMH") { "JA" } else { "NEIN" }
        $output += "Y=$([Math]::Round($y, 2))"
        $output += "  User-Spalte ($($userCol.XStart)-$($userCol.XEnd)): $hasUser | '$userText'"
        $output += "  Group-Spalte ($($groupCol.XStart)-$($groupCol.XEnd)): $hasGroup | '$groupText'"
        $output += ""
    }
    
    if ($userChunks -and ($groupText -match "ICPMH")) {
        $tableRowStarts += $y
        $userText = ($userChunks | Sort-Object X | Select-Object -ExpandProperty Text) -join " "
        $output += ">>> GEFUNDEN bei Y=$([Math]::Round($y, 2)) | User:'$userText' | Group:'$groupText'"
    }
}

$output += "Geprueft: $checkedCount Y-Gruppen"

$output += ""
$output += "Tabellenzeilen gefunden: $($tableRowStarts.Count)"
$output += ""

# === SCHRITT 5: ERSTE ZEILE EXTRAHIEREN (DETAIL) ===
if ($tableRowStarts.Count -gt 0) {
    $output += "=== SCHRITT 5: ERSTE TABELLENZEILE EXTRAHIEREN ==="
    $output += ""
    
    $tableRowStarts = $tableRowStarts | Sort-Object -Descending
    $rowStartY = $tableRowStarts[0]
    $rowEndY = if ($tableRowStarts.Count -gt 1) { $tableRowStarts[1] } else { -100 }
    
    $output += "Zeile 1: Y=$([Math]::Round($rowStartY, 2)) bis Y=$([Math]::Round($rowEndY, 2))"
    $output += ""
    
    # Alle Chunks in diesem Y-Bereich
    $rowChunks = $chunks | Where-Object {
        $_.Y -le $rowStartY -and $_.Y -gt $rowEndY
    }
    $output += "Chunks in Zeile: $($rowChunks.Count)"
    $output += ""
    
    # Fuer jede Spalte
    foreach ($col in $columns) {
        $colChunks = $rowChunks | Where-Object { 
            $_.X -ge $col.XStart -and $_.X -lt $col.XEnd 
        } | Sort-Object @{Expression="Y"; Descending=$true}, X
        
        # INTELLIGENTES Zusammenfuegen: Gleiche Y -> kein Space, andere Y -> Space
        $cellText = ""
        $lastY = $null
        foreach ($chunk in $colChunks) {
            if ($lastY -ne $null -and [Math]::Abs($chunk.Y - $lastY) -gt 1.0) {
                $cellText += " "  # Verschiedene Y-Zeilen -> Leerzeichen
            }
            $cellText += $chunk.Text
            $lastY = $chunk.Y
        }
        
        $output += "$($col.Name.PadRight(15)): '$($cellText.Trim())'"
        
        if ($colChunks) {
            foreach ($chunk in $colChunks) {
                $output += "    X=$([Math]::Round($chunk.X, 2).ToString().PadLeft(7)) Y=$([Math]::Round($chunk.Y, 2).ToString().PadLeft(7)) | '$($chunk.Text)'"
            }
        }
        $output += ""
    }
}

# === SCHRITT 6: TABELLE ===
$output += "=== SCHRITT 6: REKONSTRUIERTE TABELLE (ALLE ZEILEN) ==="
$output += ""

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

# ALLE Zeilen (nicht nur erste 5!)
for ($i = 0; $i -lt $tableRowStarts.Count; $i++) {
    $rowStartY = $tableRowStarts[$i]
    # rowEndY: Entweder naechste Tabellenzeile ODER lowestValidY (nicht tiefer als Footer!)
    $rowEndY = if ($i+1 -lt $tableRowStarts.Count) {
        $tableRowStarts[$i+1]
    } else {
        $lowestValidY - 1  # 1 Pixel oberhalb Footer
    }

    # Hole alle Chunks in diesem Y-Bereich
    $rowChunks = $chunks | Where-Object {
        $_.Y -le $rowStartY -and $_.Y -gt $rowEndY
    }

    # Sammle Zellinhalte
    $cells = @()
    for ($c = 0; $c -lt [Math]::Min(8, $columns.Count); $c++) {
        $col = $columns[$c]
        $colChunks = $rowChunks | Where-Object {
            $_.X -ge $col.XStart -and $_.X -lt $col.XEnd
        } | Sort-Object @{Expression="Y"; Descending=$true}, X

        # INTELLIGENTES Zusammenfuegen
        $cellText = ""
        $lastY = $null
        foreach ($chunk in $colChunks) {
            if ($lastY -ne $null -and [Math]::Abs($chunk.Y - $lastY) -gt 1.0) {
                $cellText += " "
            }
            $cellText += $chunk.Text
            $lastY = $chunk.Y
        }

        # RETROSPEKTIVE BEREINIGUNG: Nur bei der LETZTEN Tabellenzeile Footer-Bestandteile entfernen
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

$output | Out-File -FilePath $outputPath -Encoding UTF8

Write-Host "Fertig! -> $outputPath"
Read-Host "Enter"