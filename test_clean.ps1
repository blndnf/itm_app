# Test der Footer-Bereinigungsfunktion
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

Write-Host "===== TEST DER FOOTER-BEREINIGUNG ====="
Write-Host ""

# Test User-Spalte
$userBefore = "Joana Carvallo Date:2024-09-16"
$userAfter = Clean-FooterFromCell -cellText $userBefore -columnName "User"
Write-Host "User-Spalte:"
Write-Host "  Vorher: '$userBefore'"
Write-Host "  Nachher: '$userAfter'"
Write-Host "  Erwartet: 'Joana Carvallo'"
Write-Host "  OK: $($userAfter -eq 'Joana Carvallo')"
Write-Host ""

# Test Group-Spalte
$groupBefore = "ICPMHOperators 12:28:10"
$groupAfter = Clean-FooterFromCell -cellText $groupBefore -columnName "Group"
Write-Host "Group-Spalte:"
Write-Host "  Vorher: '$groupBefore'"
Write-Host "  Nachher: '$groupAfter'"
Write-Host "  Erwartet: 'ICPMHOperators'"
Write-Host "  OK: $($groupAfter -eq 'ICPMHOperators')"
Write-Host ""

# Test Time-Spalte
$timeBefore = "2021-03-11 06:38:29 GMT+01:00 GMT+02:00"
$timeAfter = Clean-FooterFromCell -cellText $timeBefore -columnName "Time(Local)"
Write-Host "Time-Spalte:"
Write-Host "  Vorher: '$timeBefore'"
Write-Host "  Nachher: '$timeAfter'"
Write-Host "  Erwartet: '2021-03-11 06:38:29 GMT+01:00'"
Write-Host "  OK: $($timeAfter -eq '2021-03-11 06:38:29 GMT+01:00')"
Write-Host ""

# Test Error-Spalte
$errorBefore = "Page1/4953"
$errorAfter = Clean-FooterFromCell -cellText $errorBefore -columnName "Error Message"
Write-Host "Error-Spalte:"
Write-Host "  Vorher: '$errorBefore'"
Write-Host "  Nachher: '$errorAfter'"
Write-Host "  Erwartet: ''"
Write-Host "  OK: $($errorAfter -eq '')"
Write-Host ""

# Test Error-Spalte mit Leerzeichen
$errorBefore2 = "Page 1/4953"
$errorAfter2 = Clean-FooterFromCell -cellText $errorBefore2 -columnName "Error Message"
Write-Host "Error-Spalte (mit Leerzeichen):"
Write-Host "  Vorher: '$errorBefore2'"
Write-Host "  Nachher: '$errorAfter2'"
Write-Host "  Erwartet: ''"
Write-Host "  OK: $($errorAfter2 -eq '')"
Write-Host ""

Write-Host "===== TEST ABGESCHLOSSEN ====="
