# Used by docs/build_document.py. Opens the document READ-ONLY in Word, exports a PDF
# and writes the page number of every heading (adjusted numbers, as shown in the footer).
param([string]$In, [string]$Pdf, [string]$Pages)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open($In, $false, $true)
  $d.Repaginate()
  $d.ExportAsFixedFormat($Pdf, 17)
  $out = @()
  foreach ($p in $d.Paragraphs) {
    if ($p.OutlineLevel -le 2) { $out += "{0}`t{1}`t{2}" -f $p.OutlineLevel, $p.Range.Information(1), $p.Range.Text.Trim() }
  }
  $out += "TOTAL`t$($d.ComputeStatistics(2))`t"
  $out | Set-Content -Encoding utf8 $Pages
  $d.Close($false)
} finally { $w.Quit() }
