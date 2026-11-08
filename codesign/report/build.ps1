# Build the report with Tectonic (self-contained LaTeX engine; fetches packages and runs BibTeX itself).
#   build.cmd            build main.tex once -> main.pdf
#   build.cmd watch      rebuild whenever a .tex/.bib/.sty file or a figure in figs/ changes (Ctrl+C to stop)
param([switch]$Watch, [string]$Main = "main.tex")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$tectonic = (Get-Command tectonic -ErrorAction SilentlyContinue).Source
if (-not $tectonic) {
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\tectonic\tectonic.exe"
    if (Test-Path $fallback) { $tectonic = $fallback }
}
if (-not $tectonic) {
    Write-Host "Tectonic not found. Install it (one time), then rerun:" -ForegroundColor Red
    Write-Host "  https://github.com/tectonic-typesetting/tectonic/releases  ->  tectonic-*-x86_64-pc-windows-msvc.zip"
    Write-Host "  unzip tectonic.exe into $env:LOCALAPPDATA\Programs\tectonic\"
    exit 1
}

function Invoke-Build {
    $t0 = Get-Date
    Write-Host "[$($t0.ToString('HH:mm:ss'))] building $Main ..." -ForegroundColor Cyan
    & $tectonic -X compile $Main --synctex
    $ok = ($LASTEXITCODE -eq 0)
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($ok) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] ok ($secs s) -> main.pdf" -ForegroundColor Green }
    else     { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] FAILED ($secs s)" -ForegroundColor Red }
    return $ok
}

function Get-Stamp {
    $files = @(Get-ChildItem -File -Include *.tex, *.bib, *.sty, *.bst -Path (Join-Path $PSScriptRoot "*"))
    if (Test-Path figs) { $files += Get-ChildItem -File -Recurse figs }
    ($files | ForEach-Object { "$($_.Name):$($_.LastWriteTimeUtc.Ticks)" }) -join "|"
}

if (-not $Watch) {
    if (Invoke-Build) { exit 0 } else { exit 1 }
}

Write-Host "Watching $PSScriptRoot for changes (Ctrl+C to stop)." -ForegroundColor Yellow
[void](Invoke-Build)
$last = Get-Stamp
while ($true) {
    Start-Sleep -Milliseconds 700
    $now = Get-Stamp
    if ($now -ne $last) {
        Start-Sleep -Milliseconds 300   # let the editor finish writing
        $last = Get-Stamp
        [void](Invoke-Build)
    }
}
