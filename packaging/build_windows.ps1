# Windows 用の one-folder 配布物を作る。
#   .\packaging\build_windows.ps1
# 出力: dist\kaisetsu\kaisetsu.exe(フォルダごと zip にして配る)
# 事前に `uv sync --group build` で PyInstaller を入れておく。
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"
uv run --group build pyinstaller packaging\kaisetsu.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller が失敗しました" }
$version = (uv run python -c "import kaisetsu; print(kaisetsu.__version__)").Trim()
$zip = "dist\kaisetsu-$version-windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "dist\kaisetsu" -DestinationPath $zip
Write-Host "できました: $zip"
