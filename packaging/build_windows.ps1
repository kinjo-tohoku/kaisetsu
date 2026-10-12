# Windows 用の配布物を作る: PyInstaller(one-folder)→ 起動確認 → Inno Setup のインストーラ → zip。
#   .\packaging\build_windows.ps1
# 出力: dist\kaisetsu-<ver>-windows-setup.exe、dist\kaisetsu-<ver>-windows.zip
# 事前に `uv sync --extra gui --group build` で PyInstaller を入れ、Inno Setup 6 を入れておく
# (`winget install JRSoftware.InnoSetup`)。Inno Setup が見つからなければインストーラは飛ばす。
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"

$version = (uv run python -c "import kaisetsu; print(kaisetsu.__version__)").Trim()
Write-Host "== kaisetsu $version"

# 1. PyInstaller
if (-not (Test-Path "packaging\kaisetsu.ico")) {
    uv run python packaging\make_icon.py
    if ($LASTEXITCODE -ne 0) { throw "アイコンの作成に失敗しました" }
}
uv run --group build pyinstaller packaging\kaisetsu.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller が失敗しました" }

# 2. ライセンス文書を同梱
uv run python packaging\collect_licenses.py dist\THIRD_PARTY_LICENSES.txt
if ($LASTEXITCODE -ne 0) { throw "ライセンス一覧の作成に失敗しました" }
Copy-Item LICENSE dist\kaisetsu\LICENSE.txt -Force
Copy-Item dist\THIRD_PARTY_LICENSES.txt dist\kaisetsu\ -Force

# 3. 起動確認(ウィンドウを出してサンプルデータのパターンを 1 回計算する)
Write-Host "== 起動確認"
$smoke = Join-Path ([System.IO.Path]::GetTempPath()) "kaisetsu_smoke_$PID.txt"
if (Test-Path $smoke) { Remove-Item $smoke }
if ($env:CI) {
    # CI にはデスクトップが無いので offscreen で
    $env:QT_QPA_PLATFORM = "offscreen"
    $env:QT_QPA_FONTDIR = Join-Path $env:WINDIR "Fonts"
}
$proc = Start-Process -FilePath "dist\kaisetsu\kaisetsu.exe" -ArgumentList @("--smoke-test", "`"$smoke`"") -PassThru
if (-not $proc.WaitForExit(300000)) { $proc.Kill(); throw "起動確認がタイムアウトしました(5 分)" }
if (-not (Test-Path $smoke)) { throw "起動確認の結果ファイルがありません(終了コード $($proc.ExitCode))" }
$result = Get-Content $smoke -Raw
Write-Host $result
Remove-Item $smoke
if ($proc.ExitCode -ne 0 -or -not $result.StartsWith("ok ")) {
    throw "起動確認に失敗しました(終了コード $($proc.ExitCode))"
}

# 4. Inno Setup
$candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$iscc = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $iscc) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { $iscc = $cmd.Source }
}
if ($iscc) {
    Write-Host "== Inno Setup: $iscc"
    & $iscc "/DAppVersion=$version" "packaging\kaisetsu.iss"
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup が失敗しました" }
} else {
    Write-Warning "Inno Setup 6 が見つからないのでインストーラは作りません(winget install JRSoftware.InnoSetup)"
}

# 5. zip(インストーラが使えない環境向け)
$zip = "dist\kaisetsu-$version-windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "dist\kaisetsu" -DestinationPath $zip
Write-Host "== できました:"
Get-ChildItem dist\*-windows-setup.exe, dist\*-windows.zip -ErrorAction SilentlyContinue |
    ForEach-Object { "  {0}  ({1:N0} MB)" -f $_.Name, ($_.Length / 1MB) }
