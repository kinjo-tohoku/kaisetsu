# Windows 用の配布物を作る: PyInstaller(one-folder)→ 起動確認 → Inno Setup のインストーラ → zip。
#   .\packaging\build_windows.ps1
# 出力: dist\kaisetsu-<ver>-windows-setup.exe、dist\kaisetsu-<ver>-windows.zip
# 依存はこのスクリプトが `uv sync --no-default-groups --extra gui --group build` で入れ直す
# (開発用の hypothesis / coverage / pytest を配布物に入れないため。終わったあと `uv run` は
# 自動で dev グループを戻す)。Inno Setup 6 は `winget install JRSoftware.InnoSetup` で入れておく。
# 見つからなければインストーラは飛ばす。
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"

# 0. 依存(dev グループ無し)。以降の uv run は --no-sync で、同期し直して dev が戻らないようにする
uv sync --no-default-groups --extra gui --group build
if ($LASTEXITCODE -ne 0) { throw "uv sync に失敗しました" }

$version = (uv run --no-sync python -c "import kaisetsu; print(kaisetsu.__version__)").Trim()
Write-Host "== kaisetsu $version"

# 1. PyInstaller
if (-not (Test-Path "packaging\kaisetsu.ico")) {
    uv run --no-sync python packaging\make_icon.py
    if ($LASTEXITCODE -ne 0) { throw "アイコンの作成に失敗しました" }
}
uv run --no-sync pyinstaller packaging\kaisetsu.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller が失敗しました" }

# 1b. 開発用パッケージが混ざっていないか(spec の excludes と uv sync の両方で防いでいるが、念のため)
foreach ($bad in @("hypothesis", "coverage", "pytest", "_pytest", "setuptools", "pkg_resources")) {
    if (Test-Path "dist\kaisetsu\_internal\$bad") { throw "開発用パッケージ $bad が配布物に入っています" }
}

# 2. ライセンス文書を同梱(実際に同梱したものの一覧 build\kaisetsu\bundle_manifest.json から作る。
#    出どころの分からないものがあれば失敗する)
uv run --no-sync python packaging\collect_licenses.py dist\THIRD_PARTY_LICENSES.txt `
    --manifest build\kaisetsu\bundle_manifest.json --bundle-dir dist\kaisetsu\_internal
if ($LASTEXITCODE -ne 0) { throw "ライセンス一覧の作成に失敗しました" }
$lic = Get-Content dist\THIRD_PARTY_LICENSES.txt -Raw -Encoding utf8
foreach ($must in @("GNU LESSER GENERAL PUBLIC LICENSE", "GNU GENERAL PUBLIC LICENSE", "OpenSSL", "HarfBuzz")) {
    if (-not $lic.Contains($must)) { throw "ライセンス一覧に $must がありません" }
}
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

# 3b. コマンドライン版の確認(サンプルデータで 1 回計算する)
Write-Host "== コマンドライン版の確認"
& "dist\kaisetsu\kaisetsu-cli.exe" selftest
if ($LASTEXITCODE -ne 0) { throw "kaisetsu-cli.exe selftest に失敗しました(終了コード $LASTEXITCODE)" }

# 3c. 配布物の Python が UTF-8 モードか(spec の options で有効にしている)
$utf8 = (& "dist\kaisetsu\kaisetsu-cli.exe" --utf8-mode-check)
if ($LASTEXITCODE -ne 0 -or -not "$utf8".Trim().StartsWith("utf8_mode=1")) {
    throw "配布物の Python が UTF-8 モードではありません: $utf8"
}
Write-Host "  $utf8"

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
