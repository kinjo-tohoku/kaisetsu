# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定(Windows one-folder 配布)。

    uv run --group build pyinstaller packaging/kaisetsu.spec --noconfirm

出力は dist/kaisetsu/kaisetsu.exe(GUI)と kaisetsu-cli.exe(コマンドライン、同じフォルダを共有)。
data/public(サンプルデータ)も同梱する。
インストーラまで作るときは packaging/build_windows.ps1 を使う。
PyInstaller 自体は GPL(ブートローダ例外付き)で、生成した配布物にはその制約は及ばない
(配布物のライセンスは kaisetsu と依存ライブラリのものになる)。
"""

import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821  (SPECPATH は PyInstaller が与える)

# 配布物に入れてはいけない開発用パッケージ(ライセンス一覧に載らないまま同梱されていた:
# 外部レビュー 2026-10-17 #11)。uv sync の既定で dev グループが入るので、ビルドは
# `uv sync --no-default-groups --extra gui --group build` で行い、念のためここでも除外する。
DEV_EXCLUDES = [
    "hypothesis",
    "coverage",
    "pytest",
    "_pytest",
    "pytest_cov",
    "pytestqt",
    "setuptools",
    "pkg_resources",
    "_distutils_hack",
    "mypy",
]
# 配布物の Python を UTF-8 モードで動かす(ファイルの既定の文字コードを cp932 にしない)。
# 起動スクリプトで環境変数 PYTHONUTF8 を設定しても起動後では効かない(外部レビュー #14)。
# ブートローダが解釈する形は Python の -X と同じ "X utf8"("X utf8_mode=1" は効かない。実測)。
UTF8_OPTION = ("X utf8", None, "OPTION")

datas = []
datas += collect_data_files("gemmi")
datas += collect_data_files("numba")
datas += collect_data_files("pyqtgraph")
sample_dir = ROOT / "data" / "public"
if sample_dir.is_dir():
    datas.append((str(sample_dir), "data/public"))
docs_dir = ROOT / "docs" / "user-guide"  # ヘルプ → 利用者ガイドで表示する説明書
if docs_dir.is_dir():
    datas.append((str(docs_dir), "docs/user-guide"))

hiddenimports = []
hiddenimports += collect_submodules("kaisetsu")
hiddenimports += collect_submodules("scipy.special")
hiddenimports += collect_submodules("scipy.optimize")
hiddenimports += collect_submodules("scipy.sparse")
hiddenimports += ["pyqtgraph.exporters", "matplotlib.backends.backend_agg", "matplotlib.backends.backend_svg"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"]
    + DEV_EXCLUDES,
    noarchive=False,
)


def _keep_data(entry: tuple) -> bool:
    """tests ディレクトリと極端に長い名前のファイルは同梱しない。

    numba / scipy / matplotlib のテストデータで 100 MB 近く減る。pyqtgraph の
    "CC0 legal code - applies to ..." は名前が長く、深いフォルダに入れると Windows の
    パス長 260 文字を超えてインストールが失敗する(CC0 なので同梱義務は無い)。
    """
    dest = str(entry[0]).replace("\\", "/")
    parts = dest.split("/")
    if "tests" in parts[:-1] or "test" in parts[:-1]:
        return False
    return len(parts[-1]) <= 60


def _keep_module(entry: tuple) -> bool:
    name = str(entry[0])
    return ".tests." not in name and not name.endswith(".tests") and ".tests" not in name


a.datas = [e for e in a.datas if _keep_data(e)]
a.pure = [e for e in a.pure if _keep_module(e)]
for _name in DEV_EXCLUDES:
    assert not any(str(e[0]).split(".")[0] == _name for e in a.pure), f"{_name} が同梱される"
pyz = PYZ(a.pure)  # noqa: F821

# 実際に同梱したモジュール・バイナリ・データの一覧。packaging/collect_licenses.py が、
# 依存の閉包ではなくこれからライセンス一覧を作り、未知のものがあればビルドを失敗させる。
manifest = {
    "modules": sorted({str(e[0]) for e in a.pure}),
    "binaries": sorted({str(e[0]).replace("\\", "/") for e in a.binaries}),
    "datas": sorted({str(e[0]).replace("\\", "/") for e in a.datas}),
}
manifest_path = Path(workpath) / "bundle_manifest.json"  # noqa: F821  (workpath = build/kaisetsu)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [UTF8_OPTION],  # OPTION は位置引数の TOC として渡す(options= というキーワードは無視される)
    exclude_binaries=True,
    name="kaisetsu",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "packaging" / "kaisetsu.ico"),
)
# コマンドライン版: 同じ解析結果(pyz)を使い、起動スクリプトだけ差し替えたコンソール exe
cli_scripts = [s for s in a.scripts if not str(s[1]).replace("\\", "/").endswith("packaging/launch.py")]
cli_scripts.append(("launch_cli", str(ROOT / "packaging" / "launch_cli.py"), "PYSOURCE"))
exe_cli = EXE(  # noqa: F821
    pyz,
    cli_scripts,
    [UTF8_OPTION],
    exclude_binaries=True,
    name="kaisetsu-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(ROOT / "packaging" / "kaisetsu.ico"),
)
coll = COLLECT(  # noqa: F821
    exe,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="kaisetsu",
)
