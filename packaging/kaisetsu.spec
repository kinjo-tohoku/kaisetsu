# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定(Windows one-folder 配布)。

    uv run --group build pyinstaller packaging/kaisetsu.spec --noconfirm

出力は dist/kaisetsu/kaisetsu.exe。data/public(サンプルデータ)も同梱する。
インストーラまで作るときは packaging/build_windows.ps1 を使う。
PyInstaller 自体は GPL(ブートローダ例外付き)で、生成した配布物にはその制約は及ばない
(配布物のライセンスは kaisetsu と依存ライブラリのものになる)。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821  (SPECPATH は PyInstaller が与える)

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
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"],
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
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="kaisetsu",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "packaging" / "kaisetsu.ico"),
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="kaisetsu",
)
