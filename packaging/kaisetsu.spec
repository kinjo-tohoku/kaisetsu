# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定(Windows one-folder 配布)。

    uv run --group build pyinstaller packaging/kaisetsu.spec --noconfirm

出力は dist/kaisetsu/kaisetsu.exe。data/public(サンプルデータ)も同梱する。
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
    icon=None,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="kaisetsu",
)
