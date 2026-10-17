"""GUI テストはオフスクリーンで動かす。"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# テスト中は起動時の更新確認で GitHub に行かない
os.environ.setdefault("KAISETSU_NO_UPDATE_CHECK", "1")
# offscreen プラットフォームはフォントを自動で見つけないので、Windows のフォントを指す
_win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if _win_fonts.exists():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_win_fonts))
# アプリの設定(窓の配置、前回のフォルダ)を利用者のレジストリではなく一時フォルダの ini に向ける。
# テストで閉じた窓の大きさが実際の設定に残らないようにする(kaisetsu.gui.settings.app_settings)
os.environ.setdefault(
    "KAISETSU_SETTINGS_FILE",
    str(Path(tempfile.mkdtemp(prefix="kaisetsu-test-settings-")) / "kaisetsu.ini"),
)


@pytest.fixture(autouse=True)
def _no_modal_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """閉じるときの「保存しますか?」を出さない(オフスクリーンではモーダルで止まる)。

    精密化やデータの読み込みをしたメインウィンドウは未保存の印が付くので、テストの後始末
    (qtbot が窓を閉じる)で確認ダイアログが出てしまう。既定では「保存しない」にし、確認の
    流れを見るテストはインスタンスの `_ask` を上書きする。
    """
    from PySide6.QtWidgets import QMessageBox

    from kaisetsu.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_ask", lambda self, box: QMessageBox.StandardButton.Discard)


def write_tiny_pattern(path: Path, scale: float = 1.0, seed: int = 0) -> Path:
    """小さな合成パターン(Ni、100 点)を 2 列テキストに書く(読み込みの最小点数 50 以上)。"""
    from kaisetsu.api import (
        AtomSite,
        ChebyshevBackground,
        CWInstrument,
        Lattice,
        PhaseContribution,
        Radiation,
        SampleGeometry,
        SpaceGroup,
        Structure,
        calculate_pattern,
    )

    st = Structure(
        Lattice.cubic(3.524),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)],
        name="Ni",
    )
    inst = CWInstrument(Radiation.xray(1.5406), w=0.01, s_l=0.01, h_l=0.01, polarization=0.7)
    x = np.linspace(40.0, 56.0, 100)
    bkg = ChebyshevBackground([30.0, -2.0], 40.0, 56.0)
    y = calculate_pattern(x, [PhaseContribution(st, scale=scale)], inst, SampleGeometry(), bkg)
    rng = np.random.default_rng(seed)
    np.savetxt(path, np.column_stack([x, rng.poisson(y.y_calc).astype(float)]), fmt="%.4f %.1f")
    return path


@pytest.fixture
def tiny_window(qtbot, tmp_path: Path, data_dir: Path):  # type: ignore[no-untyped-def]
    """小さな合成データ(100 点)と Ni の相を GUI の経路で読み込んだメインウィンドウ。"""
    from kaisetsu.api import CWInstrument, Radiation, SampleGeometry
    from kaisetsu.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    inst = CWInstrument(Radiation.xray(1.5406), w=0.01, s_l=0.01, h_l=0.01, polarization=0.7)
    assert win.load_data_file(
        str(write_tiny_pattern(tmp_path / "tiny.xy")), inst, SampleGeometry(), n_background=2
    )
    assert win.add_phase_file(str(data_dir / "pdf" / "Ni.cif"), name="Ni")
    return win
