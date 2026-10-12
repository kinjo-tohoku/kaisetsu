"""除外領域の GUI(図の帯、数値ダイアログ)のテスト(オフスクリーン)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.range_dialog import RangeDialog
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.scattering import Radiation

pytestmark = pytest.mark.slow


def _window(qtbot, data_dir: Path) -> MainWindow:  # type: ignore[no-untyped-def]
    win = MainWindow()
    qtbot.addWidget(win)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7
    )
    win.load_data_file(str(data_dir / "pbso4" / "PBSO4.XRA"), inst, SampleGeometry())
    win.add_phase_file(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"), name="PbSO4")
    return win


def test_shift_drag_adds_region_and_right_click_removes(qtbot, data_dir: Path) -> None:  # type: ignore[no-untyped-def]
    win = _window(qtbot, data_dir)
    h = win.project.histograms[0]
    n_all = int(np.count_nonzero(h.mask()))
    # Shift + ドラッグ相当(ViewBox からのシグナル)
    win.plot.excludeRequested.emit(30.0, 32.0)
    assert h.excluded == [(30.0, 32.0)]
    assert len(win.plot._regions) == 1
    m = h.mask()
    x = h.data.x
    assert not m[(x >= 30.0) & (x <= 32.0)].any()
    assert int(np.count_nonzero(m)) < n_all
    # 重なる範囲を足すと 1 つにまとまる
    win.plot.excludeRequested.emit(31.0, 33.5)
    assert h.excluded == [(30.0, 33.5)]
    assert len(win.plot._regions) == 1
    # 帯を動かした(帯の現在位置が反映される)
    win.plot._regions[0].setRegion((40.0, 41.0))
    win.plot._on_region_changed(win.plot._regions[0])
    assert h.excluded == [(40.0, 41.0)]
    # 右クリック相当で削除
    win.plot._on_region_removed(win.plot._regions[0])
    assert h.excluded == []
    assert len(win.plot._regions) == 0
    assert int(np.count_nonzero(h.mask())) == n_all


def test_range_dialog_roundtrip(qtbot, data_dir: Path) -> None:  # type: ignore[no-untyped-def]
    win = _window(qtbot, data_dir)
    h = win.project.histograms[0]
    h.set_excluded([(50.0, 52.0), (20.0, 21.0)])
    dlg = RangeDialog(h, win)
    qtbot.addWidget(dlg)
    assert dlg.table.rowCount() == 2
    # 逆順に書いても直す。数値でない行は無視
    dlg._add_row()
    dlg.table.item(2, 0).setText("70")
    dlg.table.item(2, 1).setText("69")
    dlg._add_row()
    dlg.table.item(3, 0).setText("abc")
    dlg.x_min.setValue(15.0)
    dlg.x_max.setValue(100.0)
    x_min, x_max, ranges = dlg.values()
    assert (x_min, x_max) == (15.0, 100.0)
    assert ranges == [(20.0, 21.0), (50.0, 52.0), (69.0, 70.0)]
    h.x_min, h.x_max = x_min, x_max
    h.set_excluded(ranges)
    assert h.excluded == [(20.0, 21.0), (50.0, 52.0), (69.0, 70.0)]
    win._refresh_plot()
    assert len(win.plot._regions) == 3
    # 範囲外の帯(両端)が描かれる
    assert len(win.plot._range_shades) == 2
    # 選択行の削除
    dlg.table.selectRow(0)
    dlg._remove_selected()
    assert dlg.table.rowCount() == 3
