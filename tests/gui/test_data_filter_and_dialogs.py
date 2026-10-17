"""ファイルダイアログのフィルタ(対応形式から生成)と、装置ダイアログの往復(レビュー 2026-10-17 GUI #8, #17)。"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from kaisetsu.gui.dialog_utils import DATA_FILTER, data_file_filter
from kaisetsu.gui.dialogs import InstrumentDialog
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.io.readers import SUPPORTED_EXTENSIONS
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.scattering import Radiation


def test_data_filter_lists_every_supported_extension() -> None:
    filt = data_file_filter()
    assert filt == DATA_FILTER and filt.startswith("回折データ (") and ";;すべて (*)" in filt
    for ext in SUPPORTED_EXTENSIONS:
        assert f"*{ext}" in filt.lower()
    for ext in ("*.ras", "*.rasx", "*.xrdml", "*.csv", "*.int", "*.histogramIgor", "*.cif"):
        assert ext in filt
    # ウィザード・逐次精密化・メインで同じものを使う
    from kaisetsu.gui import main_window, sequence_dialog, wizard

    assert main_window.DATA_FILTER is DATA_FILTER
    assert sequence_dialog.DATA_FILTER is DATA_FILTER
    assert wizard.DATA_FILTER is DATA_FILTER


def _refined_instrument() -> tuple[CWInstrument, SampleGeometry]:
    """精密化した値のように、表示桁より細かい値を持つ装置と試料。"""
    inst = CWInstrument(
        Radiation.xray(1.5405929, 1.5444274, 0.5),
        zero=0.012345678, u=0.00123456789, v=-0.0034567891, w=0.0045678912,
        x=0.0212345, y=0.0312345, s_l=0.0123, h_l=0.0234, polarization=0.7123456,
    )  # fmt: skip
    smp = SampleGeometry(displacement_mm=0.0123456, roughness_a=0.9123456)
    return inst, smp


def test_instrument_dialog_keeps_unchanged_values(qtbot) -> None:
    inst, smp = _refined_instrument()
    dlg = InstrumentDialog(inst, smp)
    qtbot.addWidget(dlg)
    out_inst, out_smp = dlg.result_instrument()
    assert out_inst == inst and out_smp == smp  # 開いて OK を押しただけなら何も丸まらない
    assert out_inst.s_l == inst.s_l and out_inst.h_l == inst.h_l
    # 1 つ変えると、それだけが変わる
    dlg.u.setValue(0.002)
    dlg.displacement.setValue(0.05)
    out2, smp2 = dlg.result_instrument()
    assert out2.u == pytest.approx(0.002) and out2.v == inst.v and out2.zero == inst.zero
    assert out2.radiation == inst.radiation
    assert smp2.displacement_mm == pytest.approx(0.05) and smp2.roughness_a == smp.roughness_a
    # SH/L を変えると S/L = H/L に分ける(これまでどおり)
    dlg.sh_l.setValue(0.05)
    out3, _ = dlg.result_instrument()
    assert out3.s_l == pytest.approx(0.025) and out3.h_l == pytest.approx(0.025)


def test_edit_instrument_without_change_keeps_esd(
    qtbot, tiny_window: MainWindow, monkeypatch
) -> None:
    win = tiny_window
    h = win.project.histograms[0]
    h.instrument.u = 0.00123456789
    win.project.build_parameters()
    win.project.params["h0:inst:u"].esd = 0.0001
    win._set_dirty(False)
    monkeypatch.setattr(InstrumentDialog, "exec", lambda self: InstrumentDialog.DialogCode.Accepted)
    win.edit_instrument()
    assert h.instrument.u == 0.00123456789
    assert win.project.params["h0:inst:u"].esd == 0.0001
    assert not win.is_dirty() and "変更なし" in win.log_view.toPlainText()
