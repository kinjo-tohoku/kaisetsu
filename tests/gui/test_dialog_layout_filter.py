"""ダイアログの高さ対策と、パラメータ表の種類フィルタ・詳細パラメータのテスト(オフスクリーン)。"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout

from kaisetsu.gui.dialog_utils import available_height, fit_dialog_to_screen
from kaisetsu.gui.dialogs import InstrumentDialog
from kaisetsu.gui.param_model import ParameterFilter, ParameterModel
from kaisetsu.refine.parameters import Parameter, ParameterSet


def test_fit_dialog_to_screen_wraps_tall_dialog(qtbot) -> None:  # type: ignore[no-untyped-def]
    dlg = QDialog()
    qtbot.addWidget(dlg)
    layout = QVBoxLayout(dlg)
    for i in range(80):
        layout.addWidget(QLabel(f"row {i}"))
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    layout.addWidget(buttons)
    assert dlg.sizeHint().height() > available_height(dlg)
    assert fit_dialog_to_screen(dlg, buttons)
    assert dlg.height() <= available_height(dlg)
    # ボタンはスクロール領域の外(ダイアログ直下)に残り、中身はスクロール領域に入る
    assert buttons.parentWidget() is dlg
    scroll = dlg.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.widget().layout() is layout


def test_fit_dialog_to_screen_leaves_small_dialog(qtbot) -> None:  # type: ignore[no-untyped-def]
    dlg = QDialog()
    qtbot.addWidget(dlg)
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel("x"))
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    layout.addWidget(buttons)
    assert not fit_dialog_to_screen(dlg, buttons)
    assert dlg.findChild(QScrollArea) is None


def test_instrument_dialog_fits_laptop_screen(qtbot) -> None:  # type: ignore[no-untyped-def]
    dlg = InstrumentDialog(None, None)
    qtbot.addWidget(dlg)
    # 2 列化で、拡大率 150% のノート PC(有効 700 px 程度)にも収まる高さになっている
    assert dlg.sizeHint().height() <= 600


def _params() -> ParameterSet:
    ps = ParameterSet()
    for name in (
        "garnet:cell:a",
        "garnet:O1:x",
        "garnet:O1:uiso",
        "h0:inst:u",
        "h0:inst:x",
        "h0:inst:zero",
        "h0:sample:rough_a",
        "h0:garnet:scale",
        "g0:pdf:scale",
    ):
        ps.add(Parameter(name, 1.0))
    ps.add(Parameter("h0:inst:z", 0.0, advanced=True, neutral_value=0.0))
    ps.add(Parameter("h0:sample:rough_b", 1.0, advanced=True))
    return ps


def _names(f: ParameterFilter) -> list[str]:
    return [str(f.data(f.index(r, 0))) for r in range(f.rowCount())]


def test_filter_scope_separates_phase_and_histogram(qtbot) -> None:  # type: ignore[no-untyped-def]
    model = ParameterModel(_params())
    f = ParameterFilter()
    f.setSourceModel(model)
    f.set_phase_names(["garnet"])
    # 「U」ボタン: 相の U_iso だけ(装置の U は出ない)
    f.set_pattern("*:*:u*")
    f.set_scope("phase")
    assert _names(f) == ["garnet:O1:uiso"]
    # 「原子」ボタン: 分率座標だけ(装置の X, Y, Z は出ない)
    f.set_pattern("*:*:[xyz]")
    assert _names(f) == ["garnet:O1:x"]
    # ヒストグラム側
    f.set_scope("hist")
    f.set_pattern("*")
    names = _names(f)
    assert "garnet:cell:a" not in names
    assert "h0:inst:u" in names and "g0:pdf:scale" in names
    # 種類を外せば両方
    f.set_scope("all")
    f.set_pattern("*:*:scale")
    assert _names(f) == ["h0:garnet:scale", "g0:pdf:scale"]
    with pytest.raises(ValueError):
        f.set_scope("bogus")


def test_filter_hides_advanced_until_used(qtbot) -> None:  # type: ignore[no-untyped-def]
    ps = _params()
    model = ParameterModel(ps)
    f = ParameterFilter()
    f.setSourceModel(model)
    f.set_pattern("*")
    names = _names(f)
    assert "h0:inst:z" not in names and "h0:sample:rough_b" not in names
    # 効いている値(Z ≠ 0)なら出る
    ps["h0:inst:z"].value = 0.01
    model.refresh()
    assert "h0:inst:z" in _names(f)
    # 動かしていれば出る
    ps["h0:sample:rough_b"].vary = True
    model.refresh()
    assert "h0:sample:rough_b" in _names(f)
    ps["h0:sample:rough_b"].vary = False
    model.refresh()
    assert "h0:sample:rough_b" not in _names(f)
    # 「詳細も表示」なら常に出る
    f.set_show_advanced(True)
    assert "h0:sample:rough_b" in _names(f)
