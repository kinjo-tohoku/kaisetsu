"""ウィザードを 2 回完了したとき: 既定では新しいプロジェクトとして始める(レビュー 2026-10-17 GUI #6)。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from kaisetsu.gui import wizard as wizard_mod
from kaisetsu.gui.dialog_utils import DATA_FILTER
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.wizard import StartWizard


def _finish(qtbot, win: MainWindow, data: Path, cif: Path) -> StartWizard:  # type: ignore[no-untyped-def]
    wiz = StartWizard(win)
    qtbot.addWidget(wiz)
    wiz.page_data.edit.setText(str(data))
    wiz.page_phase.edit.setText(str(cif))
    wiz.page_inst.dialog.apply_preset_by_name("実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)")
    wiz.accept()
    return wiz


@pytest.mark.parametrize("mode", ["new", "append", None])
def test_second_wizard_run(qtbot, data_dir: Path, monkeypatch, mode: str | None) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    recipes: list[str] = []
    monkeypatch.setattr(
        win, "run_recipe", lambda recipe, companion=None: recipes.append(recipe.name)
    )
    asked: list[int] = []
    monkeypatch.setattr(wizard_mod, "ask_start_mode", lambda parent: asked.append(1) or mode)
    monkeypatch.setattr(win, "_ask", lambda box: QMessageBox.StandardButton.Discard)
    pbso4 = (data_dir / "pbso4" / "PBSO4.XRA", data_dir / "pbso4" / "PbSO4-Wyckoff.cif")
    fap = (data_dir / "fap" / "FAP.XRA", data_dir / "fap" / "fluorapatite_COD9001232.cif")
    wiz = _finish(qtbot, win, *pbso4)
    assert wiz.page_data._filter == DATA_FILTER
    assert asked == [] and len(win.project.histograms) == 1 and len(recipes) == 1
    first_phase = next(iter(win.project.phases))
    _finish(qtbot, win, *fap)
    assert asked == [1]
    names = [h.name for h in win.project.histograms]
    if mode == "new":
        assert names == ["FAP"] and first_phase not in win.project.phases
        assert len(win.project.phases) == 1 and len(recipes) == 2
    elif mode == "append":
        assert names == ["PBSO4", "FAP"] and len(win.project.phases) == 2 and len(recipes) == 2
    else:  # キャンセル: 何も変わらず、精密化も始まらない
        assert names == ["PBSO4"] and list(win.project.phases) == [first_phase]
        assert len(recipes) == 1


def test_wizard_aborts_when_data_cannot_be_loaded(
    qtbot, data_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    recipes: list[str] = []
    monkeypatch.setattr(
        win, "run_recipe", lambda recipe, companion=None: recipes.append(recipe.name)
    )
    monkeypatch.setattr(wizard_mod.QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    from kaisetsu.gui import main_window as mw

    crit: list[str] = []
    monkeypatch.setattr(mw.QMessageBox, "critical", staticmethod(lambda *a, **k: crit.append(a[2])))
    tiny = tmp_path / "tiny.xy"
    tiny.write_text("10.0 100\n10.1 120\n10.2 90\n", encoding="utf-8")  # 3 点しかない
    wiz = _finish(qtbot, win, tiny, data_dir / "pbso4" / "PbSO4-Wyckoff.cif")
    assert "読み込めません" in wiz.page_data.describe(str(tiny))
    assert crit and "3 点" in crit[0]
    assert not win.project.histograms and not win.project.phases and recipes == []
