"""ウィザードの「解析の種類」: Le Bail を選ぶと相が Le Bail になり、構造の段階を飛ばして精密化する。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from kaisetsu.gui import wizard as wizard_mod
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.wizard import StartWizard
from kaisetsu.refine.recipe import Recipe


def test_run_page_le_bail_disables_structure_options(qtbot) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    wiz = StartWizard(win)
    qtbot.addWidget(wiz)
    page = wiz.page_run
    assert page.rb_rietveld.isChecked() and not page.is_le_bail()
    assert page.coords.isEnabled() and page.lebail.isEnabled()
    assert "原子座標" in page.steps.text()

    page.rb_lebail.setChecked(True)
    assert page.is_le_bail()
    for w in (page.coords, page.adp, page.light, page.lebail):
        assert not w.isEnabled()
    assert "原子座標" not in page.steps.text()
    assert "構造は使いません" in page.steps.text()

    page.rb_rietveld.setChecked(True)
    assert page.coords.isEnabled() and page.lebail.isEnabled()
    assert "原子座標" in page.steps.text()


@pytest.mark.slow
def test_wizard_le_bail_mode_refines_without_structure(qtbot, data_dir: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def short_recipe(*_args: object, **kw: object) -> Recipe:
        captured.update(kw)
        return (
            Recipe("short")
            .add("bg", ["h*:bkg:*", "h*:*:scale"], 3)
            .add("cell", ["*:cell:*", "h*:inst:zero"], 3)
        )

    monkeypatch.setattr(wizard_mod, "standard_recipe", short_recipe)
    win = MainWindow()
    qtbot.addWidget(win)
    wiz = StartWizard(win)
    qtbot.addWidget(wiz)
    wiz.page_data.edit.setText(str(data_dir / "pbso4" / "PBSO4.XRA"))
    wiz.page_phase.edit.setText(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    wiz.page_inst.dialog.apply_preset_by_name("実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)")
    wiz.page_run.rb_lebail.setChecked(True)

    wiz.accept()
    ph = win.project.phases["PbSO4"]
    assert ph.le_bail and not ph.pawley
    # 構造パラメータの段階と Le Bail 並走は外れる
    assert captured["coordinates"] is False and captured["adp"] is False
    assert captured["light_atom_uiso"] is False
    assert any("Le Bail にしました" in line for line in win.project.log)
    scale = [p for p in win.project.params if p.name == "h0:PbSO4:scale"]
    assert len(scale) == 1 and scale[0].locked

    assert win._thread is not None
    qtbot.waitUntil(
        lambda: win._thread is not None and not win._thread.isRunning(), timeout=180_000
    )
    qtbot.waitUntil(lambda: win.act_refine.isEnabled(), timeout=5000)
    assert win.project.last_fit is not None
    rf = win.project.rfactors()["total"]
    assert rf.rwp < 0.3
    assert not any("Le Bail(構造なし)" in line for line in win.project.log)
