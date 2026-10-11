"""ウィザードの流れ(データ → CIF → 装置 → 実行)をオフスクリーンで通す。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from kaisetsu.gui import wizard as wizard_mod
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.wizard import StartWizard
from kaisetsu.refine.recipe import Recipe

pytestmark = pytest.mark.slow


def test_wizard_sets_up_project_and_starts_refinement(qtbot, data_dir: Path, monkeypatch) -> None:
    # 時間短縮のため標準レシピを 2 段階に置き換える
    def short_recipe(*_args: object, **_kw: object) -> Recipe:
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
    assert "点" in wiz.page_data.describe(wiz.page_data.edit.text())
    wiz.page_phase.edit.setText(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    assert "P n m a" in wiz.page_phase.describe(wiz.page_phase.edit.text())
    wiz.page_inst.dialog.apply_preset_by_name("実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)")
    wiz.page_run.coords.setChecked(False)
    wiz.page_run.adp.setChecked(False)

    wiz.accept()  # 完了 → プロジェクト構築 → 精密化開始
    assert len(win.project.histograms) == 1
    assert "PbSO4" in win.project.phases
    assert win._thread is not None
    qtbot.waitUntil(
        lambda: win._thread is not None and not win._thread.isRunning(), timeout=180_000
    )
    qtbot.waitUntil(lambda: win.act_refine.isEnabled(), timeout=5000)
    assert win.project.last_fit is not None
    rf = win.project.rfactors()["total"]
    assert rf.rwp < 0.6
    assert any("レシピ" in line for line in win.project.log)
