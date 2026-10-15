"""残差の点検ダイアログ: 候補を表に出し、選んだピークを除外領域に、こぶをハンプにする(オフスクリーン)。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.gui.dialogs import RecipeDialog
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.residual_dialog import ResidualDialog
from kaisetsu.gui.wizard import StartWizard
from kaisetsu.pattern.histogram import PowderData

ROOT = Path(__file__).resolve().parents[2]
LAB6_CIF = ROOT / "data" / "public" / "lab6" / "LaB6_COD1000055.cif"
WIDTHS = {"u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02}


def _gauss(x: np.ndarray, c: float, fwhm: float, h: float) -> np.ndarray:
    s = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.asarray(h * np.exp(-0.5 * ((x - c) / s) ** 2))


def _project() -> Project:
    st = Structure.from_cif(str(LAB6_CIF))
    x = np.arange(20.0, 80.0, 0.02)
    inst = CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **WIDTHS)
    sim = Project("sim")
    sim.add_phase(st.copy(), name="L")
    sim.add_histogram(PowderData(x, np.ones_like(x)), inst, SampleGeometry(), n_background=2)
    sim.build_parameters()
    y_peaks = sim.calculate(0).y_bragg
    y_true = 3000.0 * y_peaks / y_peaks.max() + 20.0
    y_true += _gauss(x, 47.3, 0.15, 600.0) + _gauss(x, 60.0, 5.0, 300.0)
    y = np.random.default_rng(3).poisson(y_true).astype(np.float64)
    proj = Project("fit")
    proj.add_phase(st.copy(), name="L")
    proj.add_histogram(
        PowderData(x, y),
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **WIDTHS),
        SampleGeometry(),
        n_background=3,
    )
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:L:scale")
    proj.refine(max_cycles=8)
    return proj


def test_residual_dialog_applies_exclusion_and_hump(qtbot) -> None:
    proj = _project()
    dlg = ResidualDialog(proj)
    qtbot.addWidget(dlg)
    kinds = [k for k, _, _ in dlg._rows]
    assert "未同定ピーク" in kinds and "こぶ" in kinds
    assert dlg.exclude_btn.isEnabled() and dlg.hump_btn.isEnabled()
    feats = dlg.features[0]
    i_peak = next(i for i, p in enumerate(feats.peaks) if p.unindexed)
    dlg.set_checked("未同定ピーク", 0, i_peak)
    assert dlg.apply_excludes() == 1
    h = proj.histograms[0]
    assert len(h.excluded) == 1 and h.excluded[0][0] < 47.3 < h.excluded[0][1]
    assert any("除外領域を追加" in line for line in proj.log)
    # 再検出のあと、そのピークは候補から消えている
    assert not any(p.unindexed and abs(p.position - 47.3) < 0.2 for p in dlg.features[0].peaks)

    dlg.set_checked("こぶ", 0, 0)
    assert dlg.apply_humps() == 1
    assert len(h.humps) == 1 and abs(h.humps[0].center - 60.0) < 1.0
    assert "h0:hump0:center" in proj.params
    assert any("ハンプを追加" in line for line in proj.log)


def test_main_window_has_residual_action_and_recipe_options(qtbot) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.act_residual.text()
    dlg = RecipeDialog(["L"], win)
    qtbot.addWidget(dlg)
    assert dlg.ka2.isChecked() and dlg.light.isChecked()
    names = [s.name for s in dlg.recipe().steps]
    assert "Kα2 なしを試す" in names and "非対称(放射光)" in names
    dlg.ka2.setChecked(False)
    dlg.asym.setChecked(True)
    names = [s.name for s in dlg.recipe().steps]
    assert "Kα2 なしを試す" not in names and "非対称" in names
    wiz = StartWizard(win)
    qtbot.addWidget(wiz)
    assert wiz.page_run.n_background.value() == 12
    assert wiz.page_run.ka2.isChecked() and wiz.page_run.light.isChecked()
    assert wiz.page_run.lebail.isChecked()
