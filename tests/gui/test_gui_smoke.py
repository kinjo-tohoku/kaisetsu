"""GUI のスモークテスト(オフスクリーン)。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from kaisetsu.gui.dialogs import PRESETS, InstrumentDialog, RecipeDialog
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.param_model import COL_VALUE, COL_VARY
from kaisetsu.gui.worker import RefineWorker
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.scattering import Radiation

pytestmark = pytest.mark.slow


def _window(qtbot, data_dir: Path) -> MainWindow:
    win = MainWindow()
    qtbot.addWidget(win)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5),
        u=0.002,
        v=-0.002,
        w=0.004,
        x=0.02,
        y=0.02,
        s_l=0.01,
        h_l=0.01,
        polarization=0.7,
    )
    win.load_data_file(str(data_dir / "pbso4" / "PBSO4.XRA"), inst, SampleGeometry())
    win.add_phase_file(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"), name="PbSO4")
    return win


def test_main_window_loads_and_plots(qtbot, data_dir: Path, tmp_path: Path) -> None:
    win = _window(qtbot, data_dir)
    win.show()
    assert len(win.project.histograms) == 1 and "PbSO4" in win.project.phases
    assert win.param_model.rowCount() > 50
    # 初期スケールは観測に合わせて粗く決まる(極端に外れていない)
    pc = win.project.phase_contribution(0, "PbSO4")
    assert 0.01 < pc.scale < 10
    # 図に計算曲線が入っている
    assert win.plot._calc.xData is not None and len(win.plot._calc.xData) == len(
        win.project.histograms[0].data
    )
    assert len(win.plot._ticks) == 1
    # フィルタ
    win.filter_edit.setText("*:cell:*")
    assert win.param_filter.rowCount() == 6
    win.filter_edit.setText("h0:bkg:*")
    assert win.param_filter.rowCount() == 8
    win.only_free.setChecked(True)
    assert win.param_filter.rowCount() == 0
    win.only_free.setChecked(False)
    # 表示中を精密化する
    win._set_vary_shown(True)
    assert set(win.project.params.free_names()) == {f"h0:bkg:{k}" for k in range(8)}
    # チェックボックスの編集
    win.filter_edit.setText("h0:PbSO4:scale")
    idx = win.param_filter.index(0, COL_VARY)
    from PySide6.QtCore import Qt

    assert win.param_filter.setData(
        idx, Qt.CheckState.Checked.value, Qt.ItemDataRole.CheckStateRole
    )
    assert win.project.params["h0:PbSO4:scale"].vary
    # 値の編集はモデルに反映される
    vidx = win.param_filter.index(0, COL_VALUE)
    assert win.param_filter.setData(vidx, "0.5", Qt.ItemDataRole.EditRole)
    assert win.project.params["h0:PbSO4:scale"].value == pytest.approx(0.5)
    assert pc.scale == pytest.approx(0.5)
    # スクリーンショット(目視確認用)
    out = Path(os.environ.get("KAISETSU_SCREENSHOT_DIR", tmp_path)) / "main_window.png"
    win.resize(1400, 850)
    qtbot.wait(100)
    assert win.grab().save(str(out))


def test_refine_worker_runs_in_thread(qtbot, data_dir: Path, tmp_path: Path) -> None:
    win = _window(qtbot, data_dir)
    win.project.vary("h0:bkg:*", "h0:PbSO4:scale")
    worker = RefineWorker(win.project, None, max_cycles=5)
    messages: list[str] = []
    worker.progress.connect(messages.append)
    iters: list[int] = []
    worker.iteration.connect(lambda i, _c, _curves: iters.append(i))
    with qtbot.waitSignal(worker.finished, timeout=120_000) as blocker:
        win._start_worker(worker)
        assert win.progress_bar.isVisible() or not win.isVisible()
        qtbot.wait(400)
        shot = Path(os.environ.get("KAISETSU_SCREENSHOT_DIR", str(tmp_path))) / "refining.png"
        win.grab().save(str(shot))
    ok, msg = blocker.args
    assert ok, msg
    assert "Rwp" in msg
    assert win.project.last_fit is not None
    qtbot.waitUntil(lambda: not win.act_stop.isEnabled(), timeout=5000)
    assert win.progress_bar.isHidden()
    assert iters, "反復ごとの進捗シグナルが届いていない"
    assert "完了" in win.progress_label.text()


def test_instrument_and_recipe_dialogs(qtbot) -> None:
    dlg = InstrumentDialog(None, None)
    qtbot.addWidget(dlg)
    name = next(iter(PRESETS))
    dlg.apply_preset_by_name(name)
    inst, smp = dlg.result_instrument()
    assert inst.radiation.wavelength == pytest.approx(1.5405929)
    assert inst.radiation.wavelength2 is not None
    assert inst.polarization == pytest.approx(PRESETS[name]["polarization"], abs=1e-3)
    dlg.apply_preset_by_name("定波長中性子 (Debye-Scherrer)")
    inst, smp = dlg.result_instrument()
    assert inst.radiation.is_neutron and smp.kind.value == "debye-scherrer"
    rd = RecipeDialog(["PbSO4"])
    qtbot.addWidget(rd)
    rd.disp.setChecked(True)
    r = rd.recipe()
    names = [s.name for s in r.steps]
    assert names[0] == "スケールと背景" and "原子座標" in names
    assert any("sample:displacement" in p for s in r.steps for p in s.patterns)


def test_background_dialog_and_reset(qtbot, data_dir: Path) -> None:
    from kaisetsu.gui.background_dialog import BackgroundDialog

    win = _window(qtbot, data_dir)
    assert len(win.project.params.select("h0:bkg:*")) == 8
    # 読み込み時の背景推定は低角の立ち上がりを捉えている
    h = win.project.histograms[0]
    assert h.background(h.data.x[:1])[0] > 110.0
    dlg = BackgroundDialog(8)
    qtbot.addWidget(dlg)
    dlg.n_terms.setValue(12)
    assert dlg.values() == (12, True)
    win.project.reset_background(0, 12, True)
    win._refresh_all()
    assert len(win.project.params.select("h0:bkg:*")) == 12
    win.project.reset_background(0, 5, False)
    assert len(win.project.params.select("h0:bkg:*")) == 5
