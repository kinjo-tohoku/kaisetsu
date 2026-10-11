"""GUI の新機能(オフスクリーン): PDF 表示、χ² 推移、構造 3D、相の設定、PDF/変換/MEM ダイアログ、画像保存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt

from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.mem_dialog import DensityView, MEMDialog
from kaisetsu.gui.pdf_dialogs import PDFDataDialog, PDFRecipeDialog, TransformDialog
from kaisetsu.gui.phase_dialog import PhaseDialog
from kaisetsu.gui.plot_widget import HistoryPlot
from kaisetsu.gui.structure_view import StructureView
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.pdf import read_pdf_data
from kaisetsu.scattering import Radiation

pytestmark = pytest.mark.slow


def _pbso4_window(qtbot, data_dir: Path) -> MainWindow:
    win = MainWindow()
    qtbot.addWidget(win)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), u=0.002, v=-0.002, w=0.004, x=0.02, y=0.02,
        s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip
    win.load_data_file(str(data_dir / "pbso4" / "PBSO4.XRA"), inst, SampleGeometry())
    win.add_phase_file(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"), name="PbSO4")
    return win


def test_pdf_dataset_in_main_window(qtbot, data_dir: Path, tmp_path: Path) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    win.add_phase_file(str(data_dir / "pdf" / "Ni.cif"), name="Ni")
    win.load_pdf_file(
        str(data_dir / "pdf" / "Ni-xray.gr"),
        {
            "radiation": "xray",
            "qmax": 40.0,
            "qdamp": 0.06,
            "qbroad": 0.0,
            "r_min": 1.5,
            "r_max": 20.0,
        },
    )
    assert len(win.project.pdf_datasets) == 1
    assert win.current_view == "pdf"
    assert win.plot_stack.currentWidget() is win.pdf_plot
    assert win.pdf_plot._calc.xData is not None and len(win.pdf_plot._calc.xData) > 100
    assert "Rw" in win.status_label.text()
    assert win.act_pdf_recipe.isEnabled() and not win.act_recipe.isEnabled()
    # ツリーに PDF データが出る
    root = win.tree.topLevelItem(0)
    labels = [root.child(i).text(0) for i in range(root.childCount())]
    assert "PDF データ" in labels
    # 画像保存(PNG と SVG)
    png = win.pdf_plot.export_image(tmp_path / "g.png")
    assert png.exists() and png.stat().st_size > 1000
    svg = win.pdf_plot.export_image(tmp_path / "g.svg")
    assert svg.exists() and b"<svg" in svg.read_bytes()[:500]
    # CSV 書き出し先の切り替え(PDF 表示中は G(r))
    win.filter_edit.setText("g0:*")
    assert win.param_filter.rowCount() >= 8
    # PDF の標準手順ダイアログ
    dlg = PDFRecipeDialog(["Ni"])
    qtbot.addWidget(dlg)
    r = dlg.recipe()
    assert r.steps[0].name == "スケール" and any("delta2" in p for s in r.steps for p in s.patterns)
    # 短い精密化を走らせる(スケールだけ)
    win.project.fix_all()
    win.project.vary("g0:pdf:scale")
    from kaisetsu.gui.worker import RefineWorker

    worker = RefineWorker(win.project, None, max_cycles=5)
    with qtbot.waitSignal(worker.finished, timeout=120_000) as blocker:
        win._start_worker(worker)
    ok, msg = blocker.args
    assert ok and "Rw" in msg
    assert win.history_plot.values, "χ² の推移が記録されていない"


def test_pdf_data_dialog_defaults(qtbot, data_dir: Path) -> None:
    data = read_pdf_data(data_dir / "pdf" / "Ni-neutron.gr")
    dlg = PDFDataDialog(data)
    qtbot.addWidget(dlg)
    v = dlg.values()
    assert v["qmax"] == pytest.approx(27.0)
    assert v["r_min"] == pytest.approx(1.0) and v["r_max"] == pytest.approx(20.0)
    dlg.radiation.setCurrentIndex(1)
    assert dlg.values()["radiation"] == "neutron"


def test_transform_dialog_settings_and_run(qtbot, tmp_path: Path) -> None:
    # 合成 I(Q): Ni の Debye 散乱に近い簡単なもの(ピークが 1 本)
    q = np.arange(0.3, 25.0, 0.01)
    iq = 28.0**2 * np.exp(-((q / 20) ** 2)) * (1 + 0.3 * np.sin(q * 2.49) / (q * 2.49) * 12)
    p = tmp_path / "iq.dat"
    np.savetxt(p, np.column_stack([q, iq]))
    dlg = TransformDialog(None, "Ni")
    qtbot.addWidget(dlg)
    dlg.sample_edit.setText(str(p))
    dlg.qmax.setValue(24.0)
    dlg.rmax.setValue(15.0)
    s = dlg.settings()
    assert s.composition == "Ni" and s.qmax == 24.0 and s.polarization is None
    dlg.run_transform()
    assert dlg.result is not None
    data = dlg.pdf_data()
    assert data is not None and data.r[-1] == pytest.approx(15.0)
    assert data.metadata["qmax"] == 24.0


def test_structure_view_and_history(qtbot, data_dir: Path) -> None:
    from kaisetsu.crystal.cif import read_structure

    view = StructureView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    view.show()
    view.set_structure(read_structure(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    assert len(view._atoms) >= 24 and len(view._bonds) > 0
    img = view.grab().toImage()
    assert img.width() == 400
    # 回転・拡大の操作で例外が出ない
    view._last = QPointF(10, 10)
    from PySide6.QtGui import QMouseEvent

    ev = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(30, 20), Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )  # fmt: skip
    view.mouseMoveEvent(ev)
    view.reset_view()
    view.set_structure(None)
    hist = HistoryPlot()
    qtbot.addWidget(hist)
    hist.append(10.0)
    hist.mark_step("格子")
    hist.append(2.0)
    assert hist.values == [10.0, 2.0]
    hist.clear()
    assert hist.values == []


def test_phase_dialog_modes_and_aniso(qtbot, data_dir: Path) -> None:
    win = _pbso4_window(qtbot, data_dir)
    dlg = PhaseDialog(win.project, "PbSO4", 0)
    qtbot.addWidget(dlg)
    dlg.mode.setCurrentIndex(2)  # Pawley
    dlg.axis[0].setValue(0)
    dlg.axis[1].setValue(1)
    dlg.axis[2].setValue(0)
    dlg.site_boxes["Pb"].setChecked(True)
    dlg.radius.setValue(3.0)
    dlg.sh_enable.setChecked(True)
    dlg.sh_order.setValue(4)
    dlg.size_uniaxial.setChecked(True)
    dlg.size_axis[0].setValue(0)
    dlg.size_axis[1].setValue(1)
    dlg.size_axis[2].setValue(0)
    notes = dlg.apply()
    ph = win.project.phases["PbSO4"]
    assert ph.pawley and not ph.le_bail
    pc = win.project.histograms[0].contributions["PbSO4"]
    assert pc.march_dollase_axis == (0, 1, 0)
    assert pc.sh_order == 4 and pc.sh_coefficients is not None and len(pc.sh_coefficients) == 5
    assert "h0:PbSO4:sh2_1" in win.project.params.names()
    assert pc.size_axial_um is not None and pc.size_axis == (0, 1, 0)
    assert "h0:PbSO4:size_ax" in win.project.params.names()
    assert ph.structure.site("Pb").u_aniso is not None
    assert "PbSO4:Pb:u11" in win.project.params.names()
    assert ph.brindley_radius_um == 3.0
    assert any("Pawley" in n for n in notes)
    win._refresh_all()
    # 計算は Pawley の線形解を含む
    res = win.project.calculate(0)
    assert 0 in ph.pawley_solutions and len(res.peaks[0].hkl) > 0
    # 構造モデルに戻し、等方性に戻す
    dlg2 = PhaseDialog(win.project, "PbSO4", 0)
    qtbot.addWidget(dlg2)
    dlg2.mode.setCurrentIndex(0)
    dlg2.site_boxes["Pb"].setChecked(False)
    dlg2.sh_enable.setChecked(False)
    dlg2.size_uniaxial.setChecked(False)
    dlg2.apply()
    assert not ph.pawley and ph.structure.site("Pb").u_aniso is None
    assert pc.sh_order == 0 and pc.size_axial_um is None
    assert "h0:PbSO4:sh2_1" not in win.project.params.names()


def test_mem_dialog_runs_small_grid(qtbot, data_dir: Path, tmp_path: Path) -> None:
    win = _pbso4_window(qtbot, data_dir)
    # 粗い精密化(背景・スケール・格子・ゼロ点・幅)で MEM の入力を作る
    win.project.vary("h0:bkg:*", "h0:PbSO4:scale")
    win.project.refine(max_cycles=5)
    win.project.vary("PbSO4:cell:*", "h0:inst:zero", "h0:inst:w", "h0:inst:u", "h0:inst:v")
    win.project.refine(max_cycles=8)
    dlg = MEMDialog(win.project)
    qtbot.addWidget(dlg)
    assert dlg.grid[0].value() >= 8
    dlg.spacing.setValue(0.3)  # 粗い格子(約 30 × 18 × 24)で時間を節約(GUI の動作確認が目的)
    assert dlg.grid[0].value() >= 28
    with _run(dlg, qtbot):
        pass
    assert dlg.result is not None
    # 粗いモデルの F_obs と粗い格子なので C は 1 に届かない(正値性と矛盾する F が混じる)。
    # GUI の確認としては、密度が得られて最大が Pb の位置にあればよい
    assert np.all(np.isfinite(dlg.result.density)) and dlg.result.chi2 < 100.0
    st = win.project.phases["PbSO4"].structure
    top = dlg.result.maxima(1, 0.8)[0][0]
    eq, _ = st.spacegroup.equivalent_positions(st.site("Pb").position)
    d = min(float(st.lattice.distance(np.zeros(3), (top - e) - np.round(top - e))) for e in eq)
    assert d < 0.4
    assert dlg.view._density is not None
    assert "極大" in dlg.summary.text()
    # 書き出し(ファイルダイアログを通さずに直接)
    from kaisetsu.mem.io import write_ccp4

    p = write_ccp4(
        dlg.result.density, dlg.result.lattice, dlg.result.spacegroup, tmp_path / "m.ccp4"
    )
    assert p.exists()
    dv = DensityView()
    qtbot.addWidget(dv)
    dv.set_density(dlg.result.density)
    dv.slider.setValue(50)
    dv.axis.setCurrentIndex(0)


class _run:
    """MEMDialog.run() を呼び、ワーカーの終了まで待つコンテキスト。"""

    def __init__(self, dlg: MEMDialog, qtbot) -> None:  # type: ignore[no-untyped-def]
        self.dlg = dlg
        self.qtbot = qtbot

    def __enter__(self) -> None:
        self.dlg.run()
        assert self.dlg._thread is not None
        self.qtbot.waitUntil(
            lambda: self.dlg._thread is not None and not self.dlg._thread.isRunning(),
            timeout=300_000,
        )
        self.qtbot.waitUntil(lambda: self.dlg.run_btn.isEnabled(), timeout=5000)

    def __exit__(self, *exc: object) -> None:
        return None
