"""「Le Bail で格子定数だけ」ダイアログ: 合成データ 2 本を一括処理して表と CSV に出す(オフスクリーン)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    GeometryKind,
    Lattice,
    PhaseContribution,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.crystal.cif import write_structure
from kaisetsu.gui.lattice_dialog import LatticeDialog
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.io.project import save_instrument_template

pytestmark = pytest.mark.slow


def _ceo2(a: float) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ce", "Ce", [0, 0, 0]), AtomSite("O", "O", [0.25, 0.25, 0.25])],
        name="CeO2",
    )


def test_lattice_dialog_runs_batch(qtbot, tmp_path: Path) -> None:
    inst = CWInstrument(
        Radiation.xray(0.5), u=0.0002, v=-0.0001, w=0.0008, x=0.003, polarization=0.98
    )
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=300.0)
    x = np.arange(4.0, 30.0, 0.01)
    bkg = ChebyshevBackground([100.0, -10.0], 4.0, 30.0)
    paths = []
    for i, a in enumerate((5.4100, 5.4130)):
        y = calculate_pattern(x, [PhaseContribution(_ceo2(a), scale=1.0)], inst, smp, bkg).y_calc
        peaks = y - bkg(x)
        y = bkg(x) + peaks * (1e4 / peaks.max())
        y = np.random.default_rng(i).poisson(y).astype(float)
        p = tmp_path / f"pt{i}.chi"
        p.write_text(
            "title\n2-Theta Angle (Degrees)\nIntensity\n"
            + f"{len(x):8d}\n"
            + "\n".join(f"{xi:.5f} {yi:.1f}" for xi, yi in zip(x, y, strict=True)),
            encoding="utf-8",
        )
        paths.append(str(p))
    cif = tmp_path / "ceo2.cif"
    write_structure(_ceo2(5.41), cif)
    template = save_instrument_template(tmp_path / "bl.kaisetsu-inst.json", inst, smp)

    win = MainWindow()
    qtbot.addWidget(win)
    dlg = LatticeDialog(win.project, win, main=win)
    qtbot.addWidget(dlg)
    dlg.set_paths(paths, [str(cif)])
    dlg.template_path.setText(str(template))
    dlg.rb_template.setChecked(True)
    assert dlg.options()["position"] == "none" and dlg.options()["widths"] == "sample"
    inst2, smp2 = dlg.instrument_and_sample()
    assert isinstance(inst2, CWInstrument) and smp2.kind is GeometryKind.DEBYE_SCHERRER
    assert dlg._worker is None
    dlg.run()
    assert dlg._worker is not None
    with qtbot.waitSignal(dlg._worker.finished, timeout=180_000):
        pass
    qtbot.waitUntil(lambda: dlg.run_btn.isEnabled(), timeout=5000)
    assert dlg.batch is not None and all(r.ok for r in dlg.batch.results), dlg.log.toPlainText()
    a_fit = [r.cells[0].values["a"] for r in dlg.batch.results]
    assert np.allclose(a_fit, [5.4100, 5.4130], atol=1e-4)
    assert dlg.table.rowCount() == 2 and dlg.table.item(0, 1).text() == "CeO2"
    assert "5.41" in dlg.table.item(1, 2).text()
    assert dlg.save_btn.isEnabled() and dlg.open_btn.isEnabled()
    assert dlg.last_project() is not None
    out = dlg.batch.to_csv(tmp_path / "out.csv")
    assert "CeO2:a" in out.read_text(encoding="utf-8").splitlines()[0]
    # 空のメインウィンドウなら確認なしで結果を開ける
    win.adopt_project(dlg.last_project(), source="テスト")
    assert win.project.phases["CeO2"].le_bail and win.param_model.rowCount() > 0
