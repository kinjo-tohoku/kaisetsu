"""逐次精密化ダイアログと英語 UI の切り替え(オフスクリーン)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    Lattice,
    PhaseContribution,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.gui import i18n
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.sequence_dialog import SequenceDialog

pytestmark = pytest.mark.slow


def _series(tmp_path: Path) -> tuple[list[Path], CWInstrument]:
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7)
    x = np.arange(20.0, 70.0, 0.04)
    bkg = ChebyshevBackground([40.0, -3.0], 20.0, 70.0)
    rng = np.random.default_rng(2)
    paths = []
    for i, a in enumerate((4.000, 4.006, 4.012)):
        st = Structure(Lattice.cubic(a), SpaceGroup.from_symbol("P m -3 m"),
                       [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)])  # fmt: skip
        y = calculate_pattern(x, [PhaseContribution(st, scale=30.0)], inst, SampleGeometry(), bkg)
        p = tmp_path / f"s{i}.xy"
        np.savetxt(p, np.column_stack([x, rng.poisson(y.y_calc).astype(float)]))
        paths.append(p)
    return paths, inst


def test_sequence_dialog_runs(qtbot, tmp_path: Path) -> None:
    paths, inst = _series(tmp_path)
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_data_file(str(paths[0]), inst, SampleGeometry(), n_background=2)
    st = Structure(Lattice.cubic(4.0), SpaceGroup.from_symbol("P m -3 m"),
                   [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)], name="Ni")  # fmt: skip
    win.project.add_phase(st, name="Ni")
    win.project.phase_contribution(0, "Ni").scale = 30.0
    win.project.build_parameters()
    win.project.vary("h0:bkg:*", "h0:Ni:scale", "Ni:cell:a")
    dlg = SequenceDialog(win.project)
    qtbot.addWidget(dlg)
    dlg.set_paths([str(p) for p in paths])
    dlg.run()
    qtbot.waitUntil(
        lambda: dlg._thread is not None and not dlg._thread.isRunning(), timeout=120_000
    )
    qtbot.waitUntil(lambda: dlg.run_btn.isEnabled(), timeout=5000)
    assert dlg.result is not None and len(dlg.result.rows) == 3
    a = dlg.result.column("Ni:cell:a")
    assert np.allclose(a, [4.000, 4.006, 4.012], atol=1e-3)
    assert dlg.table.rowCount() == 3 and dlg.save_btn.isEnabled()
    out = tmp_path / "seq.csv"
    dlg.result.to_csv(out)
    assert out.exists()
    assert win.act_sequence.isEnabled()


def test_english_ui(qtbot) -> None:
    i18n.set_language("en")
    try:
        assert i18n.tr("ファイル(&F)") == "&File"
        win = MainWindow()
        qtbot.addWidget(win)
        titles = [a.text() for a in win.menuBar().actions()]
        assert "&File" in titles and "&Data" in titles
        assert win.act_mem.text().startswith("&MEM")
        assert win.structure_dock.windowTitle() == "Structure"
    finally:
        i18n.set_language("ja")
    assert i18n.tr("ファイル(&F)") == "ファイル(&F)"
    assert i18n.tr("未翻訳の文字列") == "未翻訳の文字列"
