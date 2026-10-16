"""図のカーソル位置に近い反射の hkl・d・相対強度をラベルに出す(選択配向の確認用)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.gui.plot_widget import PatternPlot, ReflectionHit
from kaisetsu.io import read_powder_data

WIDTHS = {"u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02}


@pytest.fixture
def pbso4_plot(qtbot, data_dir: Path) -> tuple[PatternPlot, Project]:  # type: ignore[no-untyped-def]
    proj = Project("hover")
    proj.add_phase(Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif")), name="PbSO4")
    proj.add_histogram(
        read_powder_data(data_dir / "pbso4" / "PBSO4.XRA"),
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **WIDTHS),
        SampleGeometry(),
        n_background=3,
    )
    proj.build_parameters()
    res = proj.calculate(0)
    plot = PatternPlot()
    qtbot.addWidget(plot)
    plot.set_data(proj.histograms[0].data, 1.5406)
    tables = {"PbSO4": res.peaks[0]}
    plot.set_result(res, {"PbSO4": res.peaks[0].two_theta}, None, peak_tables=tables)
    return plot, proj


def test_reflections_near_returns_nearest_hkl(pbso4_plot: tuple[PatternPlot, Project]) -> None:
    plot, proj = pbso4_plot
    table = proj.calculate(0).peaks[0]
    # 最強反射(PbSO4 なら 2θ ≈ 26.7° の 2 1 0 付近)の位置で聞くと、その反射が先頭に来る
    j = int(np.argmax(table.intensity_total))
    x = float(table.two_theta[j])
    hits = plot.reflections_near(x, 0.02)
    assert hits and isinstance(hits[0], ReflectionHit)
    assert hits[0].hkl == tuple(int(v) for v in table.hkl[j])
    assert hits[0].rel_intensity == pytest.approx(100.0)
    assert hits[0].d == pytest.approx(float(table.d[j]))
    assert hits[0].phase == "PbSO4"
    # 近い順に並び、tol の外は含まない
    assert all(abs(h.position - x) <= max(0.02, 0.6 * h.fwhm) for h in hits)
    assert [abs(h.position - x) for h in hits] == sorted(abs(h.position - x) for h in hits)
    # 反射の無いところ(データ範囲の手前)では空
    assert plot.reflections_near(5.0, 0.02) == []
    assert plot.reflection_text(5.0, 0.02) == ""


def test_reflection_text_mentions_hkl_and_phase_only_when_several(
    pbso4_plot: tuple[PatternPlot, Project],
) -> None:
    plot, proj = pbso4_plot
    table = proj.calculate(0).peaks[0]
    j = int(np.argmax(table.intensity_total))
    x = float(table.two_theta[j])
    h, k, l = (int(v) for v in table.hkl[j])
    text = plot.reflection_text(x, 0.02)
    assert text.startswith("反射: ")
    assert f"({h} {k} {l})" in text and "I=100%" in text and "Å" in text
    assert "PbSO4" not in text  # 相が 1 つなら相名は付けない

    # 相が 2 つあれば相名を付ける(同じ表をもう 1 相として渡す)
    res = proj.calculate(0)
    plot.set_result(
        res,
        {"PbSO4": table.two_theta, "second": table.two_theta},
        None,
        peak_tables={"PbSO4": table, "second": table},
    )
    text2 = plot.reflection_text(x, 0.02)
    assert "PbSO4 (" in text2 and "second (" in text2


def test_mouse_label_shows_reflection_line(pbso4_plot: tuple[PatternPlot, Project]) -> None:
    """本物の経路: マウス位置 → ラベル 2 行目に「反射: (h k l) …」。"""
    from PySide6.QtCore import QPointF

    plot, proj = pbso4_plot
    plot.resize(900, 500)
    plot.show()
    vb = plot.plot.getPlotItem().vb
    vb.updateAutoRange()
    vb.disableAutoRange()
    plot.plot.grab()  # 一度描いて座標変換を確定させる
    table = proj.calculate(0).peaks[0]
    j = int(np.argmax(table.intensity_total))
    x = float(table.two_theta[j])
    h, k, l = (int(v) for v in table.hkl[j])
    rng = vb.viewRange()
    scene_pos = vb.mapViewToScene(QPointF(x, 0.5 * (rng[1][0] + rng[1][1])))
    plot._on_mouse((scene_pos,))
    text = plot._label.textItem.toPlainText()
    lines = text.splitlines()
    assert len(lines) == 2 and lines[0].startswith("2θ = ")
    assert lines[1].startswith("反射: ") and f"({h} {k} {l})" in lines[1]
    # 反射の無い 2θ = 5° では 1 行だけ
    plot._on_mouse((vb.mapViewToScene(QPointF(5.0, 0.5 * (rng[1][0] + rng[1][1]))),))
    assert len(plot._label.textItem.toPlainText().splitlines()) == 1


def test_clear_calc_drops_reflections(pbso4_plot: tuple[PatternPlot, Project]) -> None:
    plot, proj = pbso4_plot
    table = proj.calculate(0).peaks[0]
    x = float(table.two_theta[int(np.argmax(table.intensity_total))])
    assert plot.reflections_near(x, 0.02)
    plot.clear_calc()
    assert plot.reflections_near(x, 0.02) == []
