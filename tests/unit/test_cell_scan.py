"""格子定数の初期値の走査(`kaisetsu.refine.cell_scan`)と、それを使う標準手順・逐次精密化。

合成 LaB6(放射光級の鋭い幅、最大 2 万カウント)で、
(1) 初期値が +0.6% ずれていると走査無しの「格子定数とピーク位置」は谷を見つけられないが、
    走査は正しい倍率を採ること、
(2) 走査付きの標準手順は ±3% のずれからでも真値に戻ること(走査無しでは発散する)、
(3) 初期値が正しければ何も変えないこと、
(4) 逐次精密化で格子定数が 1.5% 跳んだデータも、振り直しで成功すること
を確かめる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.refine.cell_scan import scan_all_cells, scan_cell, scan_step
from kaisetsu.refine.recipe import standard_recipe
from kaisetsu.refine.sequential import refine_sequence

ROOT = Path(__file__).resolve().parents[2]
LAB6_CIF = ROOT / "data" / "public" / "lab6" / "LaB6_COD1000055.cif"
SHARP = {"u": 0.001, "v": -0.001, "w": 0.0015, "x": 0.005, "y": 0.005}
A_TRUE = Structure.from_cif(str(LAB6_CIF)).lattice.a


def _synthetic(a_factor: float = 1.0, seed: int = 3) -> PowderData:
    st = Structure.from_cif(str(LAB6_CIF)).copy()
    sim = Project("sim")
    sim.add_phase(st, name="L")
    x = np.arange(20.0, 120.0, 0.01)
    sim.add_histogram(
        PowderData(x, np.ones_like(x)),
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **SHARP),
        SampleGeometry(),
        n_background=2,
    )
    sim.build_parameters()
    p = sim.params["L:cell:a"]
    p.value = A_TRUE * a_factor
    sim.params.evaluate_ties()
    sim.params.apply_all()
    y_peaks = sim.calculate(0).y_bragg
    y_true = 20000.0 * y_peaks / y_peaks.max() + 30.0 + 0.3 * x
    y = np.random.default_rng(seed).poisson(y_true).astype(np.float64)
    return PowderData(x, y)


def _project(data: PowderData, a_factor: float) -> Project:
    proj = Project("fit")
    proj.add_phase(Structure.from_cif(str(LAB6_CIF)).copy(), name="L")
    proj.add_histogram(
        data,
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **SHARP),
        SampleGeometry(),
        n_background=6,
    )
    proj.build_parameters()
    p = proj.params["L:cell:a"]
    p.value = p.value * a_factor
    proj.params.evaluate_ties()
    proj.params.apply_all()
    return proj


def _recipe(**kw: bool) -> object:
    return standard_recipe(
        "*", "h0", coordinates=False, adp=False, try_no_kalpha2=False, humps=False, **kw
    )


def test_scan_finds_the_basin_from_0_6_percent_off() -> None:
    proj = _project(_synthetic(), 1.006)
    step, half = scan_step(proj)
    assert 2e-5 <= step <= 5e-4 and 2 * half + 1 <= 801
    r = scan_cell(proj, "L")
    assert r.adopted and r.names == ["L:cell:a"]
    assert abs(r.factors["L:cell:a"] - 1 / 1.006) < 2 * step
    assert r.rwp_after < 0.3 * r.rwp_before
    assert abs(proj.params["L:cell:a"].value - A_TRUE) < 1e-3
    assert "採りました" in r.message


def test_scan_keeps_a_correct_start() -> None:
    proj = _project(_synthetic(), 1.0)
    r = scan_cell(proj, "L")
    assert not r.adopted
    assert "初期値のままが最良" in r.message
    assert abs(proj.params["L:cell:a"].value - A_TRUE) < 1e-9


@pytest.mark.parametrize("a_factor", [1.006, 1.03, 0.97])
def test_recipe_with_scan_recovers_from_wrong_start(a_factor: float) -> None:
    proj = _project(_synthetic(), a_factor)
    stage_a: dict[str, float] = {}

    def _progress(step: object, _res: object) -> None:
        stage_a[getattr(step, "name", "")] = proj.params["L:cell:a"].value

    _recipe().run(proj, progress=_progress)  # type: ignore[attr-defined]
    assert any("格子定数の初期値を振る" in line and "採りました" in line for line in proj.log)
    # 走査のおかげで「格子定数とピーク位置」の段階で既に真値に着く
    assert abs(stage_a["格子定数とピーク位置"] - A_TRUE) < 2e-4
    assert abs(proj.params["L:cell:a"].value - A_TRUE) < 1e-4
    assert proj.rfactors()["total"].rwp < 0.12


def test_recipe_without_scan_is_trapped_at_0_6_percent() -> None:
    """走査無しでは「格子定数とピーク位置」が谷を見つけられない(走査を入れた理由)。"""
    proj = _project(_synthetic(), 1.006)
    stage_a: dict[str, float] = {}

    def _progress(step: object, _res: object) -> None:
        stage_a[getattr(step, "name", "")] = proj.params["L:cell:a"].value

    _recipe(cell_scan=False).run(proj, progress=_progress)  # type: ignore[attr-defined]
    assert abs(stage_a["格子定数とピーク位置"] - A_TRUE) > 1e-2


def test_sequence_recovers_after_lattice_jump(tmp_path: Path) -> None:
    """3 つ目で格子定数が 1.5% 跳ぶ系列。振り直しで 3 つ目も成功し、以後も続く。"""
    paths: list[Path] = []
    for i, f in enumerate((1.0, 1.002, 1.017, 1.019)):
        d = _synthetic(f, seed=10 + i)
        p = tmp_path / f"T{i}.xy"
        np.savetxt(p, np.column_stack([d.x, d.y]), fmt="%.4f %.1f")
        paths.append(p)
    proj = _project(_synthetic(), 1.0)
    proj.vary("h0:bkg:*", "h0:L:scale", "L:cell:a", "h0:inst:zero")
    proj.refine(max_cycles=10)
    res = refine_sequence(proj, paths, params=["L:cell:a"], max_cycles=10)
    assert [r.ok for r in res.rows] == [True, True, True, True]
    a = res.column("L:cell:a")
    assert np.allclose(a, A_TRUE * np.array([1.0, 1.002, 1.017, 1.019]), atol=2e-4)
    assert "振り直して再精密化" in res.rows[2].message


def test_sequence_without_scan_fails_after_jump(tmp_path: Path) -> None:
    paths: list[Path] = []
    for i, f in enumerate((1.0, 1.017)):
        d = _synthetic(f, seed=20 + i)
        p = tmp_path / f"T{i}.xy"
        np.savetxt(p, np.column_stack([d.x, d.y]), fmt="%.4f %.1f")
        paths.append(p)
    proj = _project(_synthetic(), 1.0)
    proj.vary("h0:bkg:*", "h0:L:scale", "L:cell:a", "h0:inst:zero")
    proj.refine(max_cycles=10)
    res = refine_sequence(proj, paths, params=["L:cell:a"], max_cycles=10, cell_scan=False)
    a = res.column("L:cell:a")
    assert not res.rows[1].ok or abs(a[1] - A_TRUE * 1.017) > 1e-3


def test_scan_all_cells_handles_le_bail_phase() -> None:
    proj = _project(_synthetic(), 1.004)
    proj.start_le_bail("L")
    proj.build_parameters()
    results = scan_all_cells(proj)
    assert len(results) == 1 and results[0].adopted
    assert abs(proj.params["L:cell:a"].value - A_TRUE) < 1e-3
