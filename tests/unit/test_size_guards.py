"""格子定数やピーク幅が発散したときに、メモリを使い尽くす前に止める守りのテスト。"""

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import Project, SampleGeometry
from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.errors import PatternSizeError
from kaisetsu.io import read_powder_data
from kaisetsu.pattern import CWInstrument, PhaseContribution, calculate_pattern
from kaisetsu.refine.lsq import FitResult, refine_least_squares
from kaisetsu.refine.parameters import Parameter, ParameterSet
from kaisetsu.refine.recipe import Recipe
from kaisetsu.reflections.hkl import generate_reflections
from kaisetsu.scattering import Radiation


def _si() -> Structure:
    return Structure(
        Lattice.cubic(5.43094),
        SpaceGroup.from_symbol("Fd-3m"),
        [AtomSite("Si", "Si", [0, 0, 0], u_iso=0.005)],
        name="Si",
    )


def test_generate_reflections_rejects_runaway_cell() -> None:
    """hkl の候補は格子定数の 3 乗で増える(a = 400 Å、d_min 0.5 Å で 4e9 個)。"""
    st = _si()
    assert len(generate_reflections(st.lattice, st.spacegroup, 0.8)) > 0
    with pytest.raises(PatternSizeError, match="hkl の候補"):
        generate_reflections(Lattice.cubic(400.0), st.spacegroup, 0.5)
    # 上限は引数で変えられる(a = 20 Å、d_min 1 Å で候補 41³ ≈ 6.9e4)
    assert len(generate_reflections(Lattice.cubic(20.0), st.spacegroup, 1.0)) > 0
    with pytest.raises(PatternSizeError):
        generate_reflections(Lattice.cubic(20.0), st.spacegroup, 1.0, max_candidates=1000)


def test_calculate_pattern_rejects_runaway_width() -> None:
    """ピーク窓の点数の合計が上限を超えたら PatternSizeError。"""
    x = np.arange(20.0, 120.0, 0.02)
    inst = CWInstrument(Radiation.characteristic("Cu"), w=0.01)
    ph = PhaseContribution(_si(), scale=1.0)
    res = calculate_pattern(x, [ph], inst)
    n = int(np.sum(res.peaks[0].window_stop - res.peaks[0].window_start))
    assert n > 0
    with pytest.raises(PatternSizeError, match="ピーク窓"):
        calculate_pattern(x, [ph], inst, max_window_points=n // 2)


def test_least_squares_treats_size_error_as_rejected_step() -> None:
    """試行点で PatternSizeError が出ても、その点を不可として精密化を続け、最小に着く。"""
    p = Parameter("a", -5.0, vary=True, min=-10.0, max=10.0)
    params = ParameterSet([p])
    raised = {"n": 0}

    def residual() -> np.ndarray:
        # p < 2 では勾配 0.1 の緩い坂、p ≥ 2 では勾配 2 の急な坂で p = 2.4 が最小。
        # 信頼領域の初期半径は |x0| = 5 なので最初の試行は p = 0 で止まり、受理されて半径が
        # 2 倍になった次の試行で Gauss-Newton 歩幅 10 が通り p ≈ 10(> 3)の「大きすぎ」になる
        if p.value > 3.0:
            raised["n"] += 1
            raise PatternSizeError("大きすぎ")
        r = 0.1 * p.value - 1.0 if p.value < 2.0 else -0.8 + 2.0 * (p.value - 2.0)
        return np.array([r, 0.0])

    res = refine_least_squares(params, residual, max_cycles=50)
    assert raised["n"] >= 1
    assert abs(p.value - 2.4) < 1e-3
    assert np.isfinite(res.chi2)


def _pbso4_project(data_dir: Path) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    inst = CWInstrument(Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01)
    proj = Project("guard")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def test_cell_parameters_bounded_relative_to_start(data_dir: Path) -> None:
    """格子定数の範囲は初期値の 0.5〜2 倍(角度は 10〜170°)。"""
    proj = _pbso4_project(data_dir)
    a = proj.params["PbSO4:cell:a"]
    assert a.min == pytest.approx(0.5 * a.value)
    assert a.max == pytest.approx(2.0 * a.value)
    alpha = proj.params["PbSO4:cell:alpha"]
    assert (alpha.min, alpha.max) == (10.0, 170.0)


def test_recipe_restores_values_when_pattern_explodes(data_dir: Path) -> None:
    """段階の途中で PatternSizeError が出たら、段階前の値に戻してから例外を伝える。"""
    proj = _pbso4_project(data_dir)
    a = proj.params["PbSO4:cell:a"]
    a0 = a.value

    def exploding(**kwargs: object) -> FitResult:
        a.value = 1.9 * a0
        raise PatternSizeError("試験")

    proj.refine = exploding  # type: ignore[method-assign]
    r = Recipe("t")
    r.add("cell", ["PbSO4:cell:*"], 2)
    with pytest.raises(PatternSizeError, match="段階 'cell'"):
        r.run(proj)
    assert a.value == a0
    assert any("発散" in line for line in proj.log)


def test_least_squares_propagates_size_error_at_start() -> None:
    """最初の点から大きすぎるなら、黙って続けずに呼び出し側へ知らせる。"""
    p = Parameter("a", 0.0, vary=True)

    def bad() -> np.ndarray:
        raise PatternSizeError("最初から大きすぎ")

    with pytest.raises(PatternSizeError):
        refine_least_squares(ParameterSet([p]), bad, max_cycles=5)
