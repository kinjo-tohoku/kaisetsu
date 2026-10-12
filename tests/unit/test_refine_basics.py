import numpy as np
import pytest

from kaisetsu.crystal import CrystalSystem, SpaceGroup
from kaisetsu.refine.constraints import (
    adp_constraint,
    lattice_constraint,
    position_constraint,
)
from kaisetsu.refine.parameters import Parameter, ParameterSet
from kaisetsu.refine.rfactors import (
    bragg_rfactor,
    partition_intensities,
    powder_rfactors,
    structure_rfactor,
)


def test_parameter_set_vary_select_and_vector() -> None:
    store: dict[str, float] = {}
    ps = ParameterSet()
    for name, v in (
        ("h0:inst:zero", 0.0),
        ("h0:bkg:0", 10.0),
        ("h0:bkg:1", 1.0),
        ("P:cell:a", 5.0),
    ):
        ps.add(Parameter(name, v, setter=lambda val, n=name: store.__setitem__(n, val)))
    assert ps.vary("h0:bkg:*") == ["h0:bkg:0", "h0:bkg:1"]
    assert ps.free_names() == ["h0:bkg:0", "h0:bkg:1"]
    ps.vary("P:cell:a")
    assert np.allclose(ps.vector(), [10.0, 1.0, 5.0])
    ps.set_vector([11.0, 2.0, 5.1])
    assert store == {"h0:bkg:0": 11.0, "h0:bkg:1": 2.0, "P:cell:a": 5.1, "h0:inst:zero": 0.0}
    ps.fix("h0:bkg:*")
    assert ps.free_names() == ["P:cell:a"]
    with pytest.raises(KeyError):
        ps.vary("nothing:*")
    with pytest.raises(ValueError):
        ps.add(Parameter("P:cell:a", 1.0))


def test_parameter_ties_and_esd() -> None:
    ps = ParameterSet([Parameter("a", 4.0), Parameter("b", 4.0), Parameter("c", 1.0)])
    ps.tie("b", "a")
    ps.tie("c", ["a"], [0.5], offset=0.25)
    ps.vary("a")
    assert ps.free_names() == ["a"]
    ps.vary("b")  # 従属パラメータは vary できない
    assert ps.free_names() == ["a"]
    ps.set_vector([5.0])
    assert ps["b"].value == 5.0
    assert ps["c"].value == pytest.approx(2.75)
    ps.set_esd([0.1])
    assert ps["b"].esd == pytest.approx(0.1)
    assert ps["c"].esd == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ps.tie("a", "b")  # 連鎖禁止


@pytest.mark.parametrize(
    ("symbol", "xyz", "n_free", "deps"),
    [
        ("Pnma", (0.188, 0.25, 0.167), 2, {}),  # 4c (x, 1/4, z): y 固定
        ("Pnma", (0.085, 0.026, 0.806), 3, {}),  # 一般位置
        ("Fd-3m", (0.1, 0.1, 0.1), 1, {1: (0, 1.0), 2: (0, 1.0)}),  # 32e (x,x,x)
        ("Fd-3m", (0.0, 0.0, 0.0), 0, {}),  # 8a: 全固定
        ("P63/mmc", (1 / 3, 2 / 3, 0.25), 0, {}),  # 2c
        ("P6/mmm", (0.3, 0.6, 0.0), 1, {1: (0, 2.0)}),  # (x, 2x, 0)
        ("P4/mmm", (0.3, 0.3, 0.5), 1, {1: (0, 1.0)}),  # (x, x, 1/2)
        ("I4/mmm", (0.0, 0.0, 0.35), 1, {}),  # 4e (0, 0, z)
    ],
)
def test_position_constraints(
    symbol: str, xyz: tuple[float, float, float], n_free: int, deps: dict
) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    c = position_constraint(sg, xyz)
    assert c.n_free == n_free
    assert np.allclose(c.symmetrized, xyz, atol=1e-6)
    for j, (piv, coeff) in deps.items():
        rel = c.dependent[j]
        assert rel.pivots == (piv,)
        assert rel.coeffs[0] == pytest.approx(coeff)
    # 自由 + 従属 + 固定 = 3
    assert len(c.free) + len(c.dependent) + len(c.fixed) == 3


def test_position_constraint_symmetrizes() -> None:
    sg = SpaceGroup.from_symbol("Fd-3m")
    c = position_constraint(sg, (0.1001, 0.0999, 0.1000))  # 32e から僅かにずれた位置
    assert c.n_free == 1
    assert np.allclose(c.symmetrized, [0.1, 0.1, 0.1], atol=1e-4)


def test_adp_constraints() -> None:
    # 立方晶の 8a: U11 = U22 = U33, 非対角 0
    c = adp_constraint(SpaceGroup.from_symbol("Fd-3m"), (0, 0, 0), [0.01, 0.01, 0.01, 0, 0, 0])
    assert c.free == (0,)
    assert set(c.dependent) == {1, 2}
    assert set(c.fixed) == {3, 4, 5}
    # 六方晶 2c (P63/mmc): U11 = U22 = 2 U12, U33 自由, U13 = U23 = 0
    c = adp_constraint(SpaceGroup.from_symbol("P63/mmc"), (1 / 3, 2 / 3, 0.25))
    assert sorted(c.free) == [0, 2]
    assert c.dependent[1].pivots == (0,) and c.dependent[1].coeffs[0] == pytest.approx(1.0)
    assert c.dependent[3].pivots == (0,) and c.dependent[3].coeffs[0] == pytest.approx(0.5)
    assert set(c.fixed) == {4, 5}
    # Pnma 4c (m ⊥ b): U12 = U23 = 0、他は自由
    c = adp_constraint(SpaceGroup.from_symbol("Pnma"), (0.188, 0.25, 0.167))
    assert sorted(c.free) == [0, 1, 2, 4]
    assert set(c.fixed) == {3, 5}
    # 一般位置: 全自由
    c = adp_constraint(SpaceGroup.from_symbol("P-1"), (0.1, 0.2, 0.3))
    assert c.n_free == 6


def test_lattice_constraints() -> None:
    c = lattice_constraint(CrystalSystem.CUBIC)
    assert c.free == (0,) and c.ties == {1: 0, 2: 0}
    c = lattice_constraint(CrystalSystem.HEXAGONAL)
    assert c.free == (0, 2) and c.ties == {1: 0}
    c = lattice_constraint(CrystalSystem.MONOCLINIC, unique_axis="b")
    assert c.free == (0, 1, 2, 4)
    c = lattice_constraint(CrystalSystem.TRIGONAL, rhombohedral=True)
    assert c.free == (0, 3)
    assert lattice_constraint(CrystalSystem.TRICLINIC).free == (0, 1, 2, 3, 4, 5)


def test_rfactors() -> None:
    rng = np.random.default_rng(0)
    yc = 100 + 1000 * np.exp(-0.5 * ((np.arange(200) - 100) / 5) ** 2)
    yo = rng.poisson(yc).astype(float)
    w = 1.0 / np.maximum(yo, 1)
    r = powder_rfactors(yo, yc, w, n_params=5, y_bkg=np.full_like(yc, 100.0))
    assert 0 < r.rp < 0.2 and 0 < r.rwp < 0.2
    assert r.chi2 == pytest.approx(1.0, abs=0.5)  # 正しいモデル + Poisson 誤差
    assert r.rp_bkg > r.rp and r.rwp_bkg > r.rwp
    assert r.n_points == 200 and r.n_params == 5
    assert "Rwp" in str(r)
    assert bragg_rfactor([10, 20], [10, 20]) == 0.0
    assert bragg_rfactor([10, 20], [11, 18]) == pytest.approx(3 / 30)
    assert structure_rfactor([100, 400], [100, 400]) == 0.0
    assert structure_rfactor([100, 400], [121, 361]) == pytest.approx(2 / 30)


def test_partition_intensities_recovers_scaled_peaks() -> None:
    x = np.arange(0, 10, 0.01)
    bkg = np.full_like(x, 5.0)
    prof1 = np.exp(-0.5 * ((x - 4.0) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
    prof2 = np.exp(-0.5 * ((x - 4.15) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))  # 重なるピーク
    i_calc = np.array([10.0, 20.0])
    c1, c2 = i_calc[0] * prof1 * 0.01, i_calc[1] * prof2 * 0.01
    y_calc = bkg + c1 + c2
    # 観測が計算の 1.5 倍なら分配強度も 1.5 倍になる
    y_obs = bkg + 1.5 * (c1 + c2)
    i_obs = partition_intensities(y_obs, y_calc, bkg, [(0, len(x)), (0, len(x))], [c1, c2], i_calc)
    assert np.allclose(i_obs, 1.5 * i_calc)


def test_partition_ignores_background_misfit_in_far_tails() -> None:
    """窓の裾に背景の誤差があっても、分配強度は中心部で決まる(Le Bail の強度の膨らみ防止)。"""
    x = np.linspace(-2.0, 2.0, 2001)
    fwhm = 0.02  # 放射光並みに鋭いピーク、窓 4°(= 200 FWHM)
    prof = 1.0 / (1.0 + (2.0 * x / fwhm) ** 2)  # ローレンツ
    prof /= prof.sum()
    i_true = 1.0e5  # ピーク高さ ~6400
    bkg = np.full_like(x, 100.0)
    y_obs = bkg + i_true * prof + 20.0  # 背景が一様に 20 カウント(高さの 0.3%)ずれている
    c = 0.5 * i_true * prof  # 計算強度は真の半分
    y_calc = bkg + c
    i_calc = np.array([0.5 * i_true])
    i_core = partition_intensities(y_obs, y_calc, bkg, [(0, len(x))], [c], i_calc)
    i_full = partition_intensities(
        y_obs, y_calc, bkg, [(0, len(x))], [c], i_calc, core_fraction=0.0
    )
    # 窓全体では 20 カウント × 2001 点 = 4e4 がそのまま乗る(+40%)
    assert i_full[0] > 1.3 * i_true
    assert abs(i_core[0] - i_true) < 0.05 * i_true
