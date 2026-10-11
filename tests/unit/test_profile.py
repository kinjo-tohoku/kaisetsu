import math

import numpy as np
import pytest

from kaisetsu.profile import ChebyshevBackground, fcj_pseudo_voigt, pseudo_voigt, tch_fwhm_eta
from kaisetsu.profile.corrections import (
    bragg_brentano_shift,
    cylinder_absorption,
    debye_scherrer_shift,
    lorentz_factor,
    march_dollase,
    polarization_factor,
    polarization_from_monochromator,
    surface_roughness_suortti,
)
from kaisetsu.profile.fcj import fcj_limits, quadrature_order
from kaisetsu.profile.pseudo_voigt import pseudo_voigt_fwhm


def _fwhm_numeric(x: np.ndarray, y: np.ndarray) -> float:
    half = y.max() / 2.0
    above = np.where(y >= half)[0]
    return float(x[above[-1]] - x[above[0]])


def test_tch_limits() -> None:
    fwhm, eta = tch_fwhm_eta(0.1, 0.0)
    assert fwhm == pytest.approx(0.1)
    assert eta == pytest.approx(0.0)
    fwhm, eta = tch_fwhm_eta(0.0, 0.1)
    assert fwhm == pytest.approx(0.1)
    assert eta == pytest.approx(1.36603 - 0.47719 + 0.11116)
    fwhm, eta = tch_fwhm_eta(0.1, 0.1)
    assert 0.1 < fwhm < 0.2
    assert 0.0 < eta < 1.0


def test_pseudo_voigt_normalization_and_width() -> None:
    x = np.linspace(-20, 20, 400001)
    dx = x[1] - x[0]
    for fg, fl in ((0.1, 0.0), (0.0, 0.1), (0.08, 0.05), (0.3, 0.2)):
        y = pseudo_voigt(x, fg, fl)
        # ローレンツ成分の裾 ±20 の外側は η Γ/(π·20) 程度(最大で ~4e-3)
        assert np.sum(y) * dx == pytest.approx(1.0, abs=6e-3)
        assert _fwhm_numeric(x, y) == pytest.approx(pseudo_voigt_fwhm(fg, fl), abs=3 * dx)
        assert np.allclose(y, y[::-1])  # 対称


def test_gaussian_limit_matches_formula() -> None:
    x = np.linspace(-1, 1, 2001)
    fwhm = 0.2
    sig = fwhm / (2 * math.sqrt(2 * math.log(2)))
    expected = np.exp(-0.5 * x**2 / sig**2) / (sig * math.sqrt(2 * math.pi))
    assert np.allclose(pseudo_voigt(x, fwhm, 0.0), expected)


def test_fcj_reduces_to_pv_without_asymmetry() -> None:
    x = np.linspace(19, 21, 801)
    assert np.allclose(
        fcj_pseudo_voigt(x, 20.0, 0.1, 0.05, 0.0, 0.0), pseudo_voigt(x - 20.0, 0.1, 0.05)
    )


def test_fcj_limits_and_order() -> None:
    lo, infl = fcj_limits(20.0, 0.02, 0.02)
    assert lo < 20.0
    assert infl == pytest.approx(20.0)  # S/L = H/L なら変曲点は 2θ
    lo2, infl2 = fcj_limits(20.0, 0.03, 0.01)
    assert lo2 < infl2 < 20.0
    hi, _ = fcj_limits(120.0, 0.02, 0.02)
    assert hi > 120.0  # 90° 以上では高角側に裾
    assert quadrature_order(20.0, lo, 0.05) >= 2
    assert quadrature_order(20.0, lo, 0.0) >= 2


@pytest.mark.parametrize("pos", [15.0, 40.0, 70.0, 130.0])
def test_fcj_area_and_tail_direction(pos: float) -> None:
    x = np.linspace(pos - 8, pos + 8, 16001)
    dx = x[1] - x[0]
    y = fcj_pseudo_voigt(x, pos, 0.08, 0.04, 0.03, 0.03)
    assert np.sum(y) * dx == pytest.approx(1.0, abs=5e-3)
    # 重心が 90° 未満では低角側、90° 以上では高角側にずれる
    centroid = float(np.sum(x * y) / np.sum(y))
    if pos < 90:
        assert centroid < pos
    else:
        assert centroid > pos


def test_chebyshev_background() -> None:
    x = np.linspace(10, 150, 1001)
    const = ChebyshevBackground([3.0], 10, 150)
    assert np.allclose(const(x), 3.0)
    lin = ChebyshevBackground([1.0, 2.0], 10, 150)
    assert lin(x)[0] == pytest.approx(1.0 - 2.0)
    assert lin(x)[-1] == pytest.approx(1.0 + 2.0)
    b = lin.basis(x)
    assert b.shape == (1001, 2)
    assert np.allclose(b @ lin.coefficients, lin(x))
    with pytest.raises(ValueError):
        ChebyshevBackground([1.0], 10, 5)


def test_lorentz_polarization() -> None:
    tt = np.array([20.0, 90.0, 150.0])
    th = np.radians(tt / 2)
    assert np.allclose(lorentz_factor(tt), 1 / (2 * np.sin(th) ** 2 * np.cos(th)))
    assert np.allclose(polarization_factor(tt, 0.5), 0.5 * (1 + np.cos(np.radians(tt)) ** 2))
    assert np.allclose(polarization_factor(tt, 1.0), 1.0)
    p = polarization_from_monochromator(26.6)
    k = math.cos(math.radians(26.6)) ** 2
    assert np.allclose(polarization_factor(tt, p), (1 + k * np.cos(np.radians(tt)) ** 2) / (1 + k))


def test_march_dollase() -> None:
    from kaisetsu.crystal import Lattice, SpaceGroup

    lat = Lattice.hexagonal(3.0, 5.0)
    sg = SpaceGroup.from_symbol("P6/mmm")
    hkl = [[0, 0, 1], [1, 0, 0], [1, 0, 1]]
    assert np.allclose(march_dollase(hkl, lat, sg, (0, 0, 1), 1.0), 1.0)
    po = march_dollase(hkl, lat, sg, (0, 0, 1), 0.8)
    # r < 1: 配向軸に平行な反射 (001) が強まり、垂直な (100) が弱まる
    assert po[0] == pytest.approx(0.8 ** (-3))
    assert po[1] == pytest.approx(0.8**1.5)
    assert po[1] < 1.0 < po[0]
    # 2 乗平均則: Σ P_h は等価反射で平均しているので (101) は中間
    assert po[1] < po[2] < po[0]


def test_absorption_and_roughness() -> None:
    tt = np.linspace(5, 170, 50)
    assert np.allclose(cylinder_absorption(0.0, tt), 1.0)
    a1 = cylinder_absorption(1.0, tt)
    assert np.all((a1 > 0) & (a1 < 1))
    assert a1[-1] > a1[0]  # 高角ほど透過率が高い
    a4 = cylinder_absorption(4.0, tt)
    assert np.all(a4 < a1)
    # μR = 3 の前後で近似式が連続
    assert cylinder_absorption(2.999, 60.0) == pytest.approx(
        cylinder_absorption(3.001, 60.0), rel=0.05
    )
    assert np.allclose(surface_roughness_suortti(0.0, 0.0, tt), 1.0)
    sr = surface_roughness_suortti(0.9, 0.1, tt)
    assert np.all(sr <= 1.0 + 1e-12)


def test_position_shifts() -> None:
    tt = np.array([30.0, 90.0])
    # 変位 0.1 mm, R = 200 mm → -(2 s/R) cosθ
    s = bragg_brentano_shift(tt, 0.1, 200.0)
    assert np.allclose(s, -np.degrees(2 * 0.1 / 200 * np.cos(np.radians(tt / 2))))
    assert s[0] < 0
    # 透過
    s2 = bragg_brentano_shift(tt, 0.0, 200.0, mu_eff_inv_mm=0.05)
    assert np.allclose(s2, -np.degrees(np.sin(np.radians(tt)) / (2 * 0.05 * 200.0)))
    d = debye_scherrer_shift(tt, 0.1, 0.0, 200.0)
    assert np.allclose(d, -np.degrees(0.1 * np.cos(np.radians(tt)) / 200.0))
