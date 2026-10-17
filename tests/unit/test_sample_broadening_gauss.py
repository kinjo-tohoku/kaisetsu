"""CW の試料広がりのガウス成分に 8 ln2 が余分に掛からないこと(レビュー 2026-10-17 物理 #2)。

GSAS-II(GetSampleSigGam)は σ² に ((1−mx) Γ)² / (8 ln2) を足す。FWHM² = 8 ln2 σ² なので
ガウス FWHM² への加算は ((1−mx) Γ)²。以前は FWHM² にさらに 8 ln2 を掛けていて、
ローレンツ割合 < 1 でガウス幅が sqrt(8 ln2) = 2.35 倍になっていた。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.pattern.calculator import PhaseContribution, calculate_pattern
from kaisetsu.pattern.instrument import CWInstrument
from kaisetsu.scattering.radiation import Radiation

LAM = 1.5405929
STRUCTURE = Structure(
    Lattice.cubic(4.0), SpaceGroup.from_symbol("P m -3 m"), [AtomSite("Cu1", "Cu", [0, 0, 0])]
)
# 装置幅をほぼ 0 にして試料広がりだけを見る
INSTRUMENT = CWInstrument(Radiation.xray(LAM), u=0.0, v=0.0, w=1e-10, x=0.0, y=0.0, z=0.0)
X = np.arange(20.0, 30.0, 0.001)


def _scherrer_deg(two_theta: float, size_um: float) -> float:
    return math.degrees(LAM / (1.0e4 * size_um * math.cos(math.radians(two_theta / 2.0))))


def _strain_deg(two_theta: float, strain_e6: float) -> float:
    return math.degrees(strain_e6 * 1.0e-6 * math.tan(math.radians(two_theta / 2.0)))


def _with_instrument(gauss_fwhm: float) -> float:
    """装置のガウス幅(下限 _MIN_FWHM_G があるので 0 にはならない)と二乗和で合成した FWHM。"""
    inst = float(INSTRUMENT.fwhm_gauss(np.array([25.0]))[0])
    return math.sqrt(gauss_fwhm**2 + inst**2)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_size_broadening_splits_between_gauss_and_lorentz(fraction: float) -> None:
    pc = PhaseContribution(STRUCTURE, size_um=0.02, size_lorentz_fraction=fraction)
    table = calculate_pattern(X, [pc], INSTRUMENT).peaks[0]
    gamma = _scherrer_deg(float(table.two_theta[0]), 0.02)
    assert gamma == pytest.approx(0.45, abs=0.01)  # 20 nm で約 0.45°
    # 装置のローレンツ幅には下限 1e-5° がある
    assert table.fwhm_lorentz[0] == pytest.approx(fraction * gamma, abs=2e-5)
    assert table.fwhm_gauss[0] == pytest.approx(
        _with_instrument((1.0 - fraction) * gamma), abs=1e-6
    )


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_strain_broadening_splits_between_gauss_and_lorentz(fraction: float) -> None:
    pc = PhaseContribution(STRUCTURE, strain_e6=5000.0, strain_lorentz_fraction=fraction)
    table = calculate_pattern(X, [pc], INSTRUMENT).peaks[0]
    gamma = _strain_deg(float(table.two_theta[0]), 5000.0)
    # 装置のローレンツ幅には下限 1e-5° がある
    assert table.fwhm_lorentz[0] == pytest.approx(fraction * gamma, abs=2e-5)
    assert table.fwhm_gauss[0] == pytest.approx(
        _with_instrument((1.0 - fraction) * gamma), abs=1e-6
    )


def test_sample_broadening_returns_fwhm_squared_increment() -> None:
    """sample_broadening の戻り値は FWHM² への加算分そのもの(8 ln2 を含まない)。"""
    pc = PhaseContribution(
        STRUCTURE,
        size_um=0.05,
        strain_e6=3000.0,
        size_lorentz_fraction=0.3,
        strain_lorentz_fraction=0.6,
    )
    two_theta = np.array([25.0, 60.0, 110.0])
    add_g2, add_l = pc.sample_broadening(two_theta, LAM)
    g_size = np.array([_scherrer_deg(t, 0.05) for t in two_theta])
    g_strain = np.array([_strain_deg(t, 3000.0) for t in two_theta])
    assert np.allclose(add_l, 0.3 * g_size + 0.6 * g_strain)
    assert np.allclose(add_g2, (0.7 * g_size) ** 2 + (0.4 * g_strain) ** 2)


def test_numeric_fwhm_of_pure_gauss_size_peak() -> None:
    """ローレンツ割合 0 の実際のピーク幅が Scherrer 幅と一致する(表の値だけでなく形も)。"""
    pc = PhaseContribution(STRUCTURE, size_um=0.02, size_lorentz_fraction=0.0)
    res = calculate_pattern(X, [pc], INSTRUMENT)
    y = res.y_calc
    k = int(np.argmax(y))
    above = np.where(y >= 0.5 * y[k])[0]
    fwhm = float(X[above[-1]] - X[above[0]])
    gamma = _scherrer_deg(float(res.peaks[0].two_theta[0]), 0.02)
    assert fwhm == pytest.approx(gamma, abs=0.005)
