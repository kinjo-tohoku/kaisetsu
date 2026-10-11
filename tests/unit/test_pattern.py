from pathlib import Path

import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.io import read_powder_data
from kaisetsu.pattern import (
    CWInstrument,
    PhaseContribution,
    PowderData,
    SampleGeometry,
    calculate_pattern,
)
from kaisetsu.pattern.instrument import EIGHT_LN2, GeometryKind
from kaisetsu.profile import ChebyshevBackground
from kaisetsu.scattering import Radiation


def _si() -> Structure:
    return Structure(
        Lattice.cubic(5.43094),
        SpaceGroup.from_symbol("Fd-3m"),
        [AtomSite("Si", "Si", [0, 0, 0], u_iso=0.005)],
        name="Si",
    )


def test_instrument_from_gsas2_roundtrip() -> None:
    inst = CWInstrument.from_gsas2(
        {
            "Type": "PXC",
            "Lam1": 1.5405,
            "Lam2": 1.5443,
            "I(L2)/I(L1)": 0.5,
            "Zero": 0.01,
            "Polariz.": 0.7,
            "U": 2.0,
            "V": -2.0,
            "W": 5.0,
            "X": 1.0,
            "Y": 3.0,
            "Z": 0.0,
            "SH/L": 0.05,
        }
    )
    assert inst.radiation.wavelengths == ((1.5405, 1.0), (1.5443, 0.5))
    assert inst.u == pytest.approx(EIGHT_LN2 * 2.0 / 1e4)
    assert inst.w == pytest.approx(EIGHT_LN2 * 5.0 / 1e4)
    assert inst.y == pytest.approx(0.01)  # GSAS X (1/cosθ) → y
    assert inst.x == pytest.approx(0.03)  # GSAS Y (tanθ) → x
    assert inst.s_l == inst.h_l == pytest.approx(0.025)
    back = inst.to_gsas2_widths()
    assert back["U"] == pytest.approx(2.0)
    assert back["X"] == pytest.approx(1.0)
    assert back["Y"] == pytest.approx(3.0)
    assert back["SH/L"] == pytest.approx(0.05)
    # 幅: 2θ = 0 で Γ_G^2 = W
    assert inst.fwhm_gauss(1e-6) ** 2 == pytest.approx(inst.w, rel=1e-6)
    neut = CWInstrument.from_gsas2(
        {"Type": "PNC", "Lam": 1.909, "Zero": -0.001, "U": 354.0, "V": -760.0, "W": 651.0}
    )
    assert neut.radiation.is_neutron
    assert neut.radiation.wavelength == 1.909


def test_sample_from_gsas2() -> None:
    s = SampleGeometry.from_gsas2(
        {
            "Type": "Bragg-Brentano",
            "Gonio. radius": 200.0,
            "Shift": [100.0, False],
            "Transparency": [0.0, False],
        }
    )
    assert s.kind is GeometryKind.BRAGG_BRENTANO
    assert s.displacement_mm == pytest.approx(0.1)
    d = SampleGeometry.from_gsas2(
        {
            "Type": "Debye-Scherrer",
            "Gonio. radius": 200.0,
            "DisplaceX": [50.0, False],
            "Absorption": [1.5, False],
        }
    )
    assert d.kind is GeometryKind.DEBYE_SCHERRER
    assert d.displace_x_mm == pytest.approx(0.05)
    assert d.mu_r == 1.5
    # GSAS-II の Shift 100 μm と同じシフト量 -(0.36/πR)·100·cosθ
    tt = np.array([40.0])
    expected = -(4 * 0.09 / (np.pi * 200.0)) * 100.0 * np.cos(np.radians(20.0))
    assert s.position_shift(tt)[0] == pytest.approx(expected)


def test_powder_data_basics() -> None:
    d = PowderData(np.array([10.0, 10.1, 10.2]), np.array([100.0, 0.0, 25.0]))
    assert d.weight is not None
    assert np.allclose(d.weight, [0.01, 0.0, 0.04])
    assert np.isinf(d.sigma[1])
    assert d.step == pytest.approx(0.1)
    c = d.cropped(10.05, None)
    assert len(c) == 2
    with pytest.raises(ValueError):
        PowderData(np.array([10.0, 9.0]), np.array([1.0, 1.0]))


def test_read_gsas_std_and_xy(data_dir: Path, tmp_path: Path) -> None:
    d = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA")
    assert d.metadata["subformat"] == "STD"
    assert d.x[0] == pytest.approx(10.0)
    assert d.x[1] - d.x[0] == pytest.approx(0.025)
    assert d.y[0] == 179.0
    assert d.weight is not None and d.weight[0] == pytest.approx(1 / 179.0)
    n = read_powder_data(data_dir / "pbso4" / "PBSO4.CWN")
    assert n.x[0] == pytest.approx(10.0)
    assert n.x[1] - n.x[0] == pytest.approx(0.05)
    assert n.y[0] == 220.0
    # xy 形式
    p = tmp_path / "a.xye"
    p.write_text("# comment\n2theta I sigma\n10.0 100 10\n10.1 200 14.1\n10.2 50 7.1\n")
    xy = read_powder_data(p)
    assert len(xy) == 3
    assert xy.weight is not None and xy.weight[0] == pytest.approx(0.01)
    p2 = tmp_path / "b.xy"
    p2.write_text("10.0,100\n10.1,0\n10.2,50\n")
    xy2 = read_powder_data(p2)
    assert xy2.weight is not None and xy2.weight[1] == 0.0


def test_calculate_pattern_si_peaks() -> None:
    inst = CWInstrument(Radiation.characteristic("Cu", kalpha2=False), w=0.01, s_l=0.0, h_l=0.0)
    x = np.arange(20.0, 100.0, 0.02)
    bkg = ChebyshevBackground([5.0], x[0], x[-1])
    res = calculate_pattern(x, [PhaseContribution(_si(), scale=1.0)], inst, SampleGeometry(), bkg)
    assert res.y_calc.shape == x.shape
    assert np.allclose(res.y_background, 5.0)
    assert len(res.peaks) == 1
    tbl = res.peaks[0]
    i111 = int(np.where(np.all(tbl.hkl == [1, 1, 1], axis=1))[0][0])
    assert tbl.two_theta[i111] == pytest.approx(28.44, abs=0.01)
    # 最強ピークは (111)
    assert tbl.intensity.argmax() == i111
    # 各ピークの面積 ≈ 強度(プロファイルは面積 1)
    peak_area = np.sum(res.y_bragg) * 0.02
    assert peak_area == pytest.approx(tbl.intensity.sum(), rel=0.02)
    # 位置: y_calc の最大は 28.44 付近
    assert x[np.argmax(res.y_calc)] == pytest.approx(28.44, abs=0.03)


def test_kalpha2_doublet_and_zero_shift() -> None:
    inst = CWInstrument(Radiation.characteristic("Cu"), w=0.002, zero=0.1)
    x = np.arange(27.5, 29.5, 0.005)
    res = calculate_pattern(x, [PhaseContribution(_si())], inst)
    y = res.y_calc
    # 2 本のピーク(Kα1, Kα2)。Kα2 は約 0.07° 高角側、強度比 ~0.5
    peaks = [
        i
        for i in range(1, len(y) - 1)
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 0.1 * y.max()
    ]
    assert len(peaks) == 2
    assert x[peaks[0]] == pytest.approx(28.44 + 0.1, abs=0.02)
    assert x[peaks[1]] - x[peaks[0]] == pytest.approx(0.072, abs=0.01)
    assert y[peaks[1]] / y[peaks[0]] == pytest.approx(0.5, abs=0.05)
    res_gsas = calculate_pattern(x, [PhaseContribution(_si())], inst, kalpha2_mode="gsas")
    assert np.allclose(res_gsas.y_calc, y, rtol=0.02, atol=0.02 * y.max())
