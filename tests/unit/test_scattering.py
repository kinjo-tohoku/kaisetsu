import numpy as np
import pytest

from kaisetsu.scattering import NeutronScatteringLengths, Radiation, XrayScatteringFactors
from kaisetsu.scattering.radiation import HC_EV_ANGSTROM


def test_f0_at_zero_equals_electron_count() -> None:
    xr = XrayScatteringFactors()
    # s = 0 では f0 = Σa + c ≈ Z(中性原子)。解析近似なので 0.1% 程度ずれる
    for sym, z in (("H", 1), ("C", 6), ("O", 8), ("Si", 14), ("Fe", 26), ("Pb", 82)):
        assert float(xr.f0(sym, 0.0)) == pytest.approx(z, rel=2e-3, abs=0.02)
    assert float(xr.f0("O2-", 0.0)) == pytest.approx(10, rel=2e-3)
    assert float(xr.f0("Pb2+", 0.0)) == pytest.approx(80, rel=2e-3)


def test_f0_decreases_with_s() -> None:
    xr = XrayScatteringFactors()
    s = np.linspace(0, 1.0, 11)
    f = xr.f0("Pb", s)
    assert f.shape == s.shape
    assert np.all(np.diff(f) < 0)


def test_wk1995_close_to_it1992() -> None:
    a = XrayScatteringFactors("it1992")
    b = XrayScatteringFactors("wk1995")
    s = np.linspace(0, 1.0, 21)
    for sym in ("O", "Si", "Fe", "Pb"):
        assert np.allclose(a.f0(sym, s), b.f0(sym, s), rtol=0.01, atol=0.05)


def test_ion_fallback_to_neutral() -> None:
    xr = XrayScatteringFactors()
    assert xr.resolve("Pb2+") == "Pb2+"
    assert xr.resolve("Pb+2") == "Pb2+"
    assert xr.resolve("S6+") == "S"  # テーブルに無いイオンは中性原子に落ちる
    with pytest.raises(KeyError):
        xr.resolve("Xx")


def test_anomalous_cu_kalpha() -> None:
    xr = XrayScatteringFactors()
    lam = 1.5405929
    fe = xr.anomalous("Fe", lam)
    # 文献値(IT Vol. C Table 4.2.6.8 付近): Fe f' ≈ -1.13, f'' ≈ 3.20 (Cu Kα)
    assert fe.real == pytest.approx(-1.13, abs=0.05)
    assert fe.imag == pytest.approx(3.20, abs=0.05)
    pb = xr.anomalous("Pb2+", lam)
    assert pb.real == pytest.approx(-4.0, abs=0.3)
    assert pb.imag == pytest.approx(8.5, abs=0.2)
    # 吸収端から遠い軽元素では小さい
    o = xr.anomalous("O", lam)
    assert abs(o.real) < 0.1 and 0.0 < o.imag < 0.05


def test_neutron_lengths() -> None:
    nt = NeutronScatteringLengths()
    assert nt.scattering_length("Pb") == pytest.approx(9.405)
    assert nt.scattering_length("Pb2+") == pytest.approx(9.405)
    assert nt.scattering_length("O").real == pytest.approx(5.803)
    assert nt.scattering_length("Ti").real == pytest.approx(-3.438)
    assert nt.scattering_length("D").real == pytest.approx(6.671)
    assert nt.scattering_length("B").imag == pytest.approx(-0.213)
    assert nt.absorption_cross_section("B") == pytest.approx(767.0)
    with pytest.raises(KeyError):
        nt.scattering_length("Xx")


def test_radiation() -> None:
    cu = Radiation.characteristic("Cu")
    assert cu.is_xray
    assert cu.wavelength == pytest.approx(1.5405929)
    assert cu.wavelengths == ((1.5405929, 1.0), (1.5444274, 0.5))
    assert cu.energy_ev == pytest.approx(HC_EV_ANGSTROM / 1.5405929)
    assert cu.energy_kev == pytest.approx(8.048, abs=0.001)
    mono = Radiation.characteristic("Cu", kalpha2=False)
    assert mono.wavelengths == ((1.5405929, 1.0),)
    n = Radiation.neutron(1.909)
    assert n.is_neutron and not n.is_xray
    with pytest.raises(ValueError):
        Radiation.neutron(-1)
    with pytest.raises(ValueError):
        Radiation.characteristic("Xx")
