import math

import numpy as np
import pytest

from kaisetsu.crystal.lattice import CrystalSystem, Lattice


def test_cubic_d_spacing() -> None:
    lat = Lattice.cubic(4.0)
    assert lat.volume == pytest.approx(64.0)
    assert lat.d_spacing([1, 0, 0]) == pytest.approx(4.0)
    assert lat.d_spacing([1, 1, 1]) == pytest.approx(4.0 / math.sqrt(3))
    assert lat.d_spacing([[2, 0, 0], [2, 2, 0]]) == pytest.approx([2.0, math.sqrt(2.0)])


def test_hexagonal_d_spacing_and_volume() -> None:
    a, c = 3.0, 5.0
    lat = Lattice.hexagonal(a, c)
    # 1/d^2 = 4/3 (h^2+hk+k^2)/a^2 + l^2/c^2
    d100 = 1.0 / math.sqrt(4.0 / 3.0 / a**2)
    d001 = c
    d101 = 1.0 / math.sqrt(4.0 / 3.0 / a**2 + 1.0 / c**2)
    assert lat.d_spacing([1, 0, 0]) == pytest.approx(d100)
    assert lat.d_spacing([0, 0, 1]) == pytest.approx(d001)
    assert lat.d_spacing([1, 0, 1]) == pytest.approx(d101)
    assert lat.volume == pytest.approx(a * a * c * math.sqrt(3) / 2)


def test_monoclinic_volume_and_reciprocal() -> None:
    lat = Lattice.monoclinic(5.0, 6.0, 7.0, 100.0)
    assert lat.volume == pytest.approx(5 * 6 * 7 * math.sin(math.radians(100.0)))
    rec = lat.reciprocal
    assert rec.b == pytest.approx(1.0 / 6.0)
    assert rec.beta == pytest.approx(80.0)
    back = rec.reciprocal
    assert np.allclose(back.parameters, lat.parameters)


def test_triclinic_metric_consistency() -> None:
    lat = Lattice(5.1, 6.2, 7.3, 81.0, 95.0, 103.0)
    g = lat.metric_tensor
    assert np.allclose(g, g.T)
    assert np.allclose(g @ lat.reciprocal_metric_tensor, np.eye(3))
    m = lat.orthogonalization_matrix
    # M^T M は計量テンソルに一致する
    assert np.allclose(m.T @ m, g)
    frac = np.array([[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]])
    assert np.allclose(lat.fractional(lat.cartesian(frac)), frac)


def test_two_theta() -> None:
    lat = Lattice.cubic(5.43094)  # Si
    tt = lat.two_theta([1, 1, 1], 1.5405929)
    assert tt == pytest.approx(28.44, abs=0.01)
    assert np.isnan(lat.two_theta([20, 0, 0], 1.54))


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        Lattice(-1, 1, 1)
    with pytest.raises(ValueError):
        Lattice(1, 1, 1, 0.0, 90, 90)
    with pytest.raises(ValueError):
        Lattice(1, 1, 1, 10.0, 10.0, 170.0)  # 幾何学的に閉じない


def test_constrained() -> None:
    lat = Lattice(4.0, 4.1, 4.2, 89.0, 91.0, 90.5)
    assert lat.constrained(CrystalSystem.CUBIC) == Lattice.cubic(4.0)
    assert lat.constrained(CrystalSystem.TETRAGONAL) == Lattice.tetragonal(4.0, 4.2)
    assert lat.constrained(CrystalSystem.MONOCLINIC) == Lattice.monoclinic(4.0, 4.1, 4.2, 91.0)
    assert lat.constrained(CrystalSystem.HEXAGONAL) == Lattice.hexagonal(4.0, 4.2)
