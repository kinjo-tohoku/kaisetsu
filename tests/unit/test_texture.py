"""球面調和関数の集合組織: 基底の次元、不変性、規格化、Project での精密化と保存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    CWInstrument,
    Lattice,
    PhaseContribution,
    PowderData,
    Project,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.io.project import load_project, save_project
from kaisetsu.profile.texture import (
    basis_size,
    coefficient_names,
    harmonics_matrix,
    real_spherical_harmonics,
    symmetrized_harmonics,
    texture_factor,
    texture_index,
)


def _sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def test_real_harmonics_orthonormal() -> None:
    pts = _sphere(6000)
    for ell in (2, 4, 6):
        y = real_spherical_harmonics(ell, pts)
        gram = y.T @ y / len(pts)
        assert np.allclose(gram, np.eye(2 * ell + 1), atol=2e-3)


@pytest.mark.parametrize(
    ("symbol", "lattice", "dims"),
    [
        ("F m -3 m", Lattice.cubic(4.0), {4: 1, 6: 1, 8: 1}),
        ("P 63/m m c", Lattice.hexagonal(3.0, 5.0), {2: 1, 4: 1, 6: 2, 8: 2}),
        ("P 4/m m m", Lattice.tetragonal(3.0, 5.0), {2: 1, 4: 2, 6: 2, 8: 3}),
        ("P n m a", Lattice.orthorhombic(3.0, 4.0, 5.0), {2: 2, 4: 3, 6: 4, 8: 5}),
        ("P 1 21/c 1", Lattice.monoclinic(3.0, 4.0, 5.0, 100.0), {2: 3, 4: 5, 6: 7, 8: 9}),
        ("P -1", Lattice.from_parameters([3, 4, 5, 80, 85, 95]), {2: 5, 4: 9, 6: 13, 8: 17}),
    ],
)
def test_invariant_dimensions(symbol: str, lattice: Lattice, dims: dict[int, int]) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    basis = symmetrized_harmonics(sg, lattice, 8)
    assert {ell: u.shape[1] for ell, u in basis} == dims
    # 等価反射で同じ値
    for h in ([1, 2, 3], [2, 0, 1], [1, 1, 0]):
        eq = sg.laue_equivalents(h)
        k = harmonics_matrix(eq, lattice, basis)
        assert np.allclose(k, k[0], atol=1e-6)
    # 基底は球面上で正規直交(二乗平均 1)
    pts = _sphere(6000)
    for ell, u in basis:
        k = real_spherical_harmonics(ell, pts) @ u
        assert np.allclose(k.T @ k / len(pts), np.eye(u.shape[1]), atol=3e-3)


def test_cubic_harmonic_shape() -> None:
    """立方晶の l = 4 の不変調和関数は x⁴ + y⁴ + z⁴ − 3/5 に比例する。"""
    sg = SpaceGroup.from_symbol("P m -3 m")
    lat = Lattice.cubic(4.0)
    basis = symmetrized_harmonics(sg, lat, 4)
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0], [3, 1, 1]])
    k = harmonics_matrix(hkl, lat, basis)[:, 0]
    u = hkl / np.linalg.norm(hkl, axis=1)[:, None]
    ref = np.sum(u**4, axis=1) - 0.6
    ratio = k / ref
    assert np.allclose(ratio, ratio[0], rtol=1e-6)


def test_texture_factor_and_index() -> None:
    sg = SpaceGroup.from_symbol("P 63/m m c")
    lat = Lattice.hexagonal(3.0, 5.0)
    basis = symmetrized_harmonics(sg, lat, 2)
    assert basis_size(basis) == 1 and coefficient_names(basis) == ["sh2_1"]
    hkl = np.array([[0, 0, 2], [1, 0, 0], [1, 0, 1]])
    o = texture_factor(hkl, lat, basis, [0.3])
    # 002 と 100 は P2(cosθ) の符号が逆(軸方向で +、垂直で −)
    assert (o[0] - 1.0) * (o[1] - 1.0) < 0
    assert np.allclose(texture_factor(hkl, lat, basis, [0.0]), 1.0)
    assert texture_index([0.3, 0.4]) == pytest.approx(1.25)


def test_project_refine_recovers_texture(tmp_path: Path) -> None:
    st = Structure(
        Lattice.hexagonal(2.51, 4.07),
        SpaceGroup.from_symbol("P 63/m m c"),
        [AtomSite("Co", "Co", [1 / 3, 2 / 3, 1 / 4], u_iso=0.006)],
    )
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, polarization=0.7)
    x = np.arange(30.0, 110.0, 0.02)
    truth_pc = PhaseContribution(st, scale=50.0, sh_order=4)
    truth_pc.sh_coefficients = np.array([0.25, -0.1])
    truth = calculate_pattern(x, [truth_pc], inst, SampleGeometry())
    proj = Project()
    proj.add_phase(st.copy(), name="Co")
    proj.add_histogram(PowderData(x, truth.y_calc + 5.0), inst, SampleGeometry(), n_background=2)
    proj.phase_contribution(0, "Co").scale = 50.0
    proj.enable_texture(0, "Co", 4)
    names = proj.params.names()
    assert "h0:Co:sh2_1" in names and "h0:Co:sh4_1" in names and "h0:Co:sh6_1" not in names
    proj.vary("h0:bkg:*", "h0:Co:sh*", "h0:Co:scale")
    fit = proj.refine(max_cycles=20)
    assert fit.success
    assert proj.params["h0:Co:sh2_1"].value == pytest.approx(0.25, abs=0.01)
    assert proj.params["h0:Co:sh4_1"].value == pytest.approx(-0.1, abs=0.01)
    assert proj.texture_index(0, "Co") == pytest.approx(1.0 + 0.25**2 + 0.1**2, abs=0.01)
    assert "集合組織指数" in proj.summary()
    # 集合組織の係数の解析ヤコビアンが数値微分と一致する
    from kaisetsu.refine.jacobian import classify, project_jacobian
    from kaisetsu.refine.lsq import numerical_jacobian

    ps = proj.params
    base = proj.residuals()
    j_an = project_jacobian(proj, base)
    x0 = ps.vector()
    lo, hi = ps.bounds()

    def fun(x: np.ndarray) -> np.ndarray:
        ps.set_vector(x)
        return proj.residuals()

    j_num = numerical_jacobian(fun, x0, ps.steps(), lo, hi, fun(x0))
    ps.set_vector(x0)
    for col, name in enumerate(ps.free_names()):
        if classify(proj, ps[name])[0] != "po":
            continue
        a, n = j_an[:, col], j_num[:, col]
        assert np.max(np.abs(a - n)) / max(float(np.max(np.abs(n))), 1e-12) < 2e-2, name
    path = save_project(proj, tmp_path / "t.json")
    p2 = load_project(path)
    pc2 = p2.phase_contribution(0, "Co")
    assert pc2.sh_order == 4 and pc2.sh_coefficients is not None
    assert np.allclose(pc2.sh_coefficients, proj.phase_contribution(0, "Co").sh_coefficients)
    assert "h0:Co:sh2_1" in p2.params.names()
    proj.enable_texture(0, "Co", enable=False)
    assert "h0:Co:sh2_1" not in proj.params.names()
    with pytest.raises(ValueError):
        proj.enable_texture(0, "Co", 3)
