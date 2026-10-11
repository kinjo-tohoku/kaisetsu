"""Stephens 異方的歪み: ラウエ群の拘束、等方との等価性、異方性の効き方、Project への組み込み。"""

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
from kaisetsu.profile.stephens import (
    MONOMIALS,
    isotropic_coefficients,
    monomial_name,
    stephens_constraint,
    stephens_fwhm,
)


def test_laue_constraints() -> None:
    # 立方晶: 自由 2 個(S_400 系と S_220 系)
    c = stephens_constraint(SpaceGroup.from_symbol("F m -3 m"))
    assert c.n_free == 2
    names = {monomial_name(MONOMIALS[i]) for i in c.free}
    assert names == {"S_400", "S_220"}
    assert len(c.fixed) == 15 - 2 - len(c.dependent)
    # 正方晶 4/mmm: 4 個(S_400=S_040, S_004, S_220, S_202=S_022)
    assert stephens_constraint(SpaceGroup.from_symbol("P 4/m m m")).n_free == 4
    # 斜方晶: 6 個(偶数次のみ)
    assert stephens_constraint(SpaceGroup.from_symbol("P n m a")).n_free == 6
    # 六方晶 6/mmm: 3 個
    assert stephens_constraint(SpaceGroup.from_symbol("P 63/m m c")).n_free == 3
    # 単斜晶: 9 個、三斜晶: 15 個
    assert stephens_constraint(SpaceGroup.from_symbol("P 1 21/c 1")).n_free == 9
    assert stephens_constraint(SpaceGroup.from_symbol("P -1")).n_free == 15


def test_isotropic_equivalence() -> None:
    lat = Lattice.cubic(4.0)
    coef = isotropic_coefficients(1500.0, lat.reciprocal_metric_tensor)
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0], [3, 1, 1]])
    inv_d2 = lat.inverse_d_squared(hkl)
    tt = lat.two_theta(hkl, 1.5406)
    g = stephens_fwhm(hkl, inv_d2, tt, coef)
    expected = (180 / np.pi) * 1500e-6 * np.tan(np.radians(tt / 2))
    assert np.allclose(g, expected, rtol=1e-6)
    # PhaseContribution 経由でも同じ
    st = Structure(lat, SpaceGroup.from_symbol("P m -3 m"), [AtomSite("Ni", "Ni", [0, 0, 0])])
    pc_iso = PhaseContribution(st, strain_e6=1500.0)
    pc_ani = PhaseContribution(st, stephens=coef)
    g2_iso, l_iso = pc_iso.sample_broadening(tt, 1.5406, hkl, inv_d2)
    g2_ani, l_ani = pc_ani.sample_broadening(tt, 1.5406, hkl, inv_d2)
    assert np.allclose(l_iso, l_ani) and np.allclose(g2_iso, g2_ani)


def test_anisotropy_changes_widths_by_direction() -> None:
    lat = Lattice.cubic(4.0)
    coef = np.zeros(15)
    coef[MONOMIALS.index((4, 0, 0))] = 1e6  # h⁴ のみ(k, l 方向にも等価だが後で対称化)
    sg = SpaceGroup.from_symbol("P m -3 m")
    c = stephens_constraint(sg, coef)
    sym = c.symmetrized  # S_400 = S_040 = S_004 になる
    assert sym[MONOMIALS.index((0, 4, 0))] == pytest.approx(sym[MONOMIALS.index((4, 0, 0))])
    hkl = np.array([[2, 0, 0], [1, 1, 1]])
    tt = lat.two_theta(hkl, 1.5406)
    g = stephens_fwhm(hkl, lat.inverse_d_squared(hkl), tt, sym)
    # (h00) と (hhh) で幅の比が異なる: σ²(M) ∝ h⁴+k⁴+l⁴ なので (111) は (200) より相対的に狭い
    rel = g / np.tan(np.radians(tt / 2))
    assert rel[1] < rel[0]


def test_project_parameters_and_roundtrip(tmp_path: Path) -> None:
    st = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("P m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.005)],
    )
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, polarization=0.7)
    x = np.arange(20.0, 100.0, 0.02)
    truth = calculate_pattern(
        x, [PhaseContribution(st, scale=50.0, strain_e6=2000.0)], inst, SampleGeometry()
    )
    proj = Project()
    proj.add_phase(st.copy(), name="Ni")
    proj.add_histogram(PowderData(x, truth.y_calc + 10.0), inst, SampleGeometry(), n_background=2)
    proj.phase_contribution(0, "Ni").scale = 50.0
    proj.phase_contribution(0, "Ni").strain_e6 = 500.0
    proj.enable_stephens(0, "Ni")
    names = proj.params.names()
    assert "h0:Ni:S_400" in names and "h0:Ni:S_220" in names
    assert proj.params["h0:Ni:S_040"].tie is not None  # 立方晶の拘束
    assert proj.params["h0:Ni:S_310"].locked
    free = [
        n
        for n in names
        if (n.startswith("h0:Ni:S_") and proj.params[n].is_free)
        or (n.startswith("h0:Ni:S_") and not proj.params[n].locked and proj.params[n].tie is None)
    ]
    assert set(free) == {"h0:Ni:S_400", "h0:Ni:S_220"}
    proj.vary("h0:bkg:*", "h0:Ni:S_400", "h0:Ni:S_220")
    fit = proj.refine(max_cycles=15)
    assert fit.success
    # 等方的な真値(ε = 2000)に相当する S に近づく: S_400 と S_220 の比が等方(S_220 = 2 S_400 /3?)
    coef_true = isotropic_coefficients(2000.0, st.lattice.reciprocal_metric_tensor)
    s400 = proj.params["h0:Ni:S_400"].value
    assert s400 == pytest.approx(coef_true[MONOMIALS.index((4, 0, 0))], rel=0.15)
    path = save_project(proj, tmp_path / "s.json")
    p2 = load_project(path)
    pc2 = p2.phase_contribution(0, "Ni")
    assert pc2.stephens is not None and np.allclose(
        pc2.stephens, proj.phase_contribution(0, "Ni").stephens
    )
    assert "h0:Ni:S_400" in p2.params.names()
    proj.enable_stephens(0, "Ni", False)
    assert "h0:Ni:S_400" not in proj.params.names()
