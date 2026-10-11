"""一軸異方的な結晶子サイズ広がり: 方向依存、等方との整合、TOF、Project のパラメータと保存。"""

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
from kaisetsu.pattern.tof import TOFInstrument, calculate_pattern_tof
from kaisetsu.profile.corrections import axis_cos2


def _hex() -> Structure:
    return Structure(
        Lattice.hexagonal(2.51, 4.07),
        SpaceGroup.from_symbol("P 63/m m c"),
        [AtomSite("Co", "Co", [1 / 3, 2 / 3, 1 / 4], u_iso=0.006)],
    )


def test_axis_cos2() -> None:
    st = _hex()
    hkl = np.array([[0, 0, 2], [1, 0, 0], [1, 1, 0], [1, 0, 1]])
    c2 = axis_cos2(hkl, st.lattice, st.spacegroup, (0, 0, 1))
    assert c2[0] == pytest.approx(1.0)
    assert c2[1] == pytest.approx(0.0, abs=1e-12)
    assert c2[2] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < c2[3] < 1.0
    with pytest.raises(ValueError):
        axis_cos2(hkl, st.lattice, st.spacegroup, (0, 0, 0))


def test_uniaxial_widths_by_direction() -> None:
    st = _hex()
    hkl = np.array([[0, 0, 2], [1, 0, 0], [1, 0, 1]])
    tt = st.lattice.two_theta(hkl, 1.5406)
    inv_d2 = st.lattice.inverse_d_squared(hkl)
    # 板状(c 軸方向に薄い): 00l が太く、hk0 は等方値と同じ
    pc = PhaseContribution(st, size_um=0.2, size_axis=(0, 0, 1), size_axial_um=0.05)
    _, l_ua = pc.sample_broadening(tt, 1.5406, hkl, inv_d2)
    _, l_iso = PhaseContribution(st, size_um=0.2).sample_broadening(tt, 1.5406, hkl, inv_d2)
    assert l_ua[0] == pytest.approx(4.0 * l_iso[0], rel=1e-6)
    assert l_ua[1] == pytest.approx(l_iso[1], rel=1e-6)
    assert l_iso[2] < l_ua[2] < 4.0 * l_iso[2]
    # 軸方向のサイズが等方値と同じなら等方に戻る
    pc_same = PhaseContribution(st, size_um=0.2, size_axial_um=0.2)
    _, l_same = pc_same.sample_broadening(tt, 1.5406, hkl, inv_d2)
    assert np.allclose(l_same, l_iso)
    # hkl を渡さなければ等方値
    _, l_nohkl = pc.sample_broadening(tt, 1.5406)
    assert np.allclose(l_nohkl, l_iso)


def test_uniaxial_tof() -> None:
    st = _hex()
    inst = TOFInstrument(difc=20000.0, alpha=0.2, beta0=0.05, beta1=0.002, sig1=40.0)
    x = np.arange(10000.0, 90000.0, 5.0)
    pc_iso = PhaseContribution(st, scale=100.0, size_um=0.1)
    pc_ua = PhaseContribution(st, scale=100.0, size_um=0.1, size_axial_um=0.02)
    t_iso = calculate_pattern_tof(x, [pc_iso], inst).peaks[0]
    t_ua = calculate_pattern_tof(x, [pc_ua], inst).peaks[0]
    k002 = int(np.argmin(np.abs(t_iso.d - st.lattice.d_spacing([[0, 0, 2]])[0])))
    k100 = int(np.argmin(np.abs(t_iso.d - st.lattice.d_spacing([[1, 0, 0]])[0])))
    assert t_ua.fwhm_lorentz[k002] > t_iso.fwhm_lorentz[k002] * 2.0
    assert t_ua.fwhm_lorentz[k100] == pytest.approx(t_iso.fwhm_lorentz[k100])


def test_project_refine_and_roundtrip(tmp_path: Path) -> None:
    st = _hex()
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, polarization=0.7)
    x = np.arange(30.0, 110.0, 0.02)
    truth = calculate_pattern(
        x,
        [PhaseContribution(st, scale=50.0, size_um=0.15, size_axial_um=0.04)],
        inst,
        SampleGeometry(),
    )
    proj = Project()
    proj.add_phase(st.copy(), name="Co")
    proj.add_histogram(PowderData(x, truth.y_calc + 5.0), inst, SampleGeometry(), n_background=2)
    pc = proj.phase_contribution(0, "Co")
    pc.scale = 50.0
    pc.size_um = 0.1
    proj.enable_uniaxial_size(0, "Co", (0, 0, 1))
    assert "h0:Co:size_ax" in proj.params.names()
    assert pc.size_axial_um == pytest.approx(0.1)
    proj.vary("h0:bkg:*", "h0:Co:size", "h0:Co:size_ax")
    fit = proj.refine(max_cycles=20)
    assert fit.success
    assert proj.params["h0:Co:size"].value == pytest.approx(0.15, rel=0.05)
    assert proj.params["h0:Co:size_ax"].value == pytest.approx(0.04, rel=0.05)
    path = save_project(proj, tmp_path / "u.json")
    p2 = load_project(path)
    pc2 = p2.phase_contribution(0, "Co")
    assert pc2.size_axis == (0, 0, 1)
    assert pc2.size_axial_um == pytest.approx(pc.size_axial_um)
    assert "h0:Co:size_ax" in p2.params.names()
    proj.enable_uniaxial_size(0, "Co", enable=False)
    assert "h0:Co:size_ax" not in proj.params.names()
    assert pc.size_axial_um is None


def test_from_gsas2_uniaxial() -> None:
    st = _hex()
    pc = PhaseContribution.from_gsas2(
        st,
        {
            "Scale": [120.0, True],
            "Pref.Ori.": ["MD", 1.0, False, [0, 0, 1], 0, {}, [], 0.1],
            "Size": ["uniaxial", [0.5, 0.1, 1.0], [False, False, False], [0, 0, 1], [0.0] * 6],
            "Mustrain": ["isotropic", [1000.0, 1000.0, 1.0], [False, False, False], [0, 0, 1]],
        },
    )
    assert pc.size_um == pytest.approx(0.5)
    assert pc.size_axial_um == pytest.approx(0.1)
    assert pc.size_axis == (0, 0, 1)
