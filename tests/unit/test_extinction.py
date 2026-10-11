"""Sabine 消衰補正(CW / TOF)と TOF の波長依存吸収: 極限、単調性、パターンへの効き方、
ヤコビアン、Project のパラメータと保存。"""

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
from kaisetsu.pattern.instrument import GeometryKind
from kaisetsu.pattern.tof import TOFInstrument, calculate_pattern_tof
from kaisetsu.profile.corrections import (
    cylinder_absorption,
    cylinder_absorption_array,
    sabine_extinction,
)


def _ni() -> Structure:
    return Structure(
        Lattice.cubic(3.524),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.005)],
    )


def test_sabine_limits_and_monotone() -> None:
    f2 = np.array([100.0, 1000.0, 10000.0])
    tt = np.array([30.0, 60.0, 120.0])
    assert np.allclose(sabine_extinction(f2, 1.54, 43.8, tt, 0.0, True), 1.0)
    e = sabine_extinction(f2, 1.54, 43.8, tt, 50.0, True)
    assert np.all(e < 1.0) and np.all(e > 0.0)
    # |F|² が大きいほど強く減る
    e_same = sabine_extinction(np.full(3, 1000.0), 1.54, 43.8, np.full(3, 60.0), 50.0, True)
    e_big = sabine_extinction(np.full(3, 5000.0), 1.54, 43.8, np.full(3, 60.0), 50.0, True)
    assert np.all(e_big < e_same)
    # 小さい x では E ≈ 1 − x/2 cos²θ − x/2 sin²θ = 1 − x/2
    f2s, lam, vol, ext = 10.0, 1.0, 100.0, 1.0
    x = ext * (f2s / 100.0) * (lam / vol) ** 2  # 中性子
    e_small = sabine_extinction(np.array([f2s]), lam, vol, np.array([90.0]), ext, False)[0]
    assert e_small == pytest.approx(1.0 - 0.5 * x, abs=1e-8)
    # x > 1 の漸近式に連続につながる(x = 1 の前後で値が近い)
    vol1 = 1.0
    e_lo = sabine_extinction(np.array([99.9]), 1.0, vol1, np.array([90.0]), 1.0, False)[0]
    e_hi = sabine_extinction(np.array([100.1]), 1.0, vol1, np.array([90.0]), 1.0, False)[0]
    assert abs(e_lo - e_hi) < 0.02


def test_cylinder_absorption_array_matches_scalar() -> None:
    tt = np.array([20.0, 60.0, 120.0, 150.0])
    for mr in (0.0, 0.5, 2.9, 3.0, 3.1, 8.0):
        assert np.allclose(
            cylinder_absorption(mr, tt), cylinder_absorption_array(np.full(4, mr), tt)
        )
    # μR が反射ごとに違う
    mixed = cylinder_absorption_array(np.array([0.0, 1.0, 4.0, 10.0]), tt)
    assert mixed[0] == 1.0 and np.all(np.diff(mixed) < 0)


def test_extinction_reduces_strong_reflections_cw() -> None:
    st = _ni()
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, polarization=0.5)
    x = np.arange(30.0, 120.0, 0.02)
    base = calculate_pattern(x, [PhaseContribution(st, scale=50.0)], inst, SampleGeometry())
    ext = calculate_pattern(
        x, [PhaseContribution(st, scale=50.0, extinction=200.0)], inst, SampleGeometry()
    )
    i0, i1 = base.peaks[0].intensity, ext.peaks[0].intensity
    ratio = i1 / i0
    assert np.all(ratio < 1.0)
    # (111) と (200) は強いので高角の弱い反射より比が小さい
    k111 = int(np.argmin(np.abs(base.peaks[0].d - 2.034)))
    assert ratio[k111] < np.max(ratio)


def test_tof_absorption_and_extinction() -> None:
    st = _ni()
    inst = TOFInstrument(
        difc=20000.0, alpha=0.2, beta0=0.05, beta1=0.002, sig1=40.0, two_theta=90.0
    )
    x = np.arange(10000.0, 60000.0, 5.0)
    pc = PhaseContribution(st, scale=100.0)
    base = calculate_pattern_tof(x, [pc], inst).peaks[0]
    absorbed = calculate_pattern_tof(
        x, [pc], inst, sample=SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, mu_r=0.3)
    ).peaks[0]
    ratio = absorbed.intensity / base.intensity
    # λ = 2 d sinθ が長い(d が大きい)反射ほど吸収が強い
    order = np.argsort(base.d)
    assert np.all(np.diff(ratio[order]) <= 1e-12)
    expected = cylinder_absorption_array(0.3 * 2.0 * base.d * np.sin(np.radians(45.0)), 90.0)
    assert np.allclose(ratio, expected, rtol=1e-10)
    pc_ext = PhaseContribution(st, scale=100.0, extinction=5.0)
    ext = calculate_pattern_tof(x, [pc_ext], inst).peaks[0]
    assert np.all(ext.intensity < base.intensity)


def test_project_param_jacobian_roundtrip(tmp_path: Path) -> None:
    st = _ni()
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, polarization=0.5)
    x = np.arange(30.0, 120.0, 0.02)
    truth = calculate_pattern(
        x, [PhaseContribution(st, scale=50.0, extinction=100.0)], inst, SampleGeometry()
    )
    proj = Project()
    proj.add_phase(st.copy(), name="Ni")
    proj.add_histogram(PowderData(x, truth.y_calc + 5.0), inst, SampleGeometry(), n_background=2)
    pc = proj.phase_contribution(0, "Ni")
    pc.scale = 50.0
    pc.extinction = 30.0
    proj.build_parameters()  # 直接書き換えた値をパラメータ表に反映する
    assert "h0:Ni:ext" in proj.params.names()
    assert proj.params["h0:Ni:ext"].value == pytest.approx(30.0)
    # 消衰ありでも原子パラメータの解析ヤコビアンが数値微分と合う
    from kaisetsu.refine.jacobian import classify, project_jacobian
    from kaisetsu.refine.lsq import numerical_jacobian

    proj.vary("h0:bkg:*", "h0:Ni:scale", "Ni:Ni:uiso", "h0:Ni:ext")
    ps = proj.params
    base = proj.residuals()
    j_an = project_jacobian(proj, base)
    x0 = ps.vector()
    lo, hi = ps.bounds()

    def fun(v: np.ndarray) -> np.ndarray:
        ps.set_vector(v)
        return proj.residuals()

    j_num = numerical_jacobian(fun, x0, ps.steps(), lo, hi, fun(x0))
    ps.set_vector(x0)
    for col, name in enumerate(ps.free_names()):
        a, n = j_an[:, col], j_num[:, col]
        err = float(np.max(np.abs(a - n))) / max(float(np.max(np.abs(n))), 1e-12)
        tol = 2e-3 if classify(proj, ps[name])[0] in ("bkg", "scale") else 2e-2
        assert err < tol, (name, err)
    fit = proj.refine(max_cycles=20)
    assert fit.success
    assert proj.params["h0:Ni:ext"].value == pytest.approx(100.0, rel=0.05)
    path = save_project(proj, tmp_path / "e.json")
    p2 = load_project(path)
    assert p2.phase_contribution(0, "Ni").extinction == pytest.approx(pc.extinction)
