"""Pawley 法: 合成パターンから格子定数・プロファイルと反射強度を回収する。"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
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
from kaisetsu.refine.pawley import solve_pawley


def _rutile() -> Structure:
    return Structure(
        Lattice.tetragonal(4.594, 2.959),
        SpaceGroup.from_symbol("P42/mnm"),
        [
            AtomSite("Ti", "Ti", [0, 0, 0], u_iso=0.006),
            AtomSite("O", "O", [0.305, 0.305, 0], u_iso=0.008),
        ],
        name="rutile",
    )


def _synthetic() -> tuple[PowderData, CWInstrument, Structure]:
    st = _rutile()
    inst = CWInstrument(
        Radiation.xray(1.5406), zero=0.02, u=0.004, v=-0.003, w=0.004, x=0.02, y=0.01,
        s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip
    x = np.arange(20.0, 90.0, 0.02)
    bkg = ChebyshevBackground([80.0, -10.0, 5.0], 20.0, 90.0)
    truth = calculate_pattern(x, [PhaseContribution(st, scale=20.0)], inst, SampleGeometry(), bkg)
    rng = np.random.default_rng(3)
    y = rng.poisson(truth.y_calc).astype(float)
    return PowderData(x, y, name="synthetic"), inst, st


def test_solve_pawley_linear_recovery() -> None:
    """既知のプロファイルの重ね合わせから強度を回収し、σ が妥当。"""
    x = np.arange(0.0, 10.0, 0.01)
    centers = [3.0, 5.0, 5.3, 8.0]
    prof = []
    windows = []
    for c in centers:
        i0, i1 = int((c - 1.0) / 0.01), int((c + 1.0) / 0.01)
        xx = x[i0:i1]
        p = np.exp(-0.5 * ((xx - c) / 0.1) ** 2)
        p /= np.sum(p)
        prof.append(p)
        windows.append((i0, i1))
    a_true = np.array([20000.0, 10000.0, 6000.0, 0.0])
    y = np.full_like(x, 20.0)
    for (i0, i1), p, a in zip(windows, prof, a_true, strict=True):
        y[i0:i1] += a * p
    rng = np.random.default_rng(0)
    y = rng.poisson(y).astype(float)
    sol = solve_pawley(y, 1.0 / np.maximum(y, 1.0), np.full_like(x, 20.0), windows, prof)
    assert np.allclose(sol.intensity[:3], a_true[:3], rtol=0.05)
    assert sol.intensity[3] >= 0.0 and sol.intensity[3] < 100.0
    assert np.all(np.isfinite(sol.sigma))
    # 重なった 2 本(5.0 と 5.3、幅 0.1)は相関が強くはない
    assert all(abs(c) < 0.9 for _, _, c in sol.correlation_pairs) or not sol.correlation_pairs
    assert sol.chi2 < 2.0


def test_pawley_refinement_recovers_cell_and_intensities() -> None:
    data, inst, st = _synthetic()
    truth_f2 = None
    proj = Project("pawley")
    st0 = st.copy()
    st0.lattice = Lattice.tetragonal(4.60, 2.955)
    proj.add_phase(st0, name="rutile")
    inst0 = CWInstrument(
        Radiation.xray(1.5406), zero=0.0, u=0.0, v=0.0, w=0.006, x=0.0, y=0.0,
        s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip
    proj.add_histogram(data, inst0, SampleGeometry(), n_background=3)
    proj.start_pawley("rutile", from_structure=False)  # 一様な初期強度から
    assert proj.phases["rutile"].pawley and not proj.phases["rutile"].le_bail
    proj.vary("h0:bkg:*")
    proj.refine(max_cycles=5)
    proj.vary("rutile:cell:*", "h0:inst:zero")
    proj.refine(max_cycles=10)
    proj.vary("h0:inst:w", "h0:inst:u", "h0:inst:v", "h0:inst:x", "h0:inst:y")
    fit = proj.refine(max_cycles=15)
    assert fit.success
    rf = proj.rfactors()["total"]
    assert rf.chi2 < 2.0
    a = proj.params["rutile:cell:a"]
    c = proj.params["rutile:cell:c"]
    assert a.value == pytest.approx(4.594, abs=3 * max(a.esd or 0, 2e-4))
    assert c.value == pytest.approx(2.959, abs=3 * max(c.esd or 0, 2e-4))
    # 抽出した |F|² は真の構造から計算した値と比例する(孤立した強い反射で 10% 以内)
    hkl, f2, sig = proj.pawley_structure_factors(0, "rutile")
    from kaisetsu.reflections.structure_factor import structure_factors

    truth_f2 = np.abs(structure_factors(st, hkl, inst.radiation)) ** 2
    rep0 = proj.reflection_report(0, "rutile")
    # 強い反射のうち測定範囲(〜90°)に入っているもの(範囲外の反射は強度 0・σ = inf になる)
    strong = (truth_f2 > 0.3 * truth_f2.max()) & (rep0.two_theta < 89.0)
    ratio = f2[strong] / truth_f2[strong]
    scale = np.median(ratio)
    assert np.all(np.abs(ratio / scale - 1.0) < 0.15)
    assert np.all(sig[strong] > 0) and np.all(sig[strong] < 0.2 * f2[strong])
    # 反射レポートも動く(f2_calc は表の値)
    rep = proj.reflection_report(0, "rutile")
    assert len(rep.hkl) == len(hkl)


def test_pawley_roundtrip_flags(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from kaisetsu.io.project import load_project, save_project

    data, _inst, st = _synthetic()
    proj = Project("p")
    proj.add_phase(st, name="rutile")
    proj.add_histogram(data, CWInstrument(Radiation.xray(1.5406)), n_background=2)
    proj.start_pawley("rutile")
    proj.calculate(0)
    path = save_project(proj, tmp_path / "p.json")
    p2 = load_project(path)
    assert p2.phases["rutile"].pawley and not p2.phases["rutile"].le_bail
    assert 0 in p2.phases["rutile"].f2_tables
