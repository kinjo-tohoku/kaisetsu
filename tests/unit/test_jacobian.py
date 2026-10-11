"""半解析ヤコビアンが数値微分と一致し、精密化結果が変わらず速くなることを確認する。"""

from __future__ import annotations

import time

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
from kaisetsu.refine.jacobian import classify, project_jacobian
from kaisetsu.refine.lsq import numerical_jacobian


def _project() -> Project:
    st = Structure(
        Lattice.tetragonal(4.594, 2.959),
        SpaceGroup.from_symbol("P42/mnm"),
        [
            AtomSite("Ti", "Ti", [0, 0, 0], u_iso=0.006),
            AtomSite("O", "O", [0.305, 0.305, 0], u_iso=0.008, occupancy=0.95),
        ],
        name="rutile",
    )
    inst = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.02, u=0.004, v=-0.003, w=0.004, x=0.02, y=0.01,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    x = np.arange(20.0, 100.0, 0.02)
    bkg = ChebyshevBackground([100.0, -20.0, 5.0, 0.0], 20.0, 100.0)
    truth = calculate_pattern(
        x, [PhaseContribution(st, scale=30.0, march_dollase_r=0.9)], inst, SampleGeometry(), bkg
    )
    rng = np.random.default_rng(5)
    y = rng.poisson(truth.y_calc).astype(float)
    proj = Project("jac")
    proj.add_phase(st.copy(), name="rutile")
    proj.add_histogram(PowderData(x, y), inst, SampleGeometry(), n_background=4)
    proj.phase_contribution(0, "rutile").scale = 28.0
    proj.phase_contribution(0, "rutile").march_dollase_r = 0.92
    proj.build_parameters()
    return proj


def test_classify() -> None:
    proj = _project()
    ps = proj.params
    assert classify(proj, ps["h0:bkg:1"])[0] == "bkg"
    assert classify(proj, ps["h0:rutile:scale"])[0] == "scale"
    assert classify(proj, ps["rutile:O:x"])[0] == "intensity"
    assert classify(proj, ps["rutile:O:occ"])[0] == "intensity"
    assert classify(proj, ps["rutile:Ti:uiso"])[0] == "intensity"
    assert classify(proj, ps["h0:rutile:md_r"])[0] == "po"
    assert classify(proj, ps["rutile:cell:a"])[0] == "other"
    assert classify(proj, ps["h0:inst:w"])[0] == "other"


def test_analytic_matches_numerical() -> None:
    proj = _project()
    proj.vary(
        "h0:bkg:*", "h0:rutile:scale", "rutile:O:x", "rutile:O:occ", "rutile:*:uiso",
        "h0:rutile:md_r", "rutile:cell:a", "h0:inst:w",
    )  # fmt: skip
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
        a, n = j_an[:, col], j_num[:, col]
        scale = max(float(np.max(np.abs(n))), 1e-12)
        err = float(np.max(np.abs(a - n))) / scale
        kind = classify(proj, ps[name])[0]
        tol = 2e-3 if kind in ("bkg", "scale") else 2e-2  # 前進差分 vs 中心差分の差を含む
        assert err < tol, (name, kind, err)
    # ヤコビアン計算のあとモデルは元の状態
    assert np.allclose(proj.residuals(), base)


def test_refine_results_agree_and_faster() -> None:
    results = {}
    times = {}
    for analytic in (False, True):
        proj = _project()
        proj.vary("h0:bkg:*", "h0:rutile:scale")
        proj.refine(max_cycles=5, analytic_jacobian=analytic)
        proj.vary("rutile:O:x", "rutile:*:uiso", "rutile:O:occ", "h0:rutile:md_r", "rutile:cell:*")
        t = time.perf_counter()
        fit = proj.refine(max_cycles=15, analytic_jacobian=analytic)
        times[analytic] = time.perf_counter() - t
        results[analytic] = (fit, {n: proj.params[n].value for n in proj.params.free_names()})
    fa, va = results[True]
    fn, vn = results[False]
    assert fa.success and fn.success
    for n in va:
        esd = fa.esd[fa.names.index(n)]
        assert abs(va[n] - vn[n]) < max(0.5 * esd, 1e-6), (n, va[n], vn[n], esd)
    assert fa.chi2 == pytest.approx(fn.chi2, rel=0.02)
    # 半解析の方が速い(13 変数中 7 つが構造・PO、4 つが背景、1 つがスケール)
    assert times[True] < times[False]
