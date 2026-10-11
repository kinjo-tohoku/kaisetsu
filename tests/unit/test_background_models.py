"""背景モデル: 固定点補間、非晶質ハンプ、Project への組み込み、保存、ヤコビアン。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    CWInstrument,
    Lattice,
    PowderData,
    Project,
    Radiation,
    SpaceGroup,
    Structure,
)
from kaisetsu.io.project import load_project, save_project
from kaisetsu.profile.background import GaussianHump, InterpolatedBackground
from kaisetsu.refine.jacobian import classify, project_jacobian
from kaisetsu.refine.lsq import numerical_jacobian


def test_interpolated_background_basis_and_call() -> None:
    bk = InterpolatedBackground([10.0, 20.0, 40.0], [100.0, 50.0, 80.0], 10.0, 40.0)
    x = np.array([5.0, 10.0, 15.0, 30.0, 45.0])
    y = bk(x)
    assert y[0] == 100.0 and y[1] == 100.0 and y[2] == 75.0 and y[3] == 65.0 and y[4] == 80.0
    b = bk.basis(x)
    assert b.shape == (5, 3)
    assert np.allclose(b @ bk.coefficients, y)
    assert np.allclose(b.sum(axis=1), 1.0)  # 帽子関数の和は 1
    with pytest.raises(ValueError):
        InterpolatedBackground([10.0, 5.0], [1.0, 2.0])


def test_from_pattern_follows_envelope() -> None:
    x = np.arange(10.0, 90.0, 0.02)
    bkg_true = 200.0 * np.exp(-(x - 10.0) / 15.0) + 30.0
    y = bkg_true + 500.0 * np.exp(-0.5 * ((x - 40.0) / 0.1) ** 2)
    bk = InterpolatedBackground.from_pattern(x, y, 12)
    assert bk.order == 12
    assert np.all(np.abs(bk(x) - bkg_true) < 0.15 * bkg_true + 5.0)


def test_gaussian_hump() -> None:
    h = GaussianHump(30.0, 10.0, 100.0)
    assert h(np.array([30.0]))[0] == pytest.approx(100.0)
    assert h(np.array([35.0]))[0] == pytest.approx(50.0)  # 半値全幅


def _proj() -> Project:
    st = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("P m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.005)],
    )
    x = np.arange(10.0, 90.0, 0.05)
    rng = np.random.default_rng(0)
    y = 50.0 + 30.0 * np.exp(-0.5 * ((x - 25.0) / 6.0) ** 2) + rng.normal(0, 2, len(x))
    proj = Project()
    proj.add_phase(st, name="Ni")
    proj.add_histogram(
        PowderData(x, np.maximum(y, 1.0)), CWInstrument(Radiation.xray(1.5406)), n_background=4
    )
    # データは背景とハンプだけなので Bragg ピークは消しておく
    proj.phase_contribution(0, "Ni").scale = 0.0
    proj.build_parameters()
    return proj


def test_project_points_background_and_humps(tmp_path: Path) -> None:
    proj = _proj()
    proj.reset_background(0, 6, True, kind="points")
    h = proj.histograms[0]
    assert getattr(h.background, "kind", "") == "points" and h.background.order == 6
    names = proj.params.names()
    assert "h0:bkg:5" in names and "固定点" in proj.params["h0:bkg:0"].description
    proj.add_hump(0, 25.0, 12.0, 20.0)
    assert {"h0:hump0:center", "h0:hump0:fwhm", "h0:hump0:height"} <= set(proj.params.names())
    y0 = h.background_total(np.array([25.0]))[0]
    assert y0 > h.background(np.array([25.0]))[0]
    # 保存と読み込み
    path = save_project(proj, tmp_path / "b.json")
    p2 = load_project(path)
    h2 = p2.histograms[0]
    assert getattr(h2.background, "kind", "") == "points"
    assert np.allclose(h2.background.coefficients, h.background.coefficients)
    assert len(h2.humps) == 1 and h2.humps[0].fwhm == 12.0
    assert np.allclose(p2.calculate(0).y_background, proj.calculate(0).y_background)
    # 項数だけ変える(推定し直さない)と形が引き継がれる
    proj.reset_background(0, 9, False)
    assert h.background.order == 9 and getattr(h.background, "kind", "") == "points"
    proj.clear_humps(0)
    assert not any(n.startswith("h0:hump") for n in proj.params.names())


def test_hump_refinement_and_jacobian() -> None:
    proj = _proj()
    proj.add_hump(0, 22.0, 8.0, 10.0)
    proj.vary("h0:bkg:*", "h0:hump0:*")
    ps = proj.params
    assert classify(proj, ps["h0:hump0:center"])[0] == "other"
    base = proj.residuals()
    j_an = project_jacobian(proj, base)
    x0 = ps.vector()
    lo, hi = ps.bounds()

    def fun(x: np.ndarray) -> np.ndarray:
        ps.set_vector(x)
        return proj.residuals()

    j_num = numerical_jacobian(fun, x0, ps.steps(), lo, hi, fun(x0))
    ps.set_vector(x0)
    for col in range(len(x0)):
        scale = max(float(np.max(np.abs(j_num[:, col]))), 1e-12)
        assert float(np.max(np.abs(j_an[:, col] - j_num[:, col]))) / scale < 2e-3
    fit = proj.refine(max_cycles=15)
    assert fit.success
    assert ps["h0:hump0:center"].value == pytest.approx(25.0, abs=0.5)
    assert ps["h0:hump0:fwhm"].value == pytest.approx(6.0 * 2.3548, abs=1.5)
