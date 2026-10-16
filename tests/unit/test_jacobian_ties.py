"""ヒストグラムを跨ぐ拘束が半解析ヤコビアンに入ること(外部レビュー 2026-10-17 #3 の回帰テスト)。"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.api import ChebyshevBackground, CWInstrument, PowderData, Project, Radiation


def _tied_project(same_hist: bool = False) -> Project:
    """背景だけの 2 データセット(観測は 1 と 3)。h1 の背景定数を h0 に拘束すると最小二乗解は 2。"""
    p = Project()
    x = np.linspace(20, 30, 21)
    for y in (1.0, 3.0):
        p.add_histogram(
            PowderData(x, np.full_like(x, y), np.ones_like(x)),
            CWInstrument(Radiation.xray(1.54)),
            background=ChebyshevBackground([1.0, 0.0], 20, 30),
        )
    if same_hist:
        p.tie("h0:bkg:1", "h0:bkg:0", [0.5])
        p.vary("h0:bkg:0", "h1:bkg:*")
    else:
        p.tie("h1:bkg:0", "h0:bkg:0")
        p.vary("h0:bkg:0")
    p.params.evaluate_ties()
    p.params.apply_all()
    return p


@pytest.mark.parametrize("same_hist", [False, True])
def test_analytic_jacobian_matches_numeric_with_ties(same_hist: bool) -> None:
    p = _tied_project(same_hist)
    base = p.residuals()
    jac = p.jacobian(base)
    x0 = p.params.vector()
    num = np.empty_like(jac)
    for k in range(len(x0)):
        step = 1e-3
        xp = x0.copy()
        xp[k] += step
        p.params.set_vector(xp)
        num[:, k] = (p.residuals() - base) / step
        p.params.set_vector(x0)
    assert np.allclose(jac, num, atol=1e-6)
    if not same_hist:
        assert np.linalg.norm(jac[21:, 0]) > 1.0  # 第 2 ヒストグラムの行が空でない


def test_refine_with_analytic_jacobian_reaches_common_value() -> None:
    p = _tied_project()
    fit = p.refine(analytic_jacobian=True)
    assert fit.success
    assert p.params["h0:bkg:0"].value == pytest.approx(2.0, abs=1e-6)
    assert p.params["h1:bkg:0"].value == pytest.approx(2.0, abs=1e-6)
