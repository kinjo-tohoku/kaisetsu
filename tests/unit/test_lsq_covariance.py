"""共分散行列が特異・不良条件のヤコビアンでも半正定値で |ρ| ≤ 1 になる。

`np.linalg.inv` は特異に近い行列でも例外を出さずに桁違いの値を返すことがあり、
CaF2 の実験室データで |ρ| = 8.9 の「相関」が表示されていた(ガウス幅が全範囲で負に
切られて V, W の微分が 0 になっていた)。
"""

from __future__ import annotations

import numpy as np

from kaisetsu.refine.lsq import covariance_from_jacobian, refine_least_squares
from kaisetsu.refine.parameters import Parameter, ParameterSet


def _is_psd(cov: np.ndarray) -> bool:
    ev = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    return bool(ev.min() >= -1e-9 * max(float(ev.max()), 1.0))


def test_zero_column_and_parallel_columns() -> None:
    x = np.linspace(0.0, 1.0, 50)
    # 0, 1 列目は同じ(特異方向)、2 列目は 0(感度なし)
    jac = np.column_stack([x, x, np.zeros_like(x), np.ones_like(x), x**2])
    cov, esd, corr, insens, degen = covariance_from_jacobian(jac, 1.5)
    assert insens == [2]
    assert esd[2] == 0.0
    assert corr[2, 2] == 1.0 and np.all(corr[2, [0, 1, 3, 4]] == 0.0)
    assert np.all(np.abs(corr) <= 1.0 + 1e-12)
    assert np.isfinite(cov).all() and _is_psd(cov)
    assert len(degen) == 1 and sorted(degen[0]) == [0, 1]


def test_ill_conditioned_column_scales_match_normalized_problem() -> None:
    x = np.linspace(0.0, 1.0, 100)
    jac = np.column_stack([1e8 * x, 1e-8 * x**2, np.ones_like(x)])
    cov, esd, corr, insens, degen = covariance_from_jacobian(jac, 2.0)
    assert insens == [] and degen == []
    _cov2, esd2, corr2, _, _ = covariance_from_jacobian(
        np.column_stack([x, x**2, np.ones_like(x)]), 2.0
    )
    assert np.allclose(corr, corr2, atol=1e-8)
    assert np.allclose(esd, esd2 / np.array([1e8, 1e-8, 1.0]), rtol=1e-6)
    assert _is_psd(cov)


def test_refine_reports_insensitive_and_degenerate_parameters() -> None:
    x = np.linspace(0.0, 1.0, 40)
    y_obs = 2.0 * x + 1.0
    ps = ParameterSet(
        [
            Parameter("a", 0.0, vary=True),
            Parameter("b", 0.0, vary=True),
            Parameter("c", 0.0, vary=True),
            Parameter("d", 0.0, vary=True),
        ]
    )

    def resid() -> np.ndarray:
        # a と b は和でしか効かない。d は残差に関係しない
        return y_obs - ((ps["a"].value + ps["b"].value) * x + ps["c"].value)

    res = refine_least_squares(ps, resid, max_cycles=10)
    assert res.insensitive == ["d"]
    assert res.esd[res.names.index("d")] == 0.0
    assert [sorted(g) for g in res.degenerate] == [["a", "b"]]
    assert np.all(np.abs(res.correlation) <= 1.0 + 1e-12)
    assert np.isfinite(res.covariance).all()
    text = res.summary()
    assert "感度が無い" in text and "d" in text
    assert "組み合わせ" in text and "a, b" in text
