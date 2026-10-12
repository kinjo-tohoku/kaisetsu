"""最小二乗精密化のドライバ(Levenberg-Marquardt / Trust Region Reflective)。

目的関数は重み付き残差 r_i = sqrt(w_i) (y_obs,i - y_calc,i) の二乗和。
ヤコビアンはパラメータごとの刻み(`Parameter.step`)による前進差分で求める。
収束後、共分散 (J^T J)^{-1} × χ²_red から esd と相関行列を得る。

出典: W. H. Press et al., *Numerical Recipes*, Ch. 15; Young, *The Rietveld Method*, Ch. 1.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from kaisetsu.errors import PatternSizeError
from kaisetsu.refine.parameters import ParameterSet

FloatArray = npt.NDArray[np.float64]


@dataclass
class FitResult:
    """精密化の結果。"""

    names: list[str]
    values: FloatArray
    esd: FloatArray
    covariance: FloatArray
    correlation: FloatArray
    chi2: float
    n_data: int
    n_params: int
    n_evaluations: int
    success: bool
    message: str
    elapsed: float
    history: list[float] = field(default_factory=list)

    def high_correlations(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        out = []
        n = len(self.names)
        for i in range(n):
            for j in range(i + 1, n):
                c = float(self.correlation[i, j])
                if abs(c) >= threshold:
                    out.append((self.names[i], self.names[j], c))
        return sorted(out, key=lambda t: -abs(t[2]))

    def summary(self) -> str:
        lines = [
            f"収束: {'成功' if self.success else '未収束'}({self.message})",
            f"χ² = {self.chi2:.4f}, データ点 {self.n_data}, 変数 {self.n_params}, "
            f"関数評価 {self.n_evaluations} 回, {self.elapsed:.1f} 秒",
        ]
        for n, v, e in zip(self.names, self.values, self.esd, strict=True):
            lines.append(f"  {n:32s} = {v:14.6g} ± {e:.3g}")
        hc = self.high_correlations()
        if hc:
            lines.append("相関の強い組 (|ρ| ≥ 0.9):")
            for a, b, c in hc[:10]:
                lines.append(f"  {a} – {b}: {c:+.3f}")
        return "\n".join(lines)


def numerical_jacobian(
    fun: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    steps: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    base: FloatArray | None = None,
) -> FloatArray:
    """前進差分ヤコビアン。上限に近いパラメータは後退差分にする。"""
    r0 = fun(x) if base is None else base
    jac = np.empty((len(r0), len(x)))
    for k in range(len(x)):
        h = steps[k]
        xk = x.copy()
        if xk[k] + h > upper[k]:
            h = -h
        xk[k] += h
        if xk[k] < lower[k]:
            xk[k] = lower[k]
            h = xk[k] - x[k]
        jac[:, k] = (fun(xk) - r0) / h
    return jac


def refine_least_squares(
    params: ParameterSet,
    residual_fn: Callable[[], FloatArray],
    *,
    max_cycles: int = 20,
    ftol: float = 1e-6,
    xtol: float = 1e-8,
    verbose: int = 0,
    progress: Callable[[int, float], None] | None = None,
    jacobian_fn: Callable[[FloatArray], FloatArray] | None = None,
    n_penalty: int = 0,
) -> FitResult:
    """ParameterSet の精密化変数を最小二乗で最適化する。

    Args:
        params: パラメータ集合(vary の立ったものが変数)
        residual_fn: 現在のパラメータでの重み付き残差ベクトルを返す関数
        max_cycles: 反復(ヤコビアン評価)回数の上限
        ftol, xtol: 収束判定(scipy.optimize.least_squares に渡す)
        verbose: 0, 1, 2
        progress: 各反復後に (反復回数, χ²) で呼ばれる関数
        jacobian_fn: 直近の残差(residual_fn の返り値)を受け取り、現在のパラメータでの
            ヤコビアン (N_resid, N_free) を返す関数。None なら前進差分
        n_penalty: residual_fn の末尾にある罰則行の数。最適化には使うが、χ² とデータ点数
            からは除く
    """
    free = params.free()
    if not free:
        raise ValueError("精密化するパラメータがありません(vary を立ててください)")
    x0 = params.vector()
    lower, upper = params.bounds()
    steps = params.steps()
    x0 = np.clip(x0, lower, upper)
    n_eval = 0
    history: list[float] = []
    t0 = time.perf_counter()
    n_par = len(free)
    state = {"ssq": float("nan"), "n_data": 0}

    def fun(x: FloatArray) -> FloatArray:
        nonlocal n_eval
        n_eval += 1
        params.set_vector(x)
        try:
            r = residual_fn()
        except PatternSizeError:
            # 格子定数や幅が発散した試行点。最初の点なら呼び出し側に知らせ、途中なら
            # 「残差が巨大」として扱い、信頼領域を縮めて別の点を試させる
            if state["n_data"] == 0:
                raise
            r = np.full(int(state["n_data"]) + n_penalty, 1e10)
        if not np.all(np.isfinite(r)):
            r = np.where(np.isfinite(r), r, 1e10)
        rd = r[: len(r) - n_penalty] if n_penalty else r
        state["ssq"] = float(np.dot(rd, rd))
        state["n_data"] = len(rd)
        return r

    def reduced_chi2() -> float:
        dof = max(state["n_data"] - n_par, 1)
        return state["ssq"] / dof

    def jac(x: FloatArray) -> FloatArray:
        base = fun(x)
        if jacobian_fn is not None:
            j = jacobian_fn(base)
            params.set_vector(x)
            return j
        j = numerical_jacobian(fun, x, steps, lower, upper, base)
        params.set_vector(x)  # fun の副作用でずれた値を戻す
        return j

    kwargs: dict[str, Any] = {}
    if "callback" in inspect.signature(least_squares).parameters:

        def _cb(intermediate_result: Any = None, *args: Any) -> None:
            # SciPy のバージョンにより OptimizeResult か x 配列が渡される。
            # どちらでも直近の残差評価(受理された点)から χ²_red を出す。
            if hasattr(intermediate_result, "cost"):
                dof = max(state["n_data"] - n_par, 1)
                chi2 = float(2.0 * intermediate_result.cost) / dof
            else:
                chi2 = reduced_chi2()
            history.append(chi2)
            if progress is not None:
                progress(len(history), chi2)

        kwargs["callback"] = _cb

    try:
        res = least_squares(
            fun,
            x0,
            jac=jac,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=ftol,
            xtol=xtol,
            gtol=1e-10,
            max_nfev=max(3 * max_cycles, 5),
            verbose=verbose,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("callback", None)
        res = least_squares(
            fun, x0, jac=jac, bounds=(lower, upper), method="trf", x_scale="jac",
            ftol=ftol, xtol=xtol, gtol=1e-10, max_nfev=max(3 * max_cycles, 5), verbose=verbose,
        )  # fmt: skip

    params.set_vector(res.x)
    r = residual_fn()
    if n_penalty:
        r = r[: len(r) - n_penalty]
    n_data = len(r)
    n_par = len(res.x)
    dof = max(n_data - n_par, 1)
    chi2 = float(np.sum(r**2) / dof)
    jtj = res.jac.T @ res.jac
    cov = _safe_inverse(jtj) * chi2
    esd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(esd, esd)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    params.set_esd(esd)
    return FitResult(
        names=[p.name for p in free],
        values=np.asarray(res.x, dtype=np.float64),
        esd=esd,
        covariance=cov,
        correlation=corr,
        chi2=chi2,
        n_data=n_data,
        n_params=n_par,
        n_evaluations=n_eval,
        success=bool(res.success),
        message=str(res.message),
        elapsed=time.perf_counter() - t0,
        history=history,
    )


def _safe_inverse(a: FloatArray) -> FloatArray:
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)
