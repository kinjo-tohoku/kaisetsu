"""Pawley 法: 反射ごとの積分強度を線形最小二乗で求める(構造モデルなしの全パターンフィット)。

Le Bail 法が Rietveld 分配の反復で強度を決めるのに対し、Pawley 法は

    y_calc(x_i) = y_fixed(x_i) + Σ_k a_k p_k(x_i)   (p_k: 面積 1 に規格化した反射 k のプロファイル)

の a_k を、格子定数・プロファイル・背景を固定した各段階で重み付き線形最小二乗(a_k ≥ 0)で解く。
非線形パラメータの精密化では、残差を評価するたびに a_k を解き直す(変数射影)。
d 値が同じで完全に重なる反射(立方晶の 333/511 など)は分離できないので 1 つの未知数にまとめ、
解いた強度を現在の強度比(初期値は構造モデルか一様)で分ける(Le Bail 法と同じ扱い)。
強度の標準偏差は正規方程式の逆行列 (AᵀWA)⁻¹ χ² から得られ、重なった反射の相関も分かる。
MEM の入力(|F_obs|² と σ)に使う。

出典: G. S. Pawley, J. Appl. Cryst. 14, 357-361 (1981);
      A. Le Bail, Powder Diffr. 20, 316-326 (2005)(Le Bail 法との比較)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.optimize import lsq_linear

FloatArray = npt.NDArray[np.float64]


@dataclass
class PawleySolution:
    """反射ごとの積分強度の線形最小二乗解。"""

    intensity: FloatArray  # a_k
    sigma: FloatArray  # σ(a_k)(重なりの相関込み)
    y_bragg: FloatArray  # Σ_k a_k p_k(x)
    chi2: float
    correlation_pairs: list[tuple[int, int, float]]  # 強く相関する反射の組(|ρ| ≥ 0.9)


def solve_pawley(
    y_obs: FloatArray,
    weight: FloatArray,
    y_fixed: FloatArray,
    windows: list[tuple[int, int]],
    profiles: list[FloatArray],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    nonnegative: bool = True,
    dense_limit: int = 20_000_000,
    compute_sigma: bool = True,
    split_weights: FloatArray | None = None,
    overlap_tol: float = 1e-6,
) -> PawleySolution:
    """a_k ≥ 0 を解く。

    Args:
        y_obs, weight: 観測と重み(全点)
        y_fixed: 背景と他の相の寄与(全点)
        windows: 反射ごとの窓 [i0, i1)
        profiles: 反射ごとの面積 1 のプロファイル(窓内)。面積 0 の反射は強度 0・σ = inf
        mask: 使う点(None なら全点)
        nonnegative: 真なら a_k ≥ 0 の制約付き
        dense_limit: 点数 × 反射数がこれ以下なら密行列で解く(大きければ疎行列)
        compute_sigma: σ と相関を計算するか(密行列のとき)
        split_weights: 完全に重なる反射の組に強度を分ける比(None なら等分)
        overlap_tol: プロファイルが同じと見なす相対許容差
    """
    n = len(y_obs)
    m = len(windows)
    use = np.ones(n, dtype=bool) if mask is None else mask
    sw = np.sqrt(np.where(use, weight, 0.0))
    b = sw * (y_obs - y_fixed)
    ok = np.array([len(p) > 0 and float(np.sum(p)) > 0 for p in profiles], dtype=bool)
    intensity = np.zeros(m)
    sigma = np.full(m, np.inf)
    y_bragg = np.zeros(n)
    # 完全に重なる反射をまとめる(列 → 反射の組)
    groups: list[list[int]] = []
    by_window: dict[tuple[int, int], list[int]] = {}
    for k in np.nonzero(ok)[0]:
        key = windows[k]
        placed = False
        for gi in by_window.get(key, []):
            ref = profiles[groups[gi][0]]
            if np.allclose(profiles[k], ref, rtol=overlap_tol, atol=overlap_tol * ref.max()):
                groups[gi].append(int(k))
                placed = True
                break
        if not placed:
            by_window.setdefault(key, []).append(len(groups))
            groups.append([int(k)])
    cols = np.array([g[0] for g in groups], dtype=np.int64)
    if len(cols) == 0:
        chi2 = float(np.sum(b * b) / max(int(np.count_nonzero(use)), 1))
        return PawleySolution(intensity, sigma, y_bragg, chi2, [])
    rows: list[np.ndarray] = []
    cidx: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for j, k in enumerate(cols):
        i0, i1 = windows[k]
        idx = np.arange(i0, i1)
        rows.append(idx)
        cidx.append(np.full(len(idx), j))
        vals.append(sw[i0:i1] * profiles[k])
    r = np.concatenate(rows)
    c = np.concatenate(cidx)
    v = np.concatenate(vals)
    dense = n * len(cols) <= dense_limit
    bounds = (0.0, np.inf) if nonnegative else (-np.inf, np.inf)
    if dense:
        a_mat = np.zeros((n, len(cols)))
        np.add.at(a_mat, (r, c), v)
        res = lsq_linear(a_mat, b, bounds=bounds, method="bvls" if nonnegative else "trf")
        a_sol = np.asarray(res.x, dtype=np.float64)
    else:
        a_sp = sp.csr_matrix((v, (r, c)), shape=(n, len(cols)))
        res = lsq_linear(a_sp, b, bounds=bounds, method="trf", lsmr_tol="auto")
        a_sol = np.asarray(res.x, dtype=np.float64)
        a_mat = None
    for j, g in enumerate(groups):
        if len(g) == 1:
            intensity[g[0]] = a_sol[j]
        else:
            w = (
                np.array([max(float(split_weights[k]), 0.0) for k in g])
                if split_weights is not None
                else np.ones(len(g))
            )
            if w.sum() <= 0:
                w = np.ones(len(g))
            for k, wk in zip(g, w / w.sum(), strict=True):
                intensity[k] = a_sol[j] * wk
    for j, k in enumerate(cols):
        i0, i1 = windows[k]
        y_bragg[i0:i1] += a_sol[j] * profiles[k]
    resid = b - sw * y_bragg
    dof = max(int(np.count_nonzero(use)) - len(cols), 1)
    chi2 = float(np.sum(resid * resid) / dof)
    pairs: list[tuple[int, int, float]] = []
    if compute_sigma:
        if a_mat is not None:
            ata = a_mat.T @ a_mat
        else:
            ata = (a_sp.T @ a_sp).toarray()
        try:
            cov = np.linalg.inv(ata) * chi2
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(ata) * chi2
        d = np.sqrt(np.maximum(np.diag(cov), 0.0))
        for j, g in enumerate(groups):
            for k in g:
                sigma[k] = d[j] * (intensity[k] / a_sol[j] if a_sol[j] > 0 else 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(d, d)
        iu = np.triu_indices(len(cols), 1)
        strong = np.abs(corr[iu]) >= 0.9
        for a_i, b_i in zip(iu[0][strong], iu[1][strong], strict=True):
            pairs.append((int(cols[a_i]), int(cols[b_i]), float(corr[a_i, b_i])))
    return PawleySolution(intensity, sigma, y_bragg, chi2, pairs)
