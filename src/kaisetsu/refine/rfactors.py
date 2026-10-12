"""R 因子と適合度。

    R_p   = Σ|y_o - y_c| / Σ y_o
    R_wp  = sqrt(Σ w (y_o - y_c)^2 / Σ w y_o^2)
    R_exp = sqrt((N - P) / Σ w y_o^2)
    χ^2   = (R_wp / R_exp)^2  (GOF = sqrt(χ^2))
    R_B   = Σ|I_o - I_c| / Σ I_o(反射ごとの積分強度、Rietveld 分配による "観測" 強度)
    R_F   = Σ| |F_o| - |F_c| | / Σ |F_o|

背景差し引き版(R_p', R_wp')は分母に y_o - y_b を使う。

出典: Young, *The Rietveld Method*, Ch. 1, Table 1.3; Toby, Powder Diffr. 21, 67-70 (2006)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RFactors:
    rp: float
    rwp: float
    rexp: float
    chi2: float
    rp_bkg: float
    rwp_bkg: float
    n_points: int
    n_params: int

    @property
    def gof(self) -> float:
        return float(np.sqrt(self.chi2))

    def __str__(self) -> str:
        return (
            f"Rp = {100 * self.rp:.3f}%  Rwp = {100 * self.rwp:.3f}%  "
            f"Rexp = {100 * self.rexp:.3f}%  χ² = {self.chi2:.3f}  "
            f"(背景差引: Rp' = {100 * self.rp_bkg:.3f}%  Rwp' = {100 * self.rwp_bkg:.3f}%)"
        )


def powder_rfactors(
    y_obs: npt.ArrayLike,
    y_calc: npt.ArrayLike,
    weight: npt.ArrayLike,
    n_params: int,
    y_bkg: npt.ArrayLike | None = None,
) -> RFactors:
    """粉末パターンの R 因子。重み 0 の点は除外する。"""
    yo = np.asarray(y_obs, dtype=np.float64)
    yc = np.asarray(y_calc, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    use = w > 0
    yo, yc, w = yo[use], yc[use], w[use]
    n = len(yo)
    diff = yo - yc
    sum_wyo2 = float(np.sum(w * yo**2))
    sum_yo = float(np.sum(np.abs(yo)))
    rp = float(np.sum(np.abs(diff)) / sum_yo) if sum_yo > 0 else float("nan")
    rwp = float(np.sqrt(np.sum(w * diff**2) / sum_wyo2)) if sum_wyo2 > 0 else float("nan")
    dof = max(n - n_params, 1)
    rexp = float(np.sqrt(dof / sum_wyo2)) if sum_wyo2 > 0 else float("nan")
    chi2 = float(np.sum(w * diff**2) / dof)
    if y_bkg is not None:
        yb = np.asarray(y_bkg, dtype=np.float64)[use]
        net = yo - yb
        s1 = float(np.sum(np.abs(net)))
        s2 = float(np.sum(w * net**2))
        rp_b = float(np.sum(np.abs(diff)) / s1) if s1 > 0 else float("nan")
        rwp_b = float(np.sqrt(np.sum(w * diff**2) / s2)) if s2 > 0 else float("nan")
    else:
        rp_b = rwp_b = float("nan")
    return RFactors(rp, rwp, rexp, chi2, rp_b, rwp_b, n, n_params)


def bragg_rfactor(i_obs: npt.ArrayLike, i_calc: npt.ArrayLike) -> float:
    """R_Bragg = Σ|I_o - I_c| / Σ I_o。"""
    io = np.asarray(i_obs, dtype=np.float64)
    ic = np.asarray(i_calc, dtype=np.float64)
    s = float(np.sum(np.abs(io)))
    return float(np.sum(np.abs(io - ic)) / s) if s > 0 else float("nan")


def structure_rfactor(f2_obs: npt.ArrayLike, f2_calc: npt.ArrayLike) -> float:
    """R_F = Σ| |F_o| - |F_c| | / Σ |F_o|(|F|^2 から)。"""
    fo = np.sqrt(np.maximum(np.asarray(f2_obs, dtype=np.float64), 0.0))
    fc = np.sqrt(np.maximum(np.asarray(f2_calc, dtype=np.float64), 0.0))
    s = float(np.sum(fo))
    return float(np.sum(np.abs(fo - fc)) / s) if s > 0 else float("nan")


#: 分配に使う点の条件: 反射自身の寄与がその最大値のこの割合以上(擬 Voigt の ±3.5 FWHM 程度)
PARTITION_CORE_FRACTION = 1e-2


def _partition_mask(
    nc: FloatArray, c: FloatArray, floor: float, core_fraction: float
) -> npt.NDArray[np.bool_]:
    """分配に使う点(計算の正味強度が正で、反射自身の寄与が中心部に近い点)。"""
    ok = nc > floor
    if core_fraction > 0 and len(c):
        ok &= c > core_fraction * float(np.max(c))
    return np.asarray(ok)


def partition_intensities(
    y_obs: FloatArray,
    y_calc: FloatArray,
    y_bkg: FloatArray,
    windows: list[tuple[int, int]],
    contributions: list[FloatArray],
    i_calc: FloatArray,
    core_fraction: float = PARTITION_CORE_FRACTION,
) -> FloatArray:
    """Rietveld 分配による "観測" 積分強度(Hill & Howard 1987 / Rietveld 1969)。

        I_obs,k = I_calc,k Σ_i [ (y_obs,i - y_bkg,i) × c_k(i) / Σ_k' c_k'(i) ]
                = I_calc,k Σ_i (y_obs,i - y_bkg,i) × c_k(i) / (y_calc,i - y_bkg,i)

    c_k(i) は反射 k の点 i への寄与(I_calc,k × プロファイル)。

    和は反射自身の寄与が最大値の `core_fraction` 以上の点(擬 Voigt なら中心から
    ±3.5 FWHM ほど)に限り、Σ c_k もその範囲で取る。ピークの計算窓(低角側 50 幅、
    高角側 75 幅)の裾まで含めると、孤立した反射では c_k / Σ c_k' ≈ 1 なので窓の中の
    背景の誤差がそのまま強度に乗る。放射光データ(FWHM 0.025°、窓 4°)の Le Bail で
    弱い反射が数倍に膨らみ Rwp が Rietveld より悪くなった(opXRD 794: 12.6% → 6.8%)。
    core_fraction = 0 で従来どおり窓全体を使う。

    Args:
        windows: 反射ごとの (開始, 終了) 添字
        contributions: 反射ごとの寄与配列(窓内)
        i_calc: 反射ごとの計算積分強度(窓内寄与の面積)
        core_fraction: 分配に使う点の、反射自身の寄与の下限(最大値に対する割合)
    """
    net_obs = y_obs - y_bkg
    net_calc = y_calc - y_bkg
    out = np.zeros(len(windows))
    floor = 1e-12 * max(float(np.max(np.abs(net_calc))), 1e-300)
    for k, ((i0, i1), c) in enumerate(zip(windows, contributions, strict=True)):
        nc = net_calc[i0:i1]
        ok = _partition_mask(nc, c, floor, core_fraction)
        total = float(np.sum(c[ok]))
        if total <= 0:
            # 使える点がない(窓がデータの外、または計算強度 0)なら計算値のまま
            out[k] = i_calc[k] if len(c) and not ok.any() and np.sum(c) > 0 else 0.0
            continue
        frac = float(np.sum(c[ok] * net_obs[i0:i1][ok] / nc[ok]))
        out[k] = i_calc[k] * frac / total
    return out


def partition_uncertainties(
    sigma_y: FloatArray,
    y_calc: FloatArray,
    y_bkg: FloatArray,
    windows: list[tuple[int, int]],
    contributions: list[FloatArray],
    i_calc: FloatArray,
    core_fraction: float = PARTITION_CORE_FRACTION,
) -> FloatArray:
    """Rietveld 分配した "観測" 積分強度の標準偏差。

    I_obs,k = I_calc,k Σ_i (y_obs,i - y_bkg,i) c_k(i) / (Σ_k' c_k'(i) Σ_i c_k(i)) の
    y_obs,i についての誤差伝播:
        σ²(I_obs,k) = I_calc,k² Σ_i [c_k(i) / ((y_calc,i - y_bkg,i) Σ c_k)]² σ_i²
    重なった反射では分配の仮定(モデルの強度比)による誤差は含まれない。

    Args:
        sigma_y: 観測点の標準偏差(重み 0 の点は inf でよい)
        core_fraction: `partition_intensities` と同じ(同じ点を使う)
    """
    net_calc = y_calc - y_bkg
    out = np.zeros(len(windows))
    floor = 1e-12 * max(float(np.max(np.abs(net_calc))), 1e-300)
    for k, ((i0, i1), c) in enumerate(zip(windows, contributions, strict=True)):
        nc = net_calc[i0:i1]
        ok = _partition_mask(nc, c, floor, core_fraction)
        total = float(np.sum(c[ok]))
        if total <= 0:
            continue
        s = sigma_y[i0:i1]
        finite = ok & np.isfinite(s)
        coef = np.zeros_like(nc)
        coef[finite] = c[finite] / (nc[finite] * total)
        out[k] = i_calc[k] * float(np.sqrt(np.sum((coef[finite] * s[finite]) ** 2)))
    return out
