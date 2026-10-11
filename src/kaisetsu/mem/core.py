"""最大エントロピー法(MEM)による電子密度・核密度の再構成。

単位胞を n1 × n2 × n3 の格子点に分け、各点の「質量」m_i(電子数、中性子では fm)を未知数とする。
構造因子は F_c(h) = Σ_i m_i exp(2πi h·x_i)(FFT で計算)。

情報エントロピー   S = Σ_i [m_i - τ_i - m_i ln(m_i / τ_i)]   (τ_i: 事前分布。既定は一様。
                   Skilling の形。総量が固定なら -Σ m ln(m/τ) と定数差)
F 拘束            C = (1/N_F) Σ_h |F_obs(h) - F_c(h)|² / σ²(h)
F000 の規格化は (000) 反射を小さな σ で観測値に含めて扱う。

解法:
    - "lbfgs"(既定): Q = -S + (λ/2) N_F C を L-BFGS で最小化し、C = 1 になるよう λ を外側で
      割線法・二分法で探索する(Dysnomia と同じく制限記憶 BFGS)。変数は v = 2 sqrt(m)
      (m = v²/4 ≥ 0 が自動で満たされ、エントロピーの 2 階微分 (1/2) ln(m/τ) + 1 が
      m の 1/m より条件数が良い)。
    - "zspa": Sakata-Sato の 0 次単一ピクセル近似の反復
      m_i' = τ_i exp[(λ F000 / N_F) Σ_h (F_obs - F_c) / σ² exp(-2πi h·x_i)]
      (事前分布は前の反復の密度)。
      教科書どおりの方法で、lbfgs の検算用。

対称性: 独立反射を全対称操作で展開して格子に置く。操作 x' = R x + t に対し
F(hR) = F(h) exp(-2πi h·t)、Friedel 対 F(-h) = F(h)*。
事前分布が一様で観測が対称なら密度も対称になる。

負の散乱長(中性子)には 2 チャンネル MEM(m = m⁺ - m⁻、S = S⁺ + S⁻)を用意する。
m⁺ + m⁻ は観測に影響しないので S を最大にする値を解析的に取る(m⁺ m⁻ = τ⁺ τ⁻)。
すると d = m⁺ - m⁻ だけが変数になり、ψ = sqrt(d² + 4 τ⁺ τ⁻)、m± = (ψ ± d)/2、
S(d) = Σ [ψ - τ⁺ - τ⁻ - m⁺ ln(m⁺/τ⁺) - m⁻ ln(m⁻/τ⁻)]、dS/dd = -ln(m⁺/τ⁺)
(Gull & Skilling の正負エントロピー)。制約なしの L-BFGS で速く収束する。

出典: M. Sakata & M. Sato, Acta Cryst. A46, 263 (1990); D. M. Collins, Nature 298, 49 (1982);
      S. Kumazawa, Y. Kubota, M. Takata, M. Sakata & Y. Ishibashi, J. Appl. Cryst. 26, 453 (1993);
      S. F. Gull & J. Skilling, *MEMSYS5 Users' Manual* (1990)(正負エントロピー);
      K. Momma, T. Ikeda, A. A. Belik & F. Izumi, Powder Diffr. 28, 184 (2013)(Dysnomia)。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass
class MEMInput:
    """MEM の入力。

    Attributes:
        lattice, spacegroup: 単位胞と空間群
        hkl: 独立反射 (n, 3)
        f_obs: 観測構造因子(位相付き、複素数)(n,)
        sigma: |F_obs| の標準偏差 (n,)
        f000: F(000) = 単位胞の全電子数(中性子では散乱長の和 [fm])
        grid: 格子点数 (n1, n2, n3)。各軸で 2 max|h| + 1 以上
        sigma_f000: F000 の σ(既定は f000 × 1e-4)
        two_channel: 2 チャンネル MEM(負の散乱長用)
        f000_negative: 2 チャンネルのとき、負側の総量 |Σ b_neg|(正側は f000 + f000_negative)
    """

    lattice: Lattice
    spacegroup: SpaceGroup
    hkl: IntArray
    f_obs: ComplexArray
    sigma: FloatArray
    f000: float
    grid: tuple[int, int, int] = (64, 64, 64)
    sigma_f000: float | None = None
    two_channel: bool = False
    f000_negative: float = 0.0

    def __post_init__(self) -> None:
        self.hkl = np.atleast_2d(np.asarray(self.hkl, dtype=np.int64))
        self.f_obs = np.asarray(self.f_obs, dtype=np.complex128).ravel()
        self.sigma = np.asarray(self.sigma, dtype=np.float64).ravel()
        if not (len(self.hkl) == len(self.f_obs) == len(self.sigma)):
            raise ValueError("hkl, f_obs, sigma の長さが一致しません")
        if np.any(self.sigma <= 0):
            raise ValueError("σ は正でなければなりません")
        hmax = np.abs(self.hkl).max(axis=0) if len(self.hkl) else np.zeros(3, dtype=int)
        for k in range(3):
            if self.grid[k] < 2 * int(hmax[k]) + 1:
                raise ValueError(
                    f"格子点数 {self.grid} が粗すぎます(軸 {k}: 2×{int(hmax[k])}+1 以上が必要)"
                )


@dataclass
class MEMResult:
    """MEM の結果。density は e/Å³(中性子では fm/Å³)。"""

    density: FloatArray
    lattice: Lattice
    spacegroup: SpaceGroup
    chi2: float
    entropy: float
    lam: float
    n_iter: int
    hkl: IntArray
    f_obs: ComplexArray
    f_calc: ComplexArray
    history: list[tuple[float, float]] = field(default_factory=list)
    elapsed: float = 0.0
    message: str = ""
    density_negative: FloatArray | None = None

    @property
    def grid(self) -> tuple[int, int, int]:
        n = self.density.shape
        return (int(n[0]), int(n[1]), int(n[2]))

    def r_factor(self) -> float:
        """R_F = Σ||F_obs| - |F_calc|| / Σ|F_obs|(F000 を除く)。"""
        fo = np.abs(self.f_obs)
        fc = np.abs(self.f_calc)
        s = float(np.sum(fo))
        return float(np.sum(np.abs(fo - fc)) / s) if s > 0 else float("nan")

    def fractional_coordinates(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        n1, n2, n3 = self.grid
        return (np.arange(n1) / n1, np.arange(n2) / n2, np.arange(n3) / n3)

    def maxima(self, n: int = 10, min_separation: float = 0.5) -> list[tuple[FloatArray, float]]:
        """密度の極大(分率座標, 値)を大きい順に返す。min_separation [Å] より近いものはまとめる。"""
        rho = self.density
        n1, n2, n3 = self.grid
        order = np.argsort(rho.ravel())[::-1]
        out: list[tuple[FloatArray, float]] = []
        for flat in order:
            if len(out) >= n:
                break
            i, j, k = np.unravel_index(int(flat), rho.shape)
            x = np.array([i / n1, j / n2, k / n3])
            if rho[i, j, k] <= 0:
                break
            close = False
            for xo, _ in out:
                d = x - xo
                d -= np.round(d)
                if float(self.lattice.distance(np.zeros(3), d)) < min_separation:
                    close = True
                    break
            if not close:
                out.append((x, float(rho[i, j, k])))
        return out

    def integrate_sphere(self, center: npt.ArrayLike, radius: float) -> float:
        """中心 center(分率)・半径 radius [Å] の球内の積分値(電子数など)。"""
        n1, n2, n3 = self.grid
        c = np.asarray(center, dtype=np.float64).reshape(3)
        i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
        frac = np.stack([i / n1, j / n2, k / n3], axis=-1).reshape(-1, 3) - c
        frac -= np.round(frac)
        cart = self.lattice.cartesian(frac)
        inside = np.einsum("ij,ij->i", cart, cart) <= radius * radius
        voxel = self.lattice.volume / (n1 * n2 * n3)
        return float(np.sum(self.density.ravel()[inside]) * voxel)


# ----------------------------------------------------------------------
# 反射の展開
# ----------------------------------------------------------------------
def expand_reflections(
    spacegroup: SpaceGroup,
    hkl: IntArray,
    values: ComplexArray,
    grid: tuple[int, int, int],
    tol: float = 1e-3,
) -> tuple[IntArray, ComplexArray, IntArray]:
    """独立反射を全等価反射(Friedel 対込み)に展開し、FFT 格子上の平坦な添字に写す。

    Returns:
        flat_index: 各等価反射の格子添字 (M,)
        values_eq: 対応する F (M,)
        owner: どの独立反射から来たか (M,)
    """
    rots = spacegroup.rotations
    trans = spacegroup.translations
    n1, n2, n3 = grid
    idx_list: list[int] = []
    val_list: list[complex] = []
    owner_list: list[int] = []
    for r, (h, f) in enumerate(zip(hkl, values, strict=True)):
        heq = np.einsum("j,mjk->mk", h, rots)  # h R
        ph = np.exp(-2j * math.pi * (trans @ h))  # exp(-2πi h·t)
        feq = f * ph
        allh = np.concatenate([heq, -heq])
        allf = np.concatenate([feq, np.conj(feq)])
        seen: dict[tuple[int, int, int], complex] = {}
        for hh, ff in zip(allh, allf, strict=True):
            key = (int(hh[0]), int(hh[1]), int(hh[2]))
            prev = seen.get(key)
            if prev is None:
                seen[key] = complex(ff)
            elif abs(prev - ff) > tol * max(abs(ff), 1.0):
                raise ValueError(
                    f"反射 {tuple(int(v) for v in h)} の位相が対称性と矛盾しています"
                    f"({key}: {prev:.3g} と {ff:.3g})"
                )
        for (a, b, c), ff in seen.items():
            idx_list.append(((a % n1) * n2 + (b % n2)) * n3 + (c % n3))
            val_list.append(ff)
            owner_list.append(r)
    return (
        np.asarray(idx_list, dtype=np.int64),
        np.asarray(val_list, dtype=np.complex128),
        np.asarray(owner_list, dtype=np.int64),
    )


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
class _Model:
    """格子・観測の前処理と、目的関数・勾配の計算。"""

    def __init__(self, inp: MEMInput) -> None:
        self.inp = inp
        self.grid = inp.grid
        self.n = int(np.prod(inp.grid))
        hkl = np.vstack([np.zeros((1, 3), dtype=np.int64), inp.hkl])
        f = np.concatenate([[complex(inp.f000)], inp.f_obs])
        s000 = inp.sigma_f000 if inp.sigma_f000 is not None else max(abs(inp.f000) * 1e-4, 1e-6)
        sig = np.concatenate([[s000], inp.sigma])
        idx, vals, owner = expand_reflections(inp.spacegroup, hkl, f, inp.grid)
        counts = np.bincount(owner, minlength=len(hkl)).astype(np.float64)
        self.idx = idx
        self.f_obs_eq = vals
        self.owner = owner
        self.n_refl = len(hkl)  # N_F(F000 込み)
        # 独立反射ごとの重み 1/σ² を等価反射の数で割って配る
        self.w_eq = (1.0 / sig[owner] ** 2) / counts[owner]
        self.hkl_all = hkl
        self.owner_first = np.array(
            [int(np.nonzero(owner == r)[0][0]) for r in range(len(hkl))], dtype=np.int64
        )

    def f_calc_eq(self, m: FloatArray) -> ComplexArray:
        f = self.n * np.fft.ifftn(m.reshape(self.grid))
        return np.asarray(f.ravel()[self.idx])

    def chi2(self, fc: ComplexArray) -> float:
        d = self.f_obs_eq - fc
        return float(np.sum(self.w_eq * (d.real**2 + d.imag**2)) / self.n_refl)

    def data_gradient(self, fc: ComplexArray) -> FloatArray:
        """∂/∂m_i Σ_h w_h |F_o - F_c|² = -2 N Re[ifftn(A)]_i, A[h] = w_h (F_o - F_c)*。"""
        coef = self.w_eq * np.conj(self.f_obs_eq - fc)
        a = np.bincount(self.idx, weights=coef.real, minlength=self.n) + 1j * np.bincount(
            self.idx, weights=coef.imag, minlength=self.n
        )
        g = np.fft.ifftn(a.reshape(self.grid)).real.ravel()
        return np.asarray(-2.0 * self.n * g)

    def f_calc_independent(self, m: FloatArray) -> ComplexArray:
        fc = self.f_calc_eq(m)
        return np.asarray(fc[self.owner_first])


def _entropy(m: FloatArray, tau: FloatArray) -> tuple[float, FloatArray]:
    """S = Σ [m - τ - m ln(m/τ)] と ∂S/∂m = -ln(m/τ)。"""
    safe = np.maximum(m, 1e-300)
    ln = np.log(safe / tau)
    return float(np.sum(m - tau - m * ln)), -ln


def _entropy_two_channel(
    d: FloatArray, tau_p: FloatArray, tau_n: FloatArray
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """正負エントロピー S(d)、∂S/∂d、m⁺、m⁻(m⁺ m⁻ = τ⁺ τ⁻ で m⁺ + m⁻ を消去済み)。"""
    psi = np.sqrt(d * d + 4.0 * tau_p * tau_n)
    mp = 0.5 * (psi + d)
    mn = 0.5 * (psi - d)
    mp_safe = np.maximum(mp, 1e-300)
    mn_safe = np.maximum(mn, 1e-300)
    ln_p = np.log(mp_safe / tau_p)
    ln_n = np.log(mn_safe / tau_n)
    s_val = float(np.sum(psi - tau_p - tau_n - mp * ln_p - mn * ln_n))
    return s_val, -ln_p, mp, mn


def run_mem(
    inp: MEMInput,
    *,
    method: str = "lbfgs",
    target_chi2: float = 1.0,
    lam0: float | None = None,
    max_outer: int = 25,
    max_inner: int = 2000,
    chi2_tol: float = 0.02,
    prior: FloatArray | None = None,
    zspa_lambda: float = 0.05,
    zspa_max_iter: int = 3000,
    parametrization: str = "sqrt",
    progress: Callable[[int, float, float], None] | None = None,
) -> MEMResult:
    """MEM を実行する。

    Args:
        inp: 入力
        method: 'lbfgs' か 'zspa'
        target_chi2: 目標の C(通常 1)
        lam0: lbfgs の λ の初期値(None なら自動)
        max_outer: λ の探索回数の上限
        max_inner: 1 回の λ での L-BFGS 反復回数の上限。目標から遠い λ では 1/5 で打ち切り、
            近づいたら収束するまで続ける(足りないと C が λ に対して単調にならず探索が迷走する)
        chi2_tol: |C - target| / target の収束判定
        prior: 事前分布(格子と同じ形、正)。None なら一様
        zspa_lambda, zspa_max_iter: zspa 法の刻みと反復上限
        parametrization: 1 チャンネルの変数。'sqrt'(既定、v = 2 sqrt(m))、'bounded'(m に下限)、
            'log'(u = ln m)。2 チャンネルでは無視される(d = m⁺ - m⁻ を制約なしで最適化する)
        progress: (外側反復, λ, C) で呼ばれる
    """
    t0 = time.perf_counter()
    model = _Model(inp)
    n = model.n
    two = inp.two_channel
    f_pos = inp.f000 + inp.f000_negative if two else inp.f000
    if f_pos <= 0:
        raise ValueError("F000(正側)は正でなければなりません")
    if prior is None:
        tau_pos = np.full(n, f_pos / n)
    else:
        p = np.asarray(prior, dtype=np.float64).ravel()
        if p.shape != (n,):
            raise ValueError("prior の形が格子と一致しません")
        tau_pos = np.maximum(p, 1e-12) * (f_pos / float(np.sum(np.maximum(p, 1e-12))))
    tau_neg = np.full(n, max(inp.f000_negative, 1e-12) / n) if two else None

    history: list[tuple[float, float]] = []
    voxel = inp.lattice.volume / n

    if method == "zspa":
        if two:
            raise ValueError(
                "zspa 法は 1 チャンネルのみです(2 チャンネルは lbfgs を使ってください)"
            )
        m, chi2, n_iter, msg = _run_zspa(
            model, tau_pos, zspa_lambda, zspa_max_iter, target_chi2, history, progress
        )
        lam = zspa_lambda
        s_val, _ = _entropy(m, tau_pos)
        fc = model.f_calc_independent(m)
        return MEMResult(
            density=(m / voxel).reshape(inp.grid),
            lattice=inp.lattice,
            spacegroup=inp.spacegroup,
            chi2=chi2,
            entropy=s_val,
            lam=lam,
            n_iter=n_iter,
            hkl=inp.hkl,
            f_obs=inp.f_obs,
            f_calc=fc[1:],
            history=history,
            elapsed=time.perf_counter() - t0,
            message=msg,
        )
    if method != "lbfgs":
        raise ValueError("method は 'lbfgs' か 'zspa' です")

    if parametrization not in ("sqrt", "bounded", "log"):
        raise ValueError("parametrization は 'sqrt', 'bounded', 'log' のいずれかです")
    use_log = parametrization == "log" and not two
    use_sqrt = parametrization == "sqrt" and not two

    def to_m(x: FloatArray) -> FloatArray:
        if use_log:
            return np.exp(x)
        if use_sqrt:
            return 0.25 * x * x
        return x

    def objective(x: FloatArray, lam: float) -> tuple[float, FloatArray]:
        if two:
            assert tau_neg is not None
            s_val, gs, _mp, _mn = _entropy_two_channel(x, tau_pos, tau_neg)
            m_eff = x
        else:
            m_eff = to_m(x)
            s_val, gs = _entropy(m_eff, tau_pos)
        fc = model.f_calc_eq(m_eff)
        d = model.f_obs_eq - fc
        data = float(np.sum(model.w_eq * (d.real**2 + d.imag**2)))
        gd = model.data_gradient(fc)
        val = -s_val + 0.5 * lam * data
        grad = -gs + 0.5 * lam * gd
        if use_log:
            grad = grad * m_eff  # ∂Q/∂u = m ∂Q/∂m
        elif use_sqrt:
            grad = grad * 0.5 * x  # ∂Q/∂v = (v/2) ∂Q/∂m
        return val, grad

    if two:
        assert tau_neg is not None
        x = tau_pos - tau_neg
        bounds = None
    elif use_log:
        x = np.log(tau_pos)
        bounds = None
    elif use_sqrt:
        x = 2.0 * np.sqrt(tau_pos)
        bounds = None
    else:
        x = tau_pos.copy()
        bounds = [(1e-12 * f_pos / n, None)] * n
    if lam0 is None:
        fo2 = float(np.sum(model.w_eq * np.abs(model.f_obs_eq) ** 2))
        lam0 = 2.0 * f_pos / max(fo2, 1e-30)
    lam = lam0
    lo: float | None = None  # C > target となった最大の λ
    hi: float | None = None  # C < target となった最小の λ
    pts: list[tuple[float, float]] = []  # (log λ, log C)
    chi2 = float("nan")
    n_iter = 0
    msg = ""
    near_target = False
    for outer in range(max_outer):
        used = 0
        budget = max_inner if near_target else max(max_inner // 5, 100)
        while True:
            res: Any = minimize(
                objective,
                x,
                args=(lam,),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": max(budget - used, 50),
                    "ftol": 1e-13,
                    "gtol": 1e-8,
                    "maxcor": 15,
                },
            )
            x = np.asarray(res.x, dtype=np.float64)
            used += int(res.nit)
            # status 1 = 反復上限。収束(0)か予算到達まで温かい初期値で続ける
            if int(res.status) != 1 or used >= budget:
                break
        n_iter += used
        m_eff = x if two else to_m(x)
        chi2 = model.chi2(model.f_calc_eq(m_eff))
        history.append((lam, chi2))
        if progress is not None:
            progress(outer + 1, lam, chi2)
        if abs(chi2 - target_chi2) <= chi2_tol * target_chi2 and near_target:
            msg = f"収束: C = {chi2:.4f}(λ = {lam:.3g}、外側反復 {outer + 1} 回)"
            break
        near_target = abs(math.log(max(chi2, 1e-300) / target_chi2)) < math.log(3.0)
        if chi2 > target_chi2:
            lo = lam if lo is None else max(lo, lam)
        else:
            hi = lam if hi is None else min(hi, lam)
        pts.append((math.log(lam), math.log(max(chi2, 1e-300))))
        lam = _next_lambda(pts, lo, hi, target_chi2)
    else:
        msg = f"λ の探索が上限に達しました: C = {chi2:.4f}(目標 {target_chi2})"
    density_negative: FloatArray | None = None
    if two:
        assert tau_neg is not None
        s_val, _, _mp, mn = _entropy_two_channel(x, tau_pos, tau_neg)
        m_eff = x
        density_negative = (mn / voxel).reshape(inp.grid)
    else:
        m_eff = to_m(x)
        s_val, _ = _entropy(m_eff, tau_pos)
    fc = model.f_calc_independent(m_eff)
    return MEMResult(
        density=(m_eff / voxel).reshape(inp.grid),
        lattice=inp.lattice,
        spacegroup=inp.spacegroup,
        chi2=chi2,
        entropy=s_val,
        lam=lam,
        n_iter=n_iter,
        hkl=inp.hkl,
        f_obs=inp.f_obs,
        f_calc=fc[1:],
        history=history,
        elapsed=time.perf_counter() - t0,
        message=msg,
        density_negative=density_negative,
    )


def _next_lambda(
    pts: list[tuple[float, float]], lo: float | None, hi: float | None, target: float
) -> float:
    """次の λ。log C は log λ についてほぼ直線なので割線法で外挿し、挟めたら二分する。"""
    log_t = math.log(target)
    lam_last = math.exp(pts[-1][0])
    if lo is not None and hi is not None:
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            if abs(x2 - x1) > 1e-12 and (y2 - y1) != 0.0:
                xs = x2 + (log_t - y2) * (x2 - x1) / (y2 - y1)
                a, b = math.log(lo), math.log(hi)
                # 挟んだ区間の内側(端から 10% 以上離れる)なら割線の点を使う
                if a + 0.1 * (b - a) < xs < b - 0.1 * (b - a):
                    return math.exp(xs)
        return math.sqrt(lo * hi)
    if len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        if abs(x2 - x1) > 1e-12 and (y2 - y1) < 0.0:
            xs = x2 + (log_t - y2) * (x2 - x1) / (y2 - y1)
            step = max(min(xs - x2, math.log(100.0)), -math.log(100.0))
            return lam_last * math.exp(step)
    return lam_last * 10.0 if lo is not None else lam_last / 10.0


def _run_zspa(
    model: _Model,
    tau: FloatArray,
    lam: float,
    max_iter: int,
    target: float,
    history: list[tuple[float, float]],
    progress: Callable[[int, float, float], None] | None,
) -> tuple[FloatArray, float, int, str]:
    """Sakata-Sato の 0 次単一ピクセル近似。事前分布は前の反復の密度。"""
    f000 = float(np.sum(tau))
    m = tau.copy()
    fc = model.f_calc_eq(m)
    chi2 = model.chi2(fc)
    history.append((lam, chi2))
    for it in range(1, max_iter + 1):
        # Σ_h w_h (F_o - F_c) e^{-2πi h x} の実部 = -(1/2) × data_gradient
        corr = -0.5 * model.data_gradient(fc)
        expo = (lam * f000 / model.n_refl) * corr
        expo -= expo.max()  # オーバーフロー防止(規格化で消える)
        m_new = m * np.exp(expo)
        m_new *= f000 / float(np.sum(m_new))
        fc_new = model.f_calc_eq(m_new)
        chi2_new = model.chi2(fc_new)
        if chi2_new > chi2:
            lam *= 0.5  # 行き過ぎたら刻みを半分に
            if lam < 1e-8:
                return m, chi2, it, "刻みが小さくなりすぎました"
            continue
        m, fc, chi2 = m_new, fc_new, chi2_new
        history.append((lam, chi2))
        if progress is not None and it % 50 == 0:
            progress(it, lam, chi2)
        if chi2 <= target:
            return m, chi2, it, f"収束: C = {chi2:.4f}(反復 {it} 回)"
    return m, chi2, max_iter, f"反復上限: C = {chi2:.4f}"
