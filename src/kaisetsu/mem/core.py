"""最大エントロピー法(MEM)による電子密度・核密度の再構成。

単位胞を n1 × n2 × n3 の格子点に分け、各点の「質量」m_i(電子数、中性子では fm)を未知数とする。
構造因子は F_c(h) = Σ_i m_i exp(2πi h·x_i)(FFT で計算)。

情報エントロピー   S = Σ_i [m_i - τ_i - m_i ln(m_i / τ_i)]   (τ_i: 事前分布。既定は一様。
                   Skilling の形。総量が固定なら -Σ m ln(m/τ) と定数差)
F 拘束            C_F = (1/N_F) Σ_h |F_obs(h) - F_c(h)|² / σ²(h)
G 拘束(任意)    重なって分離できない反射の群 g について、群の強度和だけを拘束する
                   C_G = (1/N_G) Σ_g (G_obs,g - Σ_{h∈g} m_h |F_c(h)|²)² / σ_g²   (m_h: 多重度)
                   全体の C = (N_F C_F + N_G C_G) / (N_F + N_G)。Dysnomia の G 拘束と同じ考え方
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
        sigma: |F_obs| の標準偏差 (n,)。inf なら拘束に使わず N_F にも数えない
            (測定範囲外の反射など。値は f_obs に残す)
        f000: F(000) = 単位胞の全電子数(中性子では散乱長の和 [fm])
        grid: 格子点数 (n1, n2, n3)。各軸で 2 max|h| + 1 以上(全等価反射で見る)
        sigma_f000: F000 の σ(既定は f000 × 1e-4)
        two_channel: 2 チャンネル MEM(負の散乱長用)。None なら f000_negative > 0 のとき真
        f000_negative: 2 チャンネルのとき、負側の総量 |Σ b_neg|(正側は f000 + f000_negative)
        overlap_groups: G 拘束の群のリスト。各要素は (hkl の添字の配列, G_obs, σ_G, 多重度の配列)。
            群に入った反射は F 拘束から外れる(位相は f_obs のものを初期の参考にしない)
    """

    lattice: Lattice
    spacegroup: SpaceGroup
    hkl: IntArray
    f_obs: ComplexArray
    sigma: FloatArray
    f000: float
    grid: tuple[int, int, int] = (64, 64, 64)
    sigma_f000: float | None = None
    two_channel: bool | None = None
    f000_negative: float = 0.0
    overlap_groups: list[tuple[IntArray, float, float, IntArray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hkl = np.atleast_2d(np.asarray(self.hkl, dtype=np.int64))
        self.f_obs = np.asarray(self.f_obs, dtype=np.complex128).ravel()
        self.sigma = np.asarray(self.sigma, dtype=np.float64).ravel()
        if not (len(self.hkl) == len(self.f_obs) == len(self.sigma)):
            raise ValueError("hkl, f_obs, sigma の長さが一致しません")
        if np.any(self.sigma <= 0) or np.any(np.isnan(self.sigma)):
            raise ValueError("σ は正でなければなりません")
        if self.two_channel is None:
            self.two_channel = self.f000_negative > 0
        # 格子は全等価反射(h R)を折り返さずに置ける大きさが必要。独立反射の |h| だけでは、
        # 三方晶・六方晶などで等価反射の成分(−h−k など)が大きくなる分を見落とす
        if len(self.hkl):
            heq = np.einsum("nj,mjk->nmk", self.hkl, self.spacegroup.rotations)
            hmax = np.abs(heq).reshape(-1, 3).max(axis=0)
        else:
            hmax = np.zeros(3, dtype=int)
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
    #: history の各 (λ, C) について、内側の最小化が収束した(L-BFGS の収束か、C が落ち着いた)か。
    #: 偽なら反復予算で打ち切った途中の値(lbfgs のみ)
    history_converged: list[bool] = field(default_factory=list)

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
        if len(np.unique(idx)) != len(idx):
            raise ValueError(
                f"格子点数 {inp.grid} が粗すぎます(等価反射が同じ格子点に折り返されます)"
            )
        counts = np.bincount(owner, minlength=len(hkl)).astype(np.float64)
        self.idx = idx
        self.f_obs_eq = vals
        self.owner = owner
        self.n_refl = len(hkl)  # 反射数(F000 込み)
        # 独立反射ごとの重み 1/σ² を等価反射の数で割って配る(σ = inf は重み 0)
        with np.errstate(divide="ignore"):
            self.w_eq = (1.0 / sig[owner] ** 2) / counts[owner]
        constrained = np.isfinite(sig)
        self.hkl_all = hkl
        self.owner_first = np.array(
            [int(np.nonzero(owner == r)[0][0]) for r in range(len(hkl))], dtype=np.int64
        )
        self.counts = counts
        # G 拘束: 群に入った反射は F 拘束の重みを 0 にする(添字は F000 の分 1 ずれる)
        self.groups: list[tuple[IntArray, float, float, FloatArray]] = []
        for idx_r, g_obs, sig_g, mult in inp.overlap_groups:
            r_idx = np.asarray(idx_r, dtype=np.int64) + 1
            self.w_eq[np.isin(owner, r_idx)] = 0.0
            constrained[r_idx] = False
            self.groups.append(
                (r_idx, float(g_obs), float(sig_g), np.asarray(mult, dtype=np.float64))
            )
        self.n_groups = len(self.groups)
        # N_F = F 拘束に実際に入る反射の数(σ = inf と G 拘束の群に入った反射は除く)
        self.n_f = int(np.count_nonzero(constrained))

    def group_terms(self, fc_eq: ComplexArray) -> tuple[float, list[float]]:
        """Σ_g w_g r_g² と各群の残差 r_g = G_obs - Σ m_h |F_c|²。

        |F_c|² は等価反射の平均(密度が対称なら全部同じ値)。勾配と厳密に整合させるため。
        """
        total = 0.0
        resid: list[float] = []
        f2_all = fc_eq.real**2 + fc_eq.imag**2
        for r_idx, g_obs, sig_g, mult in self.groups:
            f2 = np.array([float(np.mean(f2_all[self.owner == r])) for r in r_idx])
            r_g = g_obs - float(np.sum(mult * f2))
            resid.append(r_g)
            total += (r_g / sig_g) ** 2
        return total, resid

    def group_gradient(self, fc_eq: ComplexArray, resid: list[float]) -> FloatArray:
        """∂/∂m_i Σ_g w_g r_g² = -4 N Re[ifftn(A2)]_i、A2[h] = (w_g r_g m_h / count_h) F_c(h)*。"""
        if not self.groups:
            return np.zeros(self.n)
        coef = np.zeros(len(self.idx), dtype=np.complex128)
        for (r_idx, _g_obs, sig_g, mult), r_g in zip(self.groups, resid, strict=True):
            w_g = 1.0 / (sig_g * sig_g)
            for r, m_h in zip(r_idx, mult, strict=True):
                sel = self.owner == r
                coef[sel] += (w_g * r_g * m_h / self.counts[r]) * np.conj(fc_eq[sel])
        a = np.bincount(self.idx, weights=coef.real, minlength=self.n) + 1j * np.bincount(
            self.idx, weights=coef.imag, minlength=self.n
        )
        g = np.fft.ifftn(a.reshape(self.grid)).real.ravel()
        return np.asarray(-4.0 * self.n * g)

    def chi2_total(self, fc_eq: ComplexArray) -> float:
        """C = (N_F C_F + N_G C_G) / (N_F + N_G)。"""
        d = self.f_obs_eq - fc_eq
        c_f = float(np.sum(self.w_eq * (d.real**2 + d.imag**2)))
        c_g, _ = self.group_terms(fc_eq)
        return (c_f + c_g) / max(self.n_f + self.n_groups, 1)

    def f_calc_eq(self, m: FloatArray) -> ComplexArray:
        f = self.n * np.fft.ifftn(m.reshape(self.grid))
        return np.asarray(f.ravel()[self.idx])

    def chi2(self, fc: ComplexArray) -> float:
        return self.chi2_total(fc)

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
        max_inner: 1 回の λ での L-BFGS 反復回数の目安。目標から遠い λ では 1/5 で打ち切り、
            近づいたら(目標の 1/3〜3 倍)C が落ち着くまで続ける(上限はこの 8 倍)。
            打ち切った C は λ に対して単調でないので、λ を挟む区間(lo / hi)と上限到達時に
            返す解には収束した内側の結果だけを使い、打ち切りの点は次の λ の外挿にだけ使う
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
    history_converged: list[bool] = []
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
        if model.groups:
            g_val, resid = model.group_terms(fc)
            data += g_val
            gd = gd + model.group_gradient(fc, resid)
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

    def chi2_of(xk: FloatArray) -> float:
        return model.chi2(model.f_calc_eq(xk if two else to_m(xk)))

    # 目標から遠い λ の予算と、近い λ で「C が落ち着いた」と見なす判定。
    # 近い λ では settle_window 反復ごとに C を測り、変化が settle_tol 以下なら止める
    # (max_inner 反復あたり chi2_tol × _SETTLE_RATE の変化率。上限は _HARD_CAP_FACTOR × max_inner)
    far_budget = max(max_inner // 5, 100)
    settle_window = max(max_inner // _SETTLE_WINDOW_DIV, 100)
    settle_tol = _SETTLE_RATE * chi2_tol * target_chi2 * settle_window / max_inner
    hard_cap = _HARD_CAP_FACTOR * max_inner
    lo: float | None = None  # C > target となった最大の λ(収束した内側の結果だけ)
    hi: float | None = None  # C < target となった最小の λ(同上)
    pts: list[tuple[float, float]] = []  # (log λ, log C)。打ち切りの点も含む(外挿用)
    pts_conv: list[tuple[float, float]] = []  # 収束した点だけ(挟んだ区間の中の割線法用)
    chi2 = float("nan")
    n_iter = 0
    msg = ""
    near_target = False
    # 目標に最も近かった (|log(C/目標)|, λ, C, x)。探索が上限に達したときはこれを返す。
    # 収束した解を優先し、1 つも無いときだけ打ち切りの解から選ぶ
    best_conv: tuple[float, float, float, FloatArray] | None = None
    best_any: tuple[float, float, float, FloatArray] | None = None
    for outer in range(max_outer):
        # 途中で打ち切った C は λ について単調でなく、lo / hi に入れると割線法・二分法が
        # 偶然「目標を少し超えた」打ち切りの点に縮んだまま探索上限に達する(予算を 400 に
        # 減らした実験と macOS の CI、2026-10-17)。遠い λ では小さな予算で動かし(C は
        # 外挿にだけ使う)、一度目標の近くに入ったら以後は C が落ち着くまで解く
        if near_target:
            x, used, status0, _settled = _minimize_at_lambda(
                objective, x, lam, bounds, max_iter=hard_cap, window=settle_window,
                settle_tol=settle_tol, settle_relative=False, chi2_of=chi2_of,
            )  # fmt: skip
            # C が落ち着くか上限(hard_cap)まで解いた結果は、使える最良の C として扱う
            converged = True
        else:
            x, used, status0, _settled = _minimize_at_lambda(
                objective, x, lam, bounds, max_iter=far_budget,
                window=max(far_budget // 4, 25), settle_tol=0.01, settle_relative=True,
                chi2_of=chi2_of,
            )  # fmt: skip
            converged = status0
        n_iter += used
        chi2 = chi2_of(x)
        history.append((lam, chi2))
        history_converged.append(converged)
        dist = abs(math.log(max(chi2, 1e-300) / target_chi2))
        if best_any is None or dist < best_any[0]:
            best_any = (dist, lam, chi2, x.copy())
        if converged and (best_conv is None or dist < best_conv[0]):
            best_conv = (dist, lam, chi2, x.copy())
        if progress is not None:
            progress(outer + 1, lam, chi2)
        if converged and abs(chi2 - target_chi2) <= chi2_tol * target_chi2:
            msg = f"収束: C = {chi2:.4f}(λ = {lam:.3g}、外側反復 {outer + 1} 回)"
            break
        near_target = near_target or dist < math.log(3.0)
        pt = (math.log(lam), math.log(max(chi2, 1e-300)))
        pts.append(pt)
        if converged:
            pts_conv.append(pt)
            # 区間が λ の 0.5% より狭くなってもまだ目標に入らないなら、反対側の端の C は
            # 過渡値だった(C は λ の 0.5% で 1% しか変わらない)。その端を捨てて外挿に戻る
            if chi2 > target_chi2:
                lo = lam if lo is None else max(lo, lam)
                if hi is not None and hi <= lo * (1.0 + _BRACKET_MIN_RATIO):
                    hi = None
            else:
                hi = lam if hi is None else min(hi, lam)
                if lo is not None and hi <= lo * (1.0 + _BRACKET_MIN_RATIO):
                    lo = None
        lam = _next_lambda(pts, pts_conv, lo, hi, target_chi2)
    else:
        # 打ち切りの C は λ について単調でなく、最後の λ が最良とは限らない(GitHub Actions
        # の Windows で同じ入力が C = 0.93 で終わった)。収束した解のうち目標に最も近いものを返す
        best = best_conv if best_conv is not None else best_any
        assert best is not None
        _, lam, chi2, x = best
        msg = (
            f"λ の探索が上限に達しました。目標 {target_chi2} に最も近い C = {chi2:.4f}"
            f"(λ = {lam:.3g})の解を返します"
        )
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
        history_converged=history_converged,
    )


def _minimize_at_lambda(
    objective: Callable[[FloatArray, float], tuple[float, FloatArray]],
    x0: FloatArray,
    lam: float,
    bounds: list[tuple[float, None]] | None,
    *,
    max_iter: int,
    window: int,
    settle_tol: float | None,
    settle_relative: bool,
    chi2_of: Callable[[FloatArray], float],
) -> tuple[FloatArray, int, bool, bool]:
    """1 つの λ で Q = -S + (λ/2) N_F C を L-BFGS で最小化する(x0 を温かい初期値にする)。

    終了は 3 通り: L-BFGS が収束(status 0)、window 反復ごとに測る C の変化が settle_tol 以下
    (settle_relative なら |Δ log C|)になった(C が落ち着いた)、反復数が max_iter に達した(打ち切り)。
    C は L-BFGS の各反復後に呼ばれる callback で測り、落ち着いたら StopIteration で止める
    (記憶を捨てて区切り直すより速く、止める位置も細かい)。

    Returns:
        (x, 使った反復数, L-BFGS が収束したか, C が落ち着いたか)
    """
    x = np.asarray(x0, dtype=np.float64)
    used = 0
    abnormal = 0
    count = 0
    c_prev: float | None = None
    settled = False

    def callback(xk: FloatArray) -> None:
        nonlocal count, c_prev, settled
        count += 1
        if settle_tol is None or count % window:
            return
        c_now = chi2_of(xk)
        if c_prev is not None:
            if settle_relative:
                diff = abs(math.log(max(c_now, 1e-300) / max(c_prev, 1e-300)))
            else:
                diff = abs(c_now - c_prev)
            if diff <= settle_tol:
                settled = True
                raise StopIteration
        c_prev = c_now

    while True:
        res: Any = minimize(
            objective,
            x,
            args=(lam,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": max(max_iter - used, 1),
                "ftol": 1e-13,
                "gtol": 1e-8,
                "maxcor": 15,
            },
        )
        x = np.asarray(res.x, dtype=np.float64)
        used += max(int(res.nit), 1)
        status = int(res.status)
        if status == 0 or settled or used >= max_iter:
            return x, used, status == 0, settled
        if status == 2:
            # 線形探索の失敗(丸めの違いで起きる)。記憶を捨てて同じ点から数回やり直す
            abnormal += 1
            if abnormal > 3:
                return x, used, False, False
            continue
        return x, used, False, False


#: 目標の近くの λ で「C が落ち着いた」と見なす判定: max_inner 反復あたり chi2_tol × _SETTLE_RATE
#: の変化率を、max_inner / _SETTLE_WINDOW_DIV 反復の窓ごとに測る。内側の反復の上限は
#: _HARD_CAP_FACTOR × max_inner
_SETTLE_RATE = 0.5
_SETTLE_WINDOW_DIV = 4
_HARD_CAP_FACTOR = 8
#: d log C / d log λ の典型値(目標の近く。ルチル・NaCl の合成データで -1.7〜-1.9)。
#: 収束した点が目標の近くに 1 つしか無いときの外挿に使う
_TYPICAL_SLOPE = -1.7
#: 挟んだ区間 [lo, hi] がこの比より狭くなってもまだ目標に入らなければ、矛盾する端を捨てる
_BRACKET_MIN_RATIO = 0.005
#: 割線の傾きの推定に使う点は、目標の 1/30〜30 倍の範囲にある収束した点だけ
#: (λ が小さすぎて密度が事前分布のままの点では C が飽和していて傾きが浅く出る)
_INFORMATIVE = math.log(30.0)


def _next_lambda(
    pts: list[tuple[float, float]],
    pts_conv: list[tuple[float, float]],
    lo: float | None,
    hi: float | None,
    target: float,
) -> float:
    """次の λ。log C は log λ についてほぼ直線なので割線法で外挿し、挟めたら区間の中に収める。

    pts は全部の (log λ, log C)(打ち切りの点も含む)、pts_conv は収束した点だけ。
    傾きは、目標の近く(1/30〜30 倍)の収束した点が 2 つあればその割線、1 つなら典型値
    _TYPICAL_SLOPE でその点から、無ければ(遠い λ の段階)直近 2 点の割線で外挿する。
    挟めている(lo / hi がある)ときは割線の点が区間の内側ならそれを使い、外れたら二分(幾何平均)する。
    """
    log_t = math.log(target)
    x_last = pts[-1][0]
    good = [p for p in pts_conv if abs(p[1] - log_t) < _INFORMATIVE]
    xs: float | None = None
    if len(good) >= 2:
        (x1, y1), (x2, y2) = good[-2], good[-1]
        if abs(x2 - x1) > 1e-12 and (y2 - y1) < 0.0:
            xs = x2 + (log_t - y2) * (x2 - x1) / (y2 - y1)
    if xs is None and good:
        x2, y2 = good[-1]
        xs = x2 + (log_t - y2) / _TYPICAL_SLOPE
    if xs is None and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        if abs(x2 - x1) > 1e-12 and (y2 - y1) < 0.0:
            xs = x2 + (log_t - y2) * (x2 - x1) / (y2 - y1)
    if lo is not None and hi is not None:
        # 挟んだ区間の内側なら割線の点を使う(端からの余裕は取らない: 区間が広いと余裕の分だけ
        # 目標の近くの点を捨てて二分に落ち、λ が 100 分の 1 まで飛ぶ)。外れたら二分
        if xs is not None and math.log(lo) < xs < math.log(hi):
            return math.exp(xs)
        return math.sqrt(lo * hi)
    if xs is None:
        # 傾きが取れないときは 10 倍刻み(C > 目標なら λ を増やす)
        return math.exp(x_last) * (10.0 if pts[-1][1] > log_t else 0.1)
    step = max(min(xs - x_last, math.log(100.0)), -math.log(100.0))
    return math.exp(x_last + step)


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
        expo = (lam * f000 / max(model.n_f, 1)) * corr
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
