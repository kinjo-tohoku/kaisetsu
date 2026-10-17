"""原子対分布関数 G(r) の実空間計算(PDFfit2 / diffpy.srreal と同じ定式化)。

    G(r) = (1/(N r)) Σ_i Σ_{j≠i} (b_i b_j / <b>²) δ(r - r_ij) - 4π r ρ_0

各原子対の δ 関数を幅 σ_ij のガウス関数(面積は占有率と散乱長の重み)に置き換える。

    σ_ij² = σ'_ij² (1 - δ1/r_ij - δ2/r_ij² + Q_broad² r_ij²)        (Jeong 2003 / Farrow 2007)
    σ'_ij² = e_ij^T (U_i + U_j) e_ij  (e_ij は原子対方向の単位ベクトル、U は直交座標の変位テンソル。
                                       等方なら U_i + U_j)

有限の Q_max による打ち切り(終端効果)は、細かい r グリッド上で求めた G(r) を Q 空間で
[Q_min, Q_max] に帯域制限することで表す(srreal と同じ)。データの変換で Lorch 窓
M(Q) = sin(πQ/Q_max)/(πQ/Q_max) を掛けてあれば、同じ窓を帯域制限で掛ける。その後に装置分解能
exp(-(Q_damp r)²/2)、球状粒子の形状関数 f(r, d)、スケール因子を掛ける。

原子対のガウス関数は、各グリッド点の値ではなくビン [r_i - Δr/2, r_i + Δr/2] の積分(誤差関数の差)
を Δr で割って積む。σ がグリッド間隔より小さいとき(U = 0 のサイト)も面積(配位数)が保存される。

散乱長 b: 中性子は Sears の b_c(実部)、X 線は f(Q = 0)(電子数。PDFfit2 と同じ)。
<b> は単位胞内の全原子の占有率加重平均、N は単位胞内の原子数(占有率込み)、ρ_0 = N / V。

出典: Th. Proffen & S. J. L. Billinge, J. Appl. Cryst. 32, 572-575 (1999);
      C. L. Farrow et al., J. Phys.: Condens. Matter 19, 335219 (2007) 式 (3)-(10);
      I.-K. Jeong, R. H. Heffner, M. J. Graf & S. J. L. Billinge, Phys. Rev. B 67, 104301 (2003);
      K. Kodama, S. Iikubo, T. Taguchi & S. Shamoto, Acta Cryst. A62, 444-453 (2006)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.structure import ExpandedAtoms, Structure
from kaisetsu.scattering.neutron import NeutronScatteringLengths
from kaisetsu.scattering.xray import XrayScatteringFactors

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TWO_PI2 = 2.0 * math.pi**2


@dataclass
class PDFPhase:
    """PDF に寄与する 1 相とそのパラメータ(PDFfit2 の相パラメータに対応)。

    Attributes:
        structure: 結晶構造
        scale: 相のスケール因子(PDFfit2 の pscale)
        delta1: 相関運動補正の 1/r 項の係数 [Å](高温側で支配的)
        delta2: 相関運動補正の 1/r² 項の係数 [Å²](低温側で支配的)
        sratio: r < rcut の原子対のピーク幅に掛ける比(近距離の鋭いピーク用)
        rcut: sratio を適用する距離の上限 [Å]
        spdiameter: 球状粒子の直径 [Å]。0 ならバルク(形状関数なし)
        stepcut: この距離より先の G(r) を 0 にする [Å]。0 なら無効
    """

    structure: Structure
    scale: float = 1.0
    delta1: float = 0.0
    delta2: float = 0.0
    sratio: float = 1.0
    rcut: float = 0.0
    spdiameter: float = 0.0
    stepcut: float = 0.0


@dataclass
class PairList:
    """単位胞内の原子を起点とする原子対のリスト(同じ (r, σ'²) の対はまとめ済み)。

    Attributes:
        distance: r_ij [Å]
        sigma2_prime: σ'_ij² [Å²]
        weight: 占有率と散乱長の重み occ_i occ_j b_i b_j / <b>²(まとめた対は合算)
        n_atoms: 単位胞内の原子数 N(占有率込み)
        rho0: 数密度 ρ_0 = N / V [Å^-3]
    """

    distance: FloatArray
    sigma2_prime: FloatArray
    weight: FloatArray
    n_atoms: float
    rho0: float


@dataclass
class PDFResult:
    r: FloatArray
    g_calc: FloatArray
    g_phases: list[FloatArray] = field(default_factory=list)
    pairs: list[PairList] = field(default_factory=list)


# ----------------------------------------------------------------------
# 散乱長と変位テンソル
# ----------------------------------------------------------------------
def scattering_lengths(species: Sequence[str], radiation: str) -> FloatArray:
    """散乱種ごとの b(中性子: b_c の実部 [fm]、X 線: f(0) = 電子数)。"""
    if radiation == "neutron":
        tbl = NeutronScatteringLengths()
        return np.array([tbl.scattering_length(sp).real for sp in species], dtype=np.float64)
    if radiation == "xray":
        xt = XrayScatteringFactors()
        return np.array([float(xt.f0(sp, 0.0)) for sp in species], dtype=np.float64)
    raise ValueError(f"radiation は 'xray' か 'neutron' です: {radiation!r}")


def cartesian_adps(structure: Structure, expanded: ExpandedAtoms | None = None) -> FloatArray:
    """展開した各原子の変位テンソル U を直交座標 [Å²] で返す (N, 3, 3)。

    U_cart = M N U^{ij} N M^T(M: 直交化行列、N = diag(a*, b*, c*))。
    ExpandedAtoms は β_ij = 2π² a*_i a*_j U'^{ij} を持つので β / 2π² を N U N として使う。
    """
    ex = structure.expand() if expanded is None else expanded
    m = structure.lattice.orthogonalization_matrix
    n = len(ex.positions)
    out = np.zeros((n, 3, 3))
    iso = ~ex.is_anisotropic
    eye = np.eye(3)
    out[iso] = ex.u_iso[iso][:, None, None] * eye[None]
    if np.any(~iso):
        d = ex.beta[~iso] / TWO_PI2
        out[~iso] = np.einsum("ab,nbc,dc->nad", m, d, m)
    return out


# ----------------------------------------------------------------------
# 原子対の列挙
# ----------------------------------------------------------------------
def pair_distribution(
    structure: Structure,
    r_max: float,
    radiation: str = "xray",
    *,
    merge_tol: float = 1e-5,
) -> PairList:
    """r ≤ r_max の原子対を列挙し、(r, σ'²) が同じものをまとめる。"""
    ex = structure.expand()
    lat = structure.lattice
    m = lat.orthogonalization_matrix
    pos = ex.positions
    n = len(pos)
    if n == 0:
        z = np.zeros(0)
        return PairList(z, z, z, 0.0, 0.0)
    b_species = scattering_lengths(ex.species, radiation)
    b_atom = b_species[ex.species_index]
    occ = ex.occupancy
    n_atoms = float(np.sum(occ))
    b_mean = float(np.sum(occ * b_atom) / n_atoms) if n_atoms > 0 else 1.0
    if abs(b_mean) < 1e-12:
        raise ValueError("平均散乱長が 0 です。PDF の規格化ができません")
    u_cart = cartesian_adps(structure, ex)

    rec = lat.reciprocal
    nmax = [math.ceil(r_max * v) + 1 for v in (rec.a, rec.b, rec.c)]
    ranges = [np.arange(-k, k + 1) for k in nmax]
    trans = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)

    dist_parts: list[FloatArray] = []
    s2_parts: list[FloatArray] = []
    w_parts: list[FloatArray] = []
    r2max = r_max * r_max
    j_all = np.broadcast_to(np.arange(n)[None, :], (len(trans), n)).reshape(-1)
    for i in range(n):
        frac = pos[None, :, :] + trans[:, None, :] - pos[i][None, None, :]  # (K, N, 3)
        cart = frac.reshape(-1, 3) @ m.T
        r2 = np.einsum("ij,ij->i", cart, cart)
        keep = (r2 > 1e-10) & (r2 <= r2max)
        if not np.any(keep):
            continue
        cart = cart[keep]
        r = np.sqrt(r2[keep])
        j = j_all[keep]
        rhat = cart / r[:, None]
        usum = u_cart[i][None, :, :] + u_cart[j]
        s2 = np.einsum("pa,pab,pb->p", rhat, usum, rhat)
        w = occ[i] * occ[j] * b_atom[i] * b_atom[j] / (b_mean * b_mean)
        dist_parts.append(r)
        s2_parts.append(s2)
        w_parts.append(w)
    if not dist_parts:
        z = np.zeros(0)
        return PairList(z, z, z, n_atoms, n_atoms / lat.volume)
    dist = np.concatenate(dist_parts)
    s2 = np.concatenate(s2_parts)
    w = np.concatenate(w_parts)
    # (r, σ'²) が同じ対をまとめる
    key = np.stack([np.round(dist / merge_tol), np.round(s2 / (merge_tol * 0.1))], axis=1)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    w_sum = np.bincount(inv, weights=w, minlength=len(idx))
    order = np.argsort(dist[idx], kind="stable")
    return PairList(dist[idx][order], s2[idx][order], w_sum[order], n_atoms, n_atoms / lat.volume)


# ----------------------------------------------------------------------
# ガウス関数の積算(numba)
# ----------------------------------------------------------------------
@nb.njit(cache=True)
def _accumulate_gaussians(
    r0: float,
    dr: float,
    n: int,
    centers: FloatArray,
    sigmas: FloatArray,
    weights: FloatArray,
    out: FloatArray,
) -> None:
    """重み w、中心 c、幅 σ のガウス関数を等間隔グリッド(r0 + i Δr)に積む。

    グリッド点 i には、ビン [r_i - Δr/2, r_i + Δr/2] の積分
        w [Φ((r_i + Δr/2 - c)/σ) - Φ((r_i - Δr/2 - c)/σ)] / Δr   (Φ: 標準正規分布の累積分布)
    を加える。点の値 w φ((r_i - c)/σ)/σ を使うと σ < Δr(U = 0 のサイト)で面積が c の位置に
    よって 0〜4 倍に変わるが、ビンの積分なら σ によらず Σ_i out_i Δr = w が保たれる。
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for p in range(len(centers)):
        c = centers[p]
        s = sigmas[p]
        w = weights[p]
        if w == 0.0:
            continue
        half = 6.0 * s + dr
        i0 = math.floor((c - half - r0) / dr)
        i1 = math.ceil((c + half - r0) / dr)
        if i1 < 0 or i0 >= n:
            continue
        if i0 < 0:
            i0 = 0
        if i1 > n - 1:
            i1 = n - 1
        scale = w / dr
        lower = 0.5 * (1.0 + math.erf((r0 + (i0 - 0.5) * dr - c) / s * inv_sqrt2))
        for i in range(i0, i1 + 1):
            upper = 0.5 * (1.0 + math.erf((r0 + (i + 0.5) * dr - c) / s * inv_sqrt2))
            out[i] += scale * (upper - lower)
            lower = upper


def peak_widths(pairs: PairList, phase: PDFPhase, qbroad: float) -> FloatArray:
    """σ_ij = σ'_ij sqrt(1 - δ1/r - δ2/r² + Q_broad² r²)(sratio / rcut 込み)。"""
    r = pairs.distance
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = 1.0 - phase.delta1 / r - phase.delta2 / (r * r) + (qbroad * r) ** 2
    s2 = np.where(corr > 0.0, pairs.sigma2_prime * corr, 0.0)
    sigma = np.sqrt(np.maximum(s2, 0.0))
    if phase.rcut > 0.0 and phase.sratio != 1.0:
        sigma = np.where(r < phase.rcut, sigma * phase.sratio, sigma)
    # 下限は 0 割り防止だけ(ビン積分で積むので面積は σ によらず保たれる)
    return np.asarray(np.maximum(sigma, 1e-6))


def spherical_shape_factor(r: FloatArray, diameter: float) -> FloatArray:
    """球状粒子の形状関数 f(r, d) = 1 - 3r/(2d) + r³/(2d³)(r < d)。Kodama et al. (2006)。"""
    if diameter <= 0.0:
        return np.ones_like(r)
    x = r / diameter
    f = 1.0 - 1.5 * x + 0.5 * x**3
    return np.asarray(np.where(r < diameter, f, 0.0))


def lorch_window(q: FloatArray, qmax: float) -> FloatArray:
    """Lorch 窓 M(Q) = sin(πQ/Q_max)/(πQ/Q_max)(E. Lorch, J. Phys. C 2, 229 (1969))。Q = 0 で 1。"""
    x = math.pi * np.asarray(q, dtype=np.float64) / qmax
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(x > 0, np.sin(x) / x, 1.0))


def bandlimit(
    g: FloatArray, dr: float, qmin: float, qmax: float, lorch: bool = False
) -> FloatArray:
    """等間隔グリッド上の奇関数 G(r)(r = 0 から)を Q 空間で [qmin, qmax] に帯域制限する。

    lorch が真なら帯域内に Lorch 窓 M(Q) も掛ける(データの変換で窓を掛けたときにモデルも同じ
    窓で鈍らせる。掛けないと U が 2 倍ほど大きく精密化される)。
    """
    if qmax <= 0.0 and qmin <= 0.0:
        return g
    m = len(g)
    ext = np.concatenate([g, [0.0], -g[:0:-1]])  # 奇関数に拡張(長さ 2m)
    spec = np.fft.rfft(ext)
    q = 2.0 * math.pi * np.fft.rfftfreq(len(ext), d=dr)
    keep = np.ones_like(q, dtype=bool)
    if qmax > 0.0:
        keep &= q <= qmax
        if lorch:
            spec *= lorch_window(q, qmax)
    if qmin > 0.0:
        keep &= q >= qmin
    spec[~keep] = 0.0
    return np.asarray(np.fft.irfft(spec, n=len(ext))[:m])


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
def calculate_pdf(
    r: npt.ArrayLike,
    phases: Sequence[PDFPhase],
    radiation: str = "xray",
    *,
    qmax: float = 0.0,
    qmin: float = 0.0,
    qdamp: float = 0.0,
    qbroad: float = 0.0,
    scale: float = 1.0,
    dr_fine: float = 0.01,
    pad: float = 6.0,
    pairs: Sequence[PairList] | None = None,
    lorch: bool = False,
) -> PDFResult:
    """G(r) を計算する。

    Args:
        r: 計算する距離 [Å] の配列(昇順)
        phases: 相のリスト
        radiation: 'xray' または 'neutron'
        qmax, qmin: 終端効果を表す帯域 [Å^-1]。0 なら無効
        lorch: データの変換で Lorch 窓を掛けていれば真(帯域制限で同じ窓を掛ける)
        qdamp: 装置分解能による減衰 exp(-(Q_damp r)²/2) [Å^-1]
        qbroad: 分解能によるピーク幅の広がり [Å^-1]
        scale: データセット全体のスケール因子(PDFfit2 の dscale)
        dr_fine: 内部グリッドの刻み [Å]
        pad: 内部グリッドを r_max より先に延ばす長さ [Å](ピークの裾と帯域制限の端の処理)
        pairs: 相ごとの原子対リスト(構造が変わっていなければ使い回せる)
    """
    ra = np.asarray(r, dtype=np.float64).ravel()
    if ra.size == 0:
        raise ValueError("r が空です")
    r_max = float(ra[-1])
    dr = min(dr_fine, float(np.min(np.diff(ra))) if ra.size > 1 else dr_fine)
    n_fine = math.ceil((r_max + pad) / dr) + 1
    r_fine = np.arange(n_fine, dtype=np.float64) * dr
    r_safe = np.where(r_fine > 0, r_fine, 1.0)

    g_total = np.zeros(n_fine)
    g_phases: list[FloatArray] = []
    used_pairs: list[PairList] = []
    for k, ph in enumerate(phases):
        pl = (
            pairs[k]
            if pairs is not None
            else pair_distribution(ph.structure, r_max + pad, radiation)
        )
        used_pairs.append(pl)
        rdf = np.zeros(n_fine)
        if len(pl.distance):
            sig = peak_widths(pl, ph, qbroad)
            _accumulate_gaussians(0.0, dr, n_fine, pl.distance, sig, pl.weight, rdf)
        n_at = pl.n_atoms if pl.n_atoms > 0 else 1.0
        g = rdf / (n_at * r_safe) - 4.0 * math.pi * r_fine * pl.rho0
        g[0] = 0.0
        g = bandlimit(g, dr, qmin, qmax, lorch)
        g *= spherical_shape_factor(r_fine, ph.spdiameter)
        if ph.stepcut > 0.0:
            g = np.where(r_fine > ph.stepcut, 0.0, g)
        g *= ph.scale
        g_phases.append(g)
        g_total += g
    env = scale * np.exp(-0.5 * (qdamp * r_fine) ** 2)
    g_total *= env
    g_phases = [gp * env for gp in g_phases]
    return PDFResult(
        r=ra,
        g_calc=np.interp(ra, r_fine, g_total),
        g_phases=[np.interp(ra, r_fine, gp) for gp in g_phases],
        pairs=used_pairs,
    )
