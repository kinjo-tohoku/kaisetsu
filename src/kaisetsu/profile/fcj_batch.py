"""複数ピークの FCJ pseudo-Voigt を一括計算する numba 並列カーネル。

ピークごとに並列(prange)で計算し、各ピークの寄与を 1 本の平坦な配列
(窓ごとにオフセット付き)に書き出す。パターン計算で 1 回の呼び出しに全ピークを
まとめることで、Python 側のループと関数呼び出しのオーバーヘッドを無くし、全コアを使う。

積分法は 2 通り:

- ``"accurate"``(既定): 変数変換 δ = 2θ - S u²(S = 2θ - 2φ_min)で δ → 2θ の
  1/√(2θ-δ) 特異点を消し、変曲点 2φ_infl で区間を分けて Gauss-Legendre 則を当てる。
  指数的に収束し、1 パネル 8〜48 節点で相対 1e-5 以下。
- ``"gsas"``: GSAS/GSAS-II の PSVFCJO と同じ(Gauss-Legendre 則の正の半分、節点数は
  3·S/Γ_L の表引き)。特異点のため収束は 1/n 程度で、低角では 1e-3 程度の誤差が残る。
  GSAS-II との数値比較用。
"""

from __future__ import annotations

import math
from functools import cache

import numba as nb
import numpy as np
import numpy.typing as npt

from kaisetsu.profile.fcj import _NTERMS, _half_gauss_legendre
from kaisetsu.profile.pseudo_voigt import pseudo_voigt_point, tch_fwhm_eta

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_MAX_HALF = max(_NTERMS) // 2
_NTERMS_ARR = np.array(_NTERMS, dtype=np.int64)

#: 置換法のパネルあたり節点数の候補
_SUBST_SIZES = (8, 12, 16, 24, 32, 48)
_SUBST_SIZES_ARR = np.array(_SUBST_SIZES, dtype=np.int64)

METHOD_GSAS = 0
METHOD_ACCURATE = 1


@cache
def _node_tables() -> tuple[FloatArray, FloatArray, IntArray]:
    """GSAS 流(正の半分)の求積点表(行 = 表、列 = 節点)。"""
    tab_x = np.zeros((len(_NTERMS), _MAX_HALF))
    tab_w = np.zeros((len(_NTERMS), _MAX_HALF))
    tab_n = np.zeros(len(_NTERMS), dtype=np.int64)
    for i, n in enumerate(_NTERMS):
        x, w = _half_gauss_legendre(n)
        tab_x[i, : len(x)] = x
        tab_w[i, : len(w)] = w
        tab_n[i] = len(x)
    return tab_x, tab_w, tab_n


@cache
def _full_tables() -> tuple[FloatArray, FloatArray, IntArray]:
    """置換法用の正規 Gauss-Legendre 則の表(区間 [-1, 1])。"""
    m = max(_SUBST_SIZES)
    tab_x = np.zeros((len(_SUBST_SIZES), m))
    tab_w = np.zeros((len(_SUBST_SIZES), m))
    tab_n = np.zeros(len(_SUBST_SIZES), dtype=np.int64)
    for i, n in enumerate(_SUBST_SIZES):
        x, w = np.polynomial.legendre.leggauss(n)
        tab_x[i, :n] = x
        tab_w[i, :n] = w
        tab_n[i] = n
    return tab_x, tab_w, tab_n


def fcj_limits_array(pos: FloatArray, s_l: float, h_l: float) -> tuple[FloatArray, FloatArray]:
    """2φ_min と 2φ_infl [deg] をピーク配列に対して求める(FCJ94 式 (4), (5))。"""
    cos2t = np.cos(np.radians(pos))
    apb = s_l + h_l
    amb = s_l - h_l
    arg_infl = math.sqrt(1.0 + amb * amb) * cos2t
    infl = np.where(np.abs(arg_infl) <= 1.0, np.degrees(np.arccos(np.clip(arg_infl, -1, 1))), pos)
    if s_l == 0.0 or h_l == 0.0:
        infl = pos.copy()
    arg_min = math.sqrt(1.0 + apb * apb) * cos2t
    tpm = np.where(
        np.abs(arg_min) <= 1.0,
        np.degrees(np.arccos(np.clip(arg_min, -1, 1))),
        np.where(arg_min > 0, 0.0, 180.0),
    )
    return np.asarray(tpm, dtype=np.float64), np.asarray(infl, dtype=np.float64)


def quadrature_table_index(
    pos: FloatArray, two_phi_min: FloatArray, fwhm_l: FloatArray
) -> IntArray:
    """GSAS 流の節点数の表の添字(3·S/Γ_L を超える最小の表)。"""
    span = np.abs(pos - two_phi_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(fwhm_l > 0, 3.0 * span / np.maximum(fwhm_l, 1e-12), span * 200.0)
    k = np.where(np.isfinite(k), k, _NTERMS[-1]).astype(np.int64)
    idx = np.searchsorted(_NTERMS_ARR, k, side="left")
    return np.asarray(np.minimum(idx, len(_NTERMS) - 1), dtype=np.int64)


def substitution_table_index(
    pos: FloatArray, two_phi_min: FloatArray, fwhm: FloatArray
) -> IntArray:
    """置換法のパネル節点数の表の添字。n ≈ 8 + 4·S/Γ(8〜48)。"""
    span = np.abs(pos - two_phi_min)
    n = 8.0 + 4.0 * span / np.maximum(fwhm, 1e-12)
    idx = np.searchsorted(_SUBST_SIZES_ARR, np.ceil(n).astype(np.int64), side="left")
    return np.asarray(np.minimum(idx, len(_SUBST_SIZES) - 1), dtype=np.int64)


@nb.njit(cache=True, inline="always")
def _fcj_weight(
    delta: float, pos: float, tpm: float, span_infl: float, cos2t: float, cos2t2: float,
    sin2t2: float, apb: float, small: float,
) -> float:  # fmt: skip
    rad = math.pi / 180.0
    cosd = math.cos(delta * rad)
    sind = math.sin(delta * rad)
    if pos > 45.0:
        tmp = cosd * cosd - cos2t2
    else:
        tmp = sin2t2 - sind * sind
    if tmp > 0.0:
        f = abs(cos2t) / math.sqrt(tmp)
    else:
        f = 1.0
    if abs(delta - tpm) > span_infl:
        return 2.0 * small * f / cosd
    return (-1.0 + apb * f) / cosd


@nb.njit(cache=True, parallel=True)
def _batch_kernel(
    x: FloatArray,
    pos: FloatArray,
    fwhm_g: FloatArray,
    fwhm_l: FloatArray,
    inten: FloatArray,
    i0: IntArray,
    i1: IntArray,
    offsets: IntArray,
    two_phi_min: FloatArray,
    two_phi_infl: FloatArray,
    s_l: float,
    h_l: float,
    method: int,
    tab_idx: IntArray,
    tab_x: FloatArray,
    tab_w: FloatArray,
    tab_n: IntArray,
    out: FloatArray,
) -> None:
    rad = math.pi / 180.0
    apb = s_l + h_l
    small = min(s_l, h_l)
    asym = s_l > 0.0 or h_l > 0.0
    for p in nb.prange(pos.shape[0]):
        a = i0[p]
        b = i1[p]
        if b <= a:
            continue
        off = offsets[p]
        fwhm, eta = tch_fwhm_eta(fwhm_g[p], fwhm_l[p])
        amp = inten[p]
        pk = pos[p]
        if not asym:
            for i in range(a, b):
                out[off + i - a] = amp * pseudo_voigt_point(x[i] - pk, fwhm, eta)
            continue
        cos2t = math.cos(pk * rad)
        sin2t = math.sin(pk * rad)
        cos2t2 = cos2t * cos2t
        sin2t2 = sin2t * sin2t
        tpm = two_phi_min[p]
        span_infl = abs(two_phi_infl[p] - tpm)
        t = tab_idx[p]
        nn = tab_n[t]
        for i in range(a, b):
            out[off + i - a] = 0.0
        sum_wg = 0.0
        if method == 0:
            # GSAS 流: δ = 2φ_min + S·ξ, ξ は正の半分の節点
            for k in range(nn):
                delta = tpm + (pk - tpm) * tab_x[t, k]
                g = _fcj_weight(delta, pk, tpm, span_infl, cos2t, cos2t2, sin2t2, apb, small)
                wg = tab_w[t, k] * g
                sum_wg += wg
                for i in range(a, b):
                    out[off + i - a] += wg * pseudo_voigt_point(x[i] - delta, fwhm, eta)
        else:
            # 置換法: δ = 2θ - S u², u ∈ [0, u_infl] ∪ [u_infl, 1]
            span = pk - tpm
            if span == 0.0:
                for i in range(a, b):
                    out[off + i - a] = amp * pseudo_voigt_point(x[i] - pk, fwhm, eta)
                continue
            ratio = (pk - two_phi_infl[p]) / span
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0
            u_infl = math.sqrt(ratio)
            for panel in range(2):
                if panel == 0:
                    ua, ub = 0.0, u_infl
                else:
                    ua, ub = u_infl, 1.0
                if ub - ua < 1e-12:
                    continue
                half = 0.5 * (ub - ua)
                mid = 0.5 * (ub + ua)
                for k in range(nn):
                    u = mid + half * tab_x[t, k]
                    delta = pk - span * u * u
                    g = _fcj_weight(delta, pk, tpm, span_infl, cos2t, cos2t2, sin2t2, apb, small)
                    wg = tab_w[t, k] * half * g * 2.0 * span * u
                    sum_wg += wg
                    for i in range(a, b):
                        out[off + i - a] += wg * pseudo_voigt_point(x[i] - delta, fwhm, eta)
        if sum_wg != 0.0:
            scale = amp / sum_wg
            for i in range(a, b):
                out[off + i - a] *= scale


def fcj_profile_batch(
    x: FloatArray,
    pos: FloatArray,
    fwhm_g: FloatArray,
    fwhm_l: FloatArray,
    inten: FloatArray,
    i0: IntArray,
    i1: IntArray,
    s_l: float,
    h_l: float,
    method: str = "accurate",
) -> tuple[FloatArray, IntArray, IntArray]:
    """全ピークの寄与(強度 × 面積 1 のプロファイル)を一括計算する。

    Returns:
        out: 平坦な寄与配列。ピーク p の寄与は out[offsets[p] : offsets[p] + lengths[p]]
        offsets: 各ピークの開始オフセット
        lengths: 各ピークの窓の長さ(i1 - i0、負なら 0)
    """
    if method not in ("accurate", "gsas"):
        raise ValueError(f"FCJ の積分法は 'accurate' か 'gsas' です: {method!r}")
    n = len(pos)
    lengths = np.maximum(i1 - i0, 0).astype(np.int64)
    offsets = np.zeros(n, dtype=np.int64)
    if n > 1:
        offsets[1:] = np.cumsum(lengths)[:-1]
    out = np.zeros(int(lengths.sum()), dtype=np.float64)
    if n == 0 or out.size == 0:
        return out, offsets, lengths
    s_l = max(float(s_l), 0.0)
    h_l = max(float(h_l), 0.0)
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    fg = np.ascontiguousarray(fwhm_g, dtype=np.float64)
    fl = np.ascontiguousarray(fwhm_l, dtype=np.float64)
    meth = METHOD_GSAS if method == "gsas" else METHOD_ACCURATE
    if s_l > 0.0 or h_l > 0.0:
        tpm, tpi = fcj_limits_array(pos_c, s_l, h_l)
        if meth == METHOD_GSAS:
            tab_idx = quadrature_table_index(pos_c, tpm, fl)
            tab_x, tab_w, tab_n = _node_tables()
        else:
            fwhm_tot = np.array(
                [tch_fwhm_eta(float(a), float(b))[0] for a, b in zip(fg, fl, strict=True)]
            )
            tab_idx = substitution_table_index(pos_c, tpm, fwhm_tot)
            tab_x, tab_w, tab_n = _full_tables()
    else:
        tpm = pos_c.copy()
        tpi = pos_c.copy()
        tab_idx = np.zeros(n, dtype=np.int64)
        tab_x, tab_w, tab_n = _full_tables()
    _batch_kernel(
        np.ascontiguousarray(x, dtype=np.float64),
        pos_c,
        fg,
        fl,
        np.ascontiguousarray(inten, dtype=np.float64),
        np.ascontiguousarray(i0, dtype=np.int64),
        np.ascontiguousarray(i0 + lengths, dtype=np.int64),
        offsets,
        np.ascontiguousarray(tpm),
        np.ascontiguousarray(tpi),
        s_l,
        h_l,
        meth,
        tab_idx,
        tab_x,
        tab_w,
        tab_n,
        out,
    )
    return out, offsets, lengths
