"""Finger-Cox-Jephcoat 軸発散非対称を畳み込んだ pseudo-Voigt。

Bragg 角 2θ_B のピークに対し、観測角 x での値を

    P(x) = Σ_k w_k G(δ_k) pV(x - δ_k) / Σ_k w_k G(δ_k),   δ_k ∈ [2φ_min, 2θ_B]

で計算する。G(δ) は FCJ94 の式 (7) の重み関数 W(δ)/(h cos δ)、
2φ_min は cos 2φ_min = sqrt(1 + (S/L + H/L)^2) cos 2θ_B(式 (4))、
変曲点 2φ_infl は cos 2φ_infl = sqrt(1 + (S/L - H/L)^2) cos 2θ_B(式 (5))。
積分は Gauss-Legendre 求積で行う。

実装は GSAS-II の Fortran ルーチン PSVFCJO(B. H. Toby, 1995)と同じ手順にしてあり、
GSAS-II との数値比較ができる。

出典:
    L. W. Finger, D. E. Cox & A. P. Jephcoat, J. Appl. Cryst. 27, 892-900 (1994)
"""

from __future__ import annotations

import math
from functools import cache

import numba as nb
import numpy as np
import numpy.typing as npt

from kaisetsu.profile.pseudo_voigt import pseudo_voigt_point, tch_fwhm_eta

FloatArray = npt.NDArray[np.float64]

#: GSAS と同じ求積点数の表(偶数)。[2φ_min, 2θ_B] を Γ_L の 1/3 以下の間隔で刻む。
_NTERMS = (
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 40, 50, 60, 70, 80, 90,
    100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 400,
)  # fmt: skip


@cache
def _half_gauss_legendre(n: int) -> tuple[FloatArray, FloatArray]:
    """n 点 Gauss-Legendre 則の正の半分(節点 (0,1) と重み)。"""
    x, w = np.polynomial.legendre.leggauss(n)
    pos = x > 0
    return np.ascontiguousarray(x[pos]), np.ascontiguousarray(w[pos])


def quadrature_order(two_theta: float, two_phi_min: float, fwhm: float) -> int:
    """GSAS 流の求積点数: [2φ_min, 2θ] を幅 fwhm(GSAS では Γ_L)の 1/3 以下の間隔で刻む。"""
    span = abs(two_theta - two_phi_min)
    if fwhm <= 0.0:
        k = int(span * 200.0)
    else:
        k = int(3.0 * span / fwhm)
    for n in _NTERMS:
        if k <= n:
            return n
    return _NTERMS[-1]


@nb.njit(cache=True)
def _fcj_kernel(
    x: FloatArray,
    pos: float,
    fwhm_g: float,
    fwhm_l: float,
    s_l: float,
    h_l: float,
    two_phi_min: float,
    two_phi_infl: float,
    nodes: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    fwhm, eta = tch_fwhm_eta(fwhm_g, fwhm_l)
    n = x.shape[0]
    out = np.zeros(n)
    rad = math.pi / 180.0
    cos2t = math.cos(pos * rad)
    sin2t = math.sin(pos * rad)
    cos2t2 = cos2t * cos2t
    sin2t2 = sin2t * sin2t
    apb = s_l + h_l
    small = min(s_l, h_l)
    sum_wg = 0.0
    span_infl = abs(two_phi_infl - two_phi_min)
    for k in range(nodes.shape[0]):
        delta = two_phi_min + (pos - two_phi_min) * nodes[k]
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
        if abs(delta - two_phi_min) > span_infl:
            g = 2.0 * small * f / cosd
        else:
            g = (-1.0 + apb * f) / cosd
        wg = weights[k] * g
        sum_wg += wg
        for i in range(n):
            out[i] += wg * pseudo_voigt_point(x[i] - delta, fwhm, eta)
    if sum_wg != 0.0:
        for i in range(n):
            out[i] /= sum_wg
    return out


@nb.njit(cache=True)
def _pv_kernel(x: FloatArray, pos: float, fwhm_g: float, fwhm_l: float) -> FloatArray:
    fwhm, eta = tch_fwhm_eta(fwhm_g, fwhm_l)
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        out[i] = pseudo_voigt_point(x[i] - pos, fwhm, eta)
    return out


def fcj_limits(two_theta: float, s_l: float, h_l: float) -> tuple[float, float]:
    """2φ_min と 2φ_infl [deg] を返す(FCJ94 式 (4), (5))。"""
    cos2t = math.cos(math.radians(two_theta))
    apb = s_l + h_l
    amb = s_l - h_l
    arg_infl = math.sqrt(1.0 + amb * amb) * cos2t
    if s_l == 0.0 or h_l == 0.0 or abs(arg_infl) > 1.0:
        two_phi_infl = two_theta
    else:
        two_phi_infl = math.degrees(math.acos(arg_infl))
    arg_min = math.sqrt(1.0 + apb * apb) * cos2t
    if abs(arg_min) <= 1.0:
        two_phi_min = math.degrees(math.acos(arg_min))
    else:
        two_phi_min = 0.0 if arg_min > 0 else 180.0
    return two_phi_min, two_phi_infl


def fcj_pseudo_voigt(
    x: npt.ArrayLike,
    pos: float,
    fwhm_g: float,
    fwhm_l: float,
    s_l: float,
    h_l: float,
    n_quad: int | None = None,
    method: str = "accurate",
) -> FloatArray:
    """FCJ 非対称 pseudo-Voigt(面積 ≈ 1)。

    Args:
        x: 観測角 2θ [deg] の配列
        pos: ピーク位置 2θ_B [deg]
        fwhm_g, fwhm_l: ガウス・ローレンツ成分の FWHM [deg]
        s_l, h_l: 試料長 S / 検出器スリット長 H を試料-検出器距離 L で割ったもの
        n_quad: 求積点数('gsas' 法のみ。None なら自動選択。'accurate' 法は一括カーネルが
            点数を自分で決めるので無視する)。
        method: 'accurate'(変数変換で特異点を除いた Gauss-Legendre、既定)または
            'gsas'(GSAS-II と同じ正の半分の節点)。詳細は fcj_batch モジュール参照。
    """
    xa = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    if s_l <= 0.0 and h_l <= 0.0:
        return np.asarray(_pv_kernel(xa, float(pos), float(fwhm_g), float(fwhm_l))).reshape(
            np.shape(x)
        )
    if method == "accurate":
        from kaisetsu.profile.fcj_batch import fcj_profile_batch

        n_pts = len(xa)
        out, _, _ = fcj_profile_batch(
            xa,
            np.array([float(pos)]),
            np.array([float(fwhm_g)]),
            np.array([float(fwhm_l)]),
            np.array([1.0]),
            np.array([0], dtype=np.int64),
            np.array([n_pts], dtype=np.int64),
            float(s_l),
            float(h_l),
            method="accurate",
        )
        return np.asarray(out).reshape(np.shape(x))
    if method != "gsas":
        raise ValueError(f"FCJ の積分法は 'accurate' か 'gsas' です: {method!r}")
    s_l = max(float(s_l), 0.0)
    h_l = max(float(h_l), 0.0)
    two_phi_min, two_phi_infl = fcj_limits(float(pos), s_l, h_l)
    n = n_quad if n_quad is not None else quadrature_order(float(pos), two_phi_min, float(fwhm_l))
    nodes, weights = _half_gauss_legendre(int(n))
    out = _fcj_kernel(
        xa, float(pos), float(fwhm_g), float(fwhm_l), s_l, h_l,
        two_phi_min, two_phi_infl, nodes, weights,
    )  # fmt: skip
    return np.asarray(out).reshape(np.shape(x))
