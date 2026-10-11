"""TOF ピーク形状の一括計算(numba、ピークごとに並列)。

`tof_profile.tof_peak` と同じ閉形式(back-to-back 指数 × 擬 Voigt)を、全反射についてまとめて
計算する。複素指数積分 e^w E₁(w) は scipy に頼らず、級数(負の実軸の近く・|w| < 1)、
連分数(修正 Lentz 法)、漸近展開(|w| > 50)で評価する(scipy.special.exp1 と相対差 3e-12 以内)。
erfcx は math.erfc と漸近展開(y > 25)で作る。

出典: W. H. Press et al., *Numerical Recipes* 3rd ed., 6.3(指数積分の級数と連分数);
      M. Abramowitz & I. A. Stegun, 5.1.51(漸近展開), 7.1.23(erfc の漸近展開)。
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import numpy.typing as npt

from kaisetsu.profile.pseudo_voigt import tch_fwhm_eta

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_EULER = 0.5772156649015329
_STOFW = 2.0 * math.sqrt(2.0 * math.log(2.0))
_SQRT_PI = math.sqrt(math.pi)


@nb.njit(cache=True, inline="always")
def _scaled_e1(w: complex) -> complex:
    """e^w E₁(w)(負の実軸上を除く複素平面)。"""
    aw = abs(w)
    if aw == 0.0:
        return complex(np.inf, 0.0)
    if aw > 50.0:
        inv = 1.0 / w
        term = inv
        s = inv
        for k in range(1, 30):
            term = -term * k * inv
            s += term
            if abs(term) < 1e-17 * abs(s):
                break
        return s
    if aw < 1.0 or (w.real < 0.0 and abs(w.imag) < -w.real):
        # 級数: 負の実軸の近くでは打ち消しが無いので |w| が大きくても使える
        s = -_EULER - np.log(w)
        term = 1.0 + 0.0j
        for k in range(1, 2000):
            term = term * (-w) / k
            s -= term / k
            if abs(term) < 1e-17 * k * abs(s):
                break
        return complex(np.exp(w) * s)
    # 連分数(修正 Lentz 法): e^w E₁(w) = 1/(w+1- 1²/(w+3- 2²/(w+5- …)))
    tiny = 1e-300
    b = w + 1.0
    c = complex(1.0 / tiny, 0.0)
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        a = -float(i * i)
        b = b + 2.0
        d = a * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + a / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        h = h * delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


@nb.njit(cache=True, inline="always")
def _hfunc(ex: float, y: float) -> float:
    """exp(ex + y²) erfc(y) を桁あふれなく計算する(GSAS の HFUNC に相当)。"""
    if y < -3.0:
        return math.exp(ex + y * y) * math.erfc(y)
    if y > 25.0:
        inv = 1.0 / y
        inv2 = inv * inv
        series = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2 - 1.875 * inv2**3 + 6.5625 * inv2**4
        return math.exp(ex) * inv / _SQRT_PI * series
    return math.exp(ex) * math.exp(y * y) * math.erfc(y)


@nb.njit(cache=True, parallel=True)
def _tof_batch_kernel(
    x: FloatArray,
    pos: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    sigma2: FloatArray,
    gamma: FloatArray,
    inten: FloatArray,
    i0: IntArray,
    i1: IntArray,
    offsets: IntArray,
    out: FloatArray,
) -> None:
    n_peak = len(pos)
    for k in nb.prange(n_peak):
        n = i1[k] - i0[k]
        if n <= 0:
            continue
        a = max(alpha[k], 1e-6)
        b = max(beta[k], 1e-6)
        fwhm_g = _STOFW * math.sqrt(max(sigma2[k], 1e-10))
        fwhm, eta = tch_fwhm_eta(fwhm_g, max(gamma[k], 1e-5))
        sig_p = (fwhm / _STOFW) ** 2
        sq = math.sqrt(2.0 * sig_p)
        norm = 0.5 * a * b / (a + b)
        half = 0.5 * fwhm
        use_l = eta >= 1e-6
        off = offsets[k]
        p0 = pos[k]
        s = inten[k]
        for j in range(n):
            d = x[i0[k] + j] - p0
            y1 = (a * sig_p + d) / sq
            y2 = (b * sig_p - d) / sq
            ex = -0.5 * d * d / sig_p
            tg = norm * (_hfunc(ex, y1) + _hfunc(ex, y2))
            if use_l:
                z = complex(d, -half)
                ea = _scaled_e1(a * z)
                eb = _scaled_e1(-b * z)
                tl = (norm / math.pi) * 2.0 * (ea - eb).imag
                out[off + j] = s * (eta * tl + (1.0 - eta) * tg)
            else:
                out[off + j] = s * tg


def tof_profile_batch(
    x: FloatArray,
    pos: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    sigma2: FloatArray,
    gamma: FloatArray,
    inten: FloatArray,
    i0: IntArray,
    i1: IntArray,
) -> tuple[FloatArray, IntArray, IntArray]:
    """全ピークの寄与を一括で計算する。

    Returns:
        out: 平坦な寄与配列。ピーク k の寄与は out[offsets[k] : offsets[k] + lengths[k]]
        offsets: 各ピークの開始オフセット
        lengths: 各ピークの窓の長さ(i1 - i0、負なら 0)
    """
    n = len(pos)
    lengths = np.maximum(np.asarray(i1) - np.asarray(i0), 0).astype(np.int64)
    offsets = np.zeros(n, dtype=np.int64)
    if n > 1:
        offsets[1:] = np.cumsum(lengths)[:-1]
    out = np.zeros(int(lengths.sum()), dtype=np.float64)
    if n == 0 or out.size == 0:
        return out, offsets, lengths
    _tof_batch_kernel(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(pos, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(beta, dtype=np.float64),
        np.ascontiguousarray(sigma2, dtype=np.float64),
        np.ascontiguousarray(gamma, dtype=np.float64),
        np.ascontiguousarray(inten, dtype=np.float64),
        np.ascontiguousarray(i0, dtype=np.int64),
        np.ascontiguousarray(np.asarray(i0) + lengths, dtype=np.int64),
        offsets,
        out,
    )
    return out, offsets, lengths


def scaled_exp1(w: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """e^w E₁(w) の配列版(検証用)。"""
    wa = np.ascontiguousarray(w, dtype=np.complex128).ravel()
    return _scaled_e1_array(wa).reshape(np.shape(w))


@nb.njit(cache=True)
def _scaled_e1_array(w: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    out = np.empty_like(w)
    for i in range(w.shape[0]):
        out[i] = _scaled_e1(w[i])
    return out


__all__ = ["scaled_exp1", "tof_profile_batch"]
