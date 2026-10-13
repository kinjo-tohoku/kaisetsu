"""Thompson-Cox-Hastings 型 pseudo-Voigt 関数。

ガウス成分の FWHM Γ_G とローレンツ成分の FWHM Γ_L から、全体の FWHM Γ と混合比 η を

    Γ^5 = Γ_G^5 + 2.69269 Γ_G^4 Γ_L + 2.42843 Γ_G^3 Γ_L^2 + 4.47163 Γ_G^2 Γ_L^3
          + 0.07842 Γ_G Γ_L^4 + Γ_L^5
    η = 1.36603 q - 0.47719 q^2 + 0.11116 q^3,  q = Γ_L / Γ

で決め、pV(x) = η L(x; Γ) + (1 - η) G(x; Γ) とする(面積 1 に規格化)。

出典:
    P. Thompson, D. E. Cox & J. B. Hastings, J. Appl. Cryst. 20, 79-83 (1987)
    (係数は GSAS/GSAS-II と同じ)
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: FWHM / σ (= 2 sqrt(2 ln 2))
SQRT8LN2 = 2.3548200450309493
_SQRT2PI = 2.5066282746310002
_MIN_FWHM = 1.0e-6


@nb.njit(cache=True)
def tch_fwhm_eta(fwhm_g: float, fwhm_l: float) -> tuple[float, float]:
    """TCH の全 FWHM と混合比 η を返す。"""
    g = max(fwhm_g, 0.0)
    lo = max(fwhm_l, 0.0)
    s = (
        g**5
        + 2.69269 * g**4 * lo
        + 2.42843 * g**3 * lo**2
        + 4.47163 * g**2 * lo**3
        + 0.07842 * g * lo**4
        + lo**5
    )
    fwhm = s**0.2 if s > 0.0 else _MIN_FWHM
    fwhm = max(fwhm, _MIN_FWHM)
    q = lo / fwhm
    eta = 1.36603 * q - 0.47719 * q * q + 0.11116 * q * q * q
    return fwhm, eta


@nb.njit(cache=True)
def pseudo_voigt_point(dx: float, fwhm: float, eta: float) -> float:
    """面積 1 の pseudo-Voigt の 1 点の値。"""
    sig = fwhm / SQRT8LN2
    ex = -0.5 * dx * dx / (sig * sig)
    if ex < -700.0:
        gauss = 0.0
    else:
        gauss = math.exp(ex) / (sig * _SQRT2PI)
    lor = (2.0 / (math.pi * fwhm)) / (1.0 + 4.0 * dx * dx / (fwhm * fwhm))
    return eta * lor + (1.0 - eta) * gauss


@nb.njit(cache=True)
def _pseudo_voigt_array(dx: FloatArray, fwhm_g: float, fwhm_l: float) -> FloatArray:
    fwhm, eta = tch_fwhm_eta(fwhm_g, fwhm_l)
    out = np.empty_like(dx)
    for i in range(dx.shape[0]):
        out[i] = pseudo_voigt_point(dx[i], fwhm, eta)
    return out


def pseudo_voigt(dx: npt.ArrayLike, fwhm_g: float, fwhm_l: float) -> FloatArray:
    """pseudo-Voigt(面積 1)。dx はピーク中心からの距離の配列。単位は fwhm と同じ。"""
    arr = np.ascontiguousarray(np.asarray(dx, dtype=np.float64).ravel())
    return np.asarray(_pseudo_voigt_array(arr, float(fwhm_g), float(fwhm_l))).reshape(np.shape(dx))


def pseudo_voigt_fwhm(fwhm_g: float, fwhm_l: float) -> float:
    """全 FWHM だけを返す。"""
    return float(tch_fwhm_eta(float(fwhm_g), float(fwhm_l))[0])


def tch_fwhm_array(fwhm_g: npt.ArrayLike, fwhm_l: npt.ArrayLike) -> FloatArray:
    """TCH の全 FWHM を配列で返す(`tch_fwhm_eta` と同じ式の NumPy 版)。"""
    g = np.maximum(np.asarray(fwhm_g, dtype=np.float64), 0.0)
    lo = np.maximum(np.asarray(fwhm_l, dtype=np.float64), 0.0)
    s = (
        g**5
        + 2.69269 * g**4 * lo
        + 2.42843 * g**3 * lo**2
        + 4.47163 * g**2 * lo**3
        + 0.07842 * g * lo**4
        + lo**5
    )
    return np.asarray(np.maximum(np.where(s > 0.0, s, _MIN_FWHM**5) ** 0.2, _MIN_FWHM))
