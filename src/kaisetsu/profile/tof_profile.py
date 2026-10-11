"""TOF 中性子のピーク形状: 擬 Voigt と back-to-back 指数関数の畳み込み(GSAS の type 3 相当)。

パルス中性子源のピークは、モデレータからの放出時間分布(立ち上がり α、減衰 β の 2 つの指数関数)
と、分解能による対称な広がり(擬 Voigt)の畳み込みで表す:

    h(t) = N [exp(α t) (t < 0), exp(-β t) (t > 0)],   N = αβ / (α + β)
    P(Δ) = η (L ⊗ h)(Δ) + (1 - η) (G ⊗ h)(Δ)

閉形式(Von Dreele, Jorgensen & Windsor 1982; Thompson, Cox & Hastings 1987):
    G ⊗ h = N/2 · exp(-Δ²/2σ²) [erfcx(y₁) + erfcx(y₂)],  y₁ = (ασ² + Δ)/(σ√2), y₂ = (βσ² - Δ)/(σ√2)
    L ⊗ h = N/π · Im[ e^{αz} E₁(αz) - e^{-βz} E₁(-βz) ],  z = Δ - iΓ/2(Γ: ローレンツ FWHM)
擬 Voigt の混合は TCH 流(両成分の FWHM を共通にし、η と FWHM を σ, γ から決める)。
GSAS / GSAS-II の EPSVOIGT と同じ扱い(そちらは複素指数積分を自前の近似で計算する)。

単位: Δ, σ, Γ は μs、α, β は 1/μs。
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.special import erfc, erfcx, exp1

from kaisetsu.profile.pseudo_voigt import tch_fwhm_eta

FloatArray = npt.NDArray[np.float64]

_SQ2 = math.sqrt(2.0)
_STOFW = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _exp_e1(w: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """e^w E₁(w) をオーバーフローなしで計算する(|w| が大きいところは漸近展開)。"""
    out = np.empty_like(w)
    big = np.abs(w) > 500.0
    small = ~big
    if np.any(small):
        out[small] = np.exp(w[small]) * exp1(w[small])
    if np.any(big):
        ws = w[big]
        # e^w E₁(w) ~ 1/w (1 - 1/w + 2/w² - 6/w³)
        inv = 1.0 / ws
        out[big] = inv * (1.0 - inv * (1.0 - inv * (2.0 - 6.0 * inv)))
    return out


def _h(ex: FloatArray, y: FloatArray) -> FloatArray:
    """exp(ex + y²) erfc(y) を桁あふれなく計算する(GSAS の HFUNC に相当)。

    y > -3 では exp(ex) erfcx(y)、y が大きく負のところでは exp(ex + y²) erfc(y)。
    """
    out = np.empty_like(ex)
    neg = y < -3.0
    pos = ~neg
    if np.any(pos):
        out[pos] = np.exp(ex[pos]) * erfcx(y[pos])
    if np.any(neg):
        out[neg] = np.exp(ex[neg] + y[neg] ** 2) * erfc(y[neg])
    return out


def tof_peak(
    dt: npt.ArrayLike, alpha: float, beta: float, sigma2: float, gamma: float
) -> FloatArray:
    """面積 1 の TOF ピーク形状 P(Δ)。

    Args:
        dt: ピーク中心からのずれ Δ [μs]
        alpha, beta: 指数関数の立ち上がり・減衰 [1/μs]
        sigma2: ガウス分散 σ² [μs²]
        gamma: ローレンツ FWHM [μs]
    """
    d = np.asarray(dt, dtype=np.float64)
    alpha = max(float(alpha), 1e-6)
    beta = max(float(beta), 1e-6)
    fwhm_g = _STOFW * math.sqrt(max(float(sigma2), 1e-10))
    fwhm, eta = tch_fwhm_eta(fwhm_g, max(float(gamma), 1e-5))
    # TCH: 両成分とも共通の FWHM
    sig_p = (fwhm / _STOFW) ** 2
    sq = math.sqrt(2.0 * sig_p)
    norm = 0.5 * alpha * beta / (alpha + beta)
    y1 = (alpha * sig_p + d) / sq
    y2 = (beta * sig_p - d) / sq
    ex = -0.5 * d * d / sig_p
    tg = norm * (_h(ex, y1) + _h(ex, y2))
    if eta < 1e-6:
        # ローレンツ成分が無い(X = Y = Z = 0 の装置)ときは複素指数積分を省く(計算時間の大半)
        return np.asarray(tg)
    half = 0.5 * fwhm
    z = d - 1j * half
    ea = _exp_e1(np.asarray(alpha * z, dtype=np.complex128))
    eb = _exp_e1(np.asarray(-beta * z, dtype=np.complex128))
    tl = (norm / math.pi) * 2.0 * np.imag(np.subtract(ea, eb))
    return np.asarray(eta * tl + (1.0 - eta) * tg)


def tof_peak_fwhm(alpha: float, beta: float, sigma2: float, gamma: float) -> float:
    """目安の全幅 [μs](対称部分の TCH FWHM + 指数関数の幅)。窓の決定に使う。"""
    fwhm_g = _STOFW * math.sqrt(max(sigma2, 1e-10))
    fwhm, _ = tch_fwhm_eta(fwhm_g, max(gamma, 1e-5))
    return float(fwhm + math.log(2.0) / max(alpha, 1e-6) + math.log(2.0) / max(beta, 1e-6))


__all__ = ["tof_peak", "tof_peak_fwhm"]
