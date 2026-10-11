"""TOF ピーク形状: 閉形式が数値畳み込みと一致し、面積 1、極限で擬 Voigt に戻る。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kaisetsu.profile.pseudo_voigt import pseudo_voigt, tch_fwhm_eta
from kaisetsu.profile.tof_profile import tof_peak, tof_peak_fwhm

_STOFW = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _numerical(
    dt: np.ndarray, alpha: float, beta: float, sigma2: float, gamma: float
) -> np.ndarray:
    """細かいグリッドでの数値畳み込み(TCH 擬 Voigt ⊗ back-to-back 指数)。"""
    fwhm_g = _STOFW * math.sqrt(sigma2)
    fwhm, _eta = tch_fwhm_eta(fwhm_g, gamma)
    span = 60.0 * fwhm + 40.0 / min(alpha, beta)
    h = fwhm / 200.0
    t = np.arange(-span, span, h)
    pv = pseudo_voigt(t, fwhm_g, gamma)
    with np.errstate(over="ignore"):
        kern = np.where(t < 0, np.exp(alpha * t), np.exp(-beta * t)) * alpha * beta / (alpha + beta)
    conv = np.convolve(pv, kern, mode="same") * h
    return np.interp(dt, t, conv)


@pytest.mark.parametrize(
    "alpha,beta,sigma2,gamma",
    [
        (0.5, 0.05, 100.0, 5.0),
        (0.2, 0.02, 400.0, 20.0),
        (1.0, 0.3, 50.0, 0.5),
        (0.05, 0.01, 900.0, 40.0),
    ],
)
def test_closed_form_matches_numerical(
    alpha: float, beta: float, sigma2: float, gamma: float
) -> None:
    dt = np.linspace(-200.0, 600.0, 801)
    y = tof_peak(dt, alpha, beta, sigma2, gamma)
    y_num = _numerical(dt, alpha, beta, sigma2, gamma)
    assert np.max(np.abs(y - y_num)) < 1e-2 * np.max(y_num)  # 数値畳み込み側の離散化誤差程度
    # 面積 1
    wide = np.linspace(-2000.0, 6000.0, 80001)
    # 面積 1(有限範囲なのでローレンツの裾 ≈ γ/(π R) だけ欠ける)
    area = np.trapezoid(tof_peak(wide, alpha, beta, sigma2, gamma), wide)
    assert area == pytest.approx(1.0, abs=gamma / (math.pi * 2000.0) + 1e-3)
    # 非対称: 減衰側(β 側、Δ > 0)に裾を引く
    assert y[dt > 100].sum() > y[dt < -100].sum()


def test_limits_to_pseudo_voigt() -> None:
    dt = np.linspace(-100.0, 100.0, 2001)
    sigma2, gamma = 100.0, 5.0
    fwhm_g = _STOFW * math.sqrt(sigma2)
    fwhm, _eta = tch_fwhm_eta(fwhm_g, gamma)
    y = tof_peak(dt, 50.0, 50.0, sigma2, gamma)  # α, β → 大: 指数関数が δ 関数に近づく
    pv = pseudo_voigt(dt, fwhm_g, gamma)
    assert np.max(np.abs(y - pv)) < 0.02 * pv.max()
    assert tof_peak_fwhm(0.5, 0.05, 100.0, 5.0) > fwhm
