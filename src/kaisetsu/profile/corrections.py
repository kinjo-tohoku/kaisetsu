"""強度補正とピーク位置補正(定波長)。

- Lorentz 因子 1/(2 sin^2θ cosθ)
- 偏光因子 (1 - P) cos^2 2θ + P (P = 0.5 で無偏光、P = 1 で散乱面に垂直な完全偏光)
- March-Dollase 選択配向
- 円筒試料の吸収(Debye-Scherrer)
- 表面粗さ(Suortti、Bragg-Brentano)
- 試料変位・透過によるピークシフト

出典:
    - Pecharsky & Zavalij, Ch. 2.10 (LP), 2.11 (選択配向)
    - W. A. Dollase, J. Appl. Cryst. 19, 267-272 (1986)
    - N. N. Lobanov & L. Alte da Veiga, 6th European Powder Diffraction Conf. (1998) 吸収の近似式
      (GSAS-II と同じ係数)
    - P. Suortti, J. Appl. Cryst. 5, 325-331 (1972)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

FloatArray = npt.NDArray[np.float64]


def lorentz_factor(two_theta: npt.ArrayLike) -> FloatArray:
    """粉末回折の Lorentz 因子 1/(2 sin^2θ cosθ)。"""
    th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
    return np.asarray(1.0 / (2.0 * np.sin(th) ** 2 * np.cos(th)))


def polarization_factor(two_theta: npt.ArrayLike, polarization: float) -> FloatArray:
    """偏光因子 (1 - P) cos^2 2θ + P。

    P = 0.5: 無偏光(モノクロメータ無し)。
    P = 1/(1 + cos^2 2θ_m): 回折ビーム側モノクロメータ(2θ_m はモノクロメータの回折角)。
    P = 1: 散乱面に垂直に完全偏光(放射光の鉛直散乱配置)。
    """
    tt = np.radians(np.asarray(two_theta, dtype=np.float64))
    return np.asarray((1.0 - polarization) * np.cos(tt) ** 2 + polarization)


def polarization_from_monochromator(two_theta_mono: float) -> float:
    """モノクロメータの回折角 2θ_m [deg] から偏光係数 P を求める。"""
    k = math.cos(math.radians(two_theta_mono)) ** 2
    return 1.0 / (1.0 + k)


def march_dollase(
    hkl: npt.ArrayLike,
    lattice: Lattice,
    spacegroup: SpaceGroup,
    axis: npt.ArrayLike,
    r: float,
) -> FloatArray:
    """March-Dollase 選択配向補正(反射ごと)。

    P_h = (1/M) Σ_{等価反射} (r^2 cos^2 α + sin^2 α / r)^(-3/2)
    α は配向軸(逆格子ベクトル)と各等価反射の散乱ベクトルのなす角。
    r = 1 で補正なし、r < 1 で板状(配向軸方向の反射が強まる)。
    """
    h = np.atleast_2d(np.asarray(hkl, dtype=np.int64))
    out = np.ones(len(h))
    if r == 1.0:
        return out
    gs = lattice.reciprocal_metric_tensor
    ax = np.asarray(axis, dtype=np.float64)
    ax_len = math.sqrt(float(ax @ gs @ ax))
    if ax_len == 0.0:
        raise ValueError("選択配向軸が零ベクトルです")
    for i, hh in enumerate(h):
        eq = spacegroup.laue_equivalents(hh).astype(np.float64)
        dots = eq @ gs @ ax
        lens = np.sqrt(np.einsum("ij,jk,ik->i", eq, gs, eq))
        cos_a = np.clip(dots / (lens * ax_len), -1.0, 1.0)
        sin2 = 1.0 - cos_a**2
        out[i] = float(np.mean((r * r * cos_a**2 + sin2 / r) ** -1.5))
    return out


def cylinder_absorption(mu_r: float, two_theta: npt.ArrayLike) -> FloatArray:
    """円筒試料の透過率 A(θ)(Debye-Scherrer、入射ビームが試料全体を照らす場合)。

    μR ≤ 3 と μR > 3 で別の近似式(Lobanov & Alte da Veiga)。μR = 0 で 1。
    """
    tt = np.asarray(two_theta, dtype=np.float64)
    sth2 = np.sin(np.radians(tt / 2.0)) ** 2
    if mu_r <= 0.0:
        return np.ones_like(tt)
    if mu_r <= 3.0:
        t0 = 16.0 / (3.0 * math.pi)
        t1 = (
            (25.99978 - 0.01911 * sth2**0.25) * np.exp(-0.024551 * sth2)
            + 0.109561 * np.sqrt(sth2)
            - 26.04556
        )
        t2 = (
            -0.02489
            - 0.39499 * sth2
            + 1.219077 * sth2**1.5
            - 1.31268 * sth2**2
            + 0.871081 * sth2**2.5
            - 0.2327 * sth2**3
        )
        t3 = 0.003045 + 0.018167 * sth2 - 0.03305 * sth2**2
        return np.asarray(np.exp(-t0 * mu_r - t1 * mu_r**2 - t2 * mu_r**3 - t3 * mu_r**4))
    t1 = 1.433902 + 11.07504 * sth2 - 8.77629 * sth2 * sth2 + 10.02088 * sth2**3 - 3.36778 * sth2**4
    t2 = (0.013869 - 0.01249 * sth2) * np.exp(3.27094 * sth2) + (0.337894 + 13.77317 * sth2) / (
        1.0 + 11.53544 * sth2
    ) ** 1.555039
    t3 = 1.933433 / (1.0 + 23.12967 * sth2) ** 1.686715 - 0.13576 * np.sqrt(sth2) + 1.163198
    t4 = 0.044365 - 0.04259 / (1.0 + 0.41051 * sth2) ** 148.4202
    return np.asarray(((t1 - t4) / (1.0 + t2 * (mu_r - 3.0)) ** t3 + t4) / 100.0)


def surface_roughness_suortti(a: float, b: float, two_theta: npt.ArrayLike) -> FloatArray:
    """Suortti の表面粗さ補正(Bragg-Brentano)。a = b = 0 で 1。"""
    tt = np.asarray(two_theta, dtype=np.float64)
    if a == 0.0 and b == 0.0:
        return np.ones_like(tt)
    sth = np.sin(np.radians(tt / 2.0))
    t1 = np.exp(-b / sth)
    t2 = a + (1.0 - a) * math.exp(-b)
    return np.asarray((a + (1.0 - a) * t1) / t2)


def bragg_brentano_shift(
    two_theta: npt.ArrayLike,
    displacement_mm: float,
    radius_mm: float,
    mu_eff_inv_mm: float = 0.0,
) -> FloatArray:
    """Bragg-Brentano 光学系の試料変位・透過によるピークシフト Δ2θ [deg]。

    変位 s(試料面が焦点円の中心からずれた量、mm): Δ2θ = -(2 s / R) cosθ
    透過(有効線吸収係数 μ_eff [1/mm]): Δ2θ = -sin 2θ / (2 μ_eff R)
    """
    tt = np.asarray(two_theta, dtype=np.float64)
    th = np.radians(tt / 2.0)
    shift = -(2.0 * displacement_mm / radius_mm) * np.cos(th)
    if mu_eff_inv_mm > 0.0:
        shift = shift - np.sin(2.0 * th) / (2.0 * mu_eff_inv_mm * radius_mm)
    return np.asarray(np.degrees(shift))


def debye_scherrer_shift(
    two_theta: npt.ArrayLike,
    displace_x_mm: float,
    displace_y_mm: float,
    radius_mm: float,
) -> FloatArray:
    """Debye-Scherrer 光学系の試料変位によるピークシフト Δ2θ [deg]。

    Δ2θ = -(Δx cos 2θ + Δy sin 2θ) / R(Δx はビーム方向、Δy はそれに垂直)
    """
    tt = np.radians(np.asarray(two_theta, dtype=np.float64))
    return np.asarray(
        np.degrees(-(displace_x_mm * np.cos(tt) + displace_y_mm * np.sin(tt)) / radius_mm)
    )
