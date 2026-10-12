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


def axis_cos2(
    hkl: npt.ArrayLike, lattice: Lattice, spacegroup: SpaceGroup, axis: npt.ArrayLike
) -> FloatArray:
    """反射の散乱ベクトルと軸(逆格子方向 hkl)のなす角の cos² を、ラウエ等価反射で平均したもの。

    一軸異方的な結晶子サイズ広がり(`PhaseContribution.inverse_size`)に使う。
    軸が対称軸なら等価反射はすべて同じ角をなすので平均は不要だが、一般の軸でも使えるようにする。
    """
    h = np.atleast_2d(np.asarray(hkl, dtype=np.int64))
    gs = lattice.reciprocal_metric_tensor
    ax = np.asarray(axis, dtype=np.float64)
    ax_len = math.sqrt(float(ax @ gs @ ax))
    if ax_len == 0.0:
        raise ValueError("サイズの軸が零ベクトルです")
    out = np.zeros(len(h))
    for i, hh in enumerate(h):
        eq = spacegroup.laue_equivalents(hh).astype(np.float64)
        dots = eq @ gs @ ax
        lens = np.sqrt(np.einsum("ij,jk,ik->i", eq, gs, eq))
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_a = np.where(lens > 0, dots / (lens * ax_len), 0.0)
        out[i] = float(np.mean(np.clip(cos_a, -1.0, 1.0) ** 2))
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


def cylinder_absorption_array(mu_r: npt.ArrayLike, two_theta: npt.ArrayLike) -> FloatArray:
    """μR が反射ごとに違う場合(TOF: μR ∝ λ)の円筒試料の透過率(ベクトル化)。"""
    mr, tt = np.broadcast_arrays(
        np.asarray(mu_r, dtype=np.float64), np.asarray(two_theta, dtype=np.float64)
    )
    sth2 = np.sin(np.radians(tt / 2.0)) ** 2
    # μR ≤ 3 の式
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
    low = np.exp(-t0 * mr - t1 * mr**2 - t2 * mr**3 - t3 * mr**4)
    # μR > 3 の式
    u1 = 1.433902 + 11.07504 * sth2 - 8.77629 * sth2 * sth2 + 10.02088 * sth2**3 - 3.36778 * sth2**4
    u2 = (0.013869 - 0.01249 * sth2) * np.exp(3.27094 * sth2) + (0.337894 + 13.77317 * sth2) / (
        1.0 + 11.53544 * sth2
    ) ** 1.555039
    u3 = 1.933433 / (1.0 + 23.12967 * sth2) ** 1.686715 - 0.13576 * np.sqrt(sth2) + 1.163198
    u4 = 0.044365 - 0.04259 / (1.0 + 0.41051 * sth2) ** 148.4202
    with np.errstate(invalid="ignore", divide="ignore"):
        high = ((u1 - u4) / (1.0 + u2 * np.maximum(mr - 3.0, 0.0)) ** u3 + u4) / 100.0
    out = np.where(mr > 3.0, high, low)
    return np.asarray(np.where(mr > 0.0, out, 1.0))


_SABINE_COEF = np.array([-0.5, 0.25, -0.10416667, 0.036458333, -0.0109375, 2.8497409e-3])


def sabine_extinction(
    f2: npt.ArrayLike,
    wavelength: npt.ArrayLike,
    volume: float,
    two_theta: npt.ArrayLike,
    extinction: float,
    is_xray: bool,
) -> FloatArray:
    """Sabine の粉末消衰補正(GSAS-II と同じ定式化)。

    x = E · F² (λ/V)²(F は 10⁻¹² cm 単位。X 線は F[e] × r_e と偏光 (1 + cos²2θ)/2 を含める)
    E_B = 1/√(1 + x)、E_L = 1 − x/2 + x²/4 − …(x ≤ 1)、E_L ≈ √(2/π)(1 − 1/(8x))/√x(x > 1)
    補正 = E_B sin²θ + E_L cos²θ。E の単位は GSAS-II の Extinction と同じ(数値を揃えてある)。

    Args:
        f2: |F|²(X 線は電子²、中性子は fm²。内部で (10⁻¹² cm)² に換算する)
        wavelength: 反射ごとの波長 [Å](CW ではスカラー)
        volume: 単位胞体積 [Å³]
        two_theta: 反射の 2θ [deg](TOF ではバンクの 2θ)
        extinction: 係数 E(0 で補正なし)
        is_xray: X 線なら偏光と r_e² を含める

    出典: T. M. Sabine, Int. Tables Vol. C, 6.4; T. M. Sabine, R. B. Von Dreele & J. E. Jørgensen,
          Acta Cryst. A44, 374-379 (1988); GSAS-II GSASIIstrMath.GetPwdrExt。
    """
    f2a = np.asarray(f2, dtype=np.float64)
    tt = np.asarray(two_theta, dtype=np.float64)
    if extinction == 0.0:
        return np.ones(np.broadcast(f2a, tt).shape)
    lam = np.asarray(wavelength, dtype=np.float64)
    sth2 = np.sin(np.radians(tt / 2.0)) ** 2
    c2th = 1.0 - 2.0 * sth2
    if is_xray:
        f2_cm = f2a * 0.079411 * (1.0 + c2th**2) / 2.0  # r_e² = 0.0794 (10⁻¹² cm)²
    else:
        f2_cm = f2a / 100.0  # fm² → (10⁻¹² cm)²
    x = extinction * f2_cm * (lam / volume) ** 2
    x = np.broadcast_to(x, np.broadcast(x, sth2).shape).copy()
    exb = np.where(x > -1.0, 1.0 / np.sqrt(np.maximum(1.0 + x, 1e-12)), 1.0)
    exl = np.ones_like(x)
    small = (x > 0.0) & (x <= 1.0)
    if np.any(small):
        xs = x[small]
        exl[small] = 1.0 + sum(c * xs ** (i + 1) for i, c in enumerate(_SABINE_COEF))
    big = x > 1.0
    if np.any(big):
        xb = x[big]
        exl[big] = math.sqrt(2.0 / math.pi) * (1.0 - 0.125 / xb) / np.sqrt(xb)
    return np.asarray(exb * sth2 + exl * (1.0 - sth2))


def surface_roughness_suortti(a: float, b: float, two_theta: npt.ArrayLike) -> FloatArray:
    """Suortti の表面粗さ補正(Bragg-Brentano)。

    f(θ) = [a + (1 − a) exp(−b / sinθ)] / [a + (1 − a) exp(−b)]
    a = 1(または a = b = 0)で補正なし。a が小さいほど低角の強度が落ちる。
    b は角度依存の鋭さで、a と強く相関するので通常は 1 に固定して a だけ動かす。

    出典: P. Suortti, J. Appl. Cryst. 5, 325-331 (1972)。GSAS-II の SurfRoughA/B と同じ式
    """
    tt = np.asarray(two_theta, dtype=np.float64)
    if a == 1.0 or (a == 0.0 and b == 0.0):
        return np.ones_like(tt)
    sth = np.sin(np.radians(tt / 2.0))
    t1 = np.exp(-b / sth)
    t2 = a + (1.0 - a) * math.exp(-b)
    return np.asarray((a + (1.0 - a) * t1) / t2)


def bragg_brentano_shift(
    two_theta: npt.ArrayLike,
    displacement_mm: float,
    radius_mm: float,
    transparency_mm: float = 0.0,
) -> FloatArray:
    """Bragg-Brentano 光学系の試料変位・透過によるピークシフト Δ2θ [deg]。

    変位 s(試料面が焦点円の中心からずれた量、mm): Δ2θ = -(2 s / R) cosθ
    透過(有効侵入深さ t = 1/μ_eff [mm]): Δ2θ = -t sin 2θ / (2 R)
        t = 0 で補正なし。μ_eff でなく t を変数にするのは、μ_eff → 0 で発散しないようにするため。

    出典: Klug & Alexander, X-ray Diffraction Procedures, 2nd ed. (1974) Sec. 5-3.4;
        Wilson, Mathematical Theory of X-ray Powder Diffractometry (1963)
    """
    tt = np.asarray(two_theta, dtype=np.float64)
    th = np.radians(tt / 2.0)
    shift = -(2.0 * displacement_mm / radius_mm) * np.cos(th)
    if transparency_mm != 0.0:
        shift = shift - transparency_mm * np.sin(2.0 * th) / (2.0 * radius_mm)
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


def brindley_factor(x: npt.ArrayLike, n_points: int = 200) -> FloatArray:
    """Brindley の微小吸収因子 τ(x)、x = (μ_p - μ̄) R(無次元)。

    球状粒子を仮定し、散乱点を通るビーム方向の弦長 ℓ(ρ) = 2R sqrt(1 - (ρ/R)²) に沿った
    吸収差 exp(-(μ_p - μ̄) ℓ) を粒子体積で平均する:

        τ(x) = 3 ∫_0^1 u sqrt(1 - u²) exp(-2 x sqrt(1 - u²)) du

    x > 0(周囲より吸収が強い相)で τ < 1 となり、その相の見かけの強度が下がる。
    Brindley の表(2θ 依存を含む)との差は数 % 以内。

    出典: G. W. Brindley, Phil. Mag. 36, 347-369 (1945);
          J. C. Taylor & C. E. Matulis, J. Appl. Cryst. 24, 14-17 (1991)。
    """
    xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
    # u = sin t と置換して端点の特異性を無くす: 3 ∫_0^{π/2} sin t cos²t exp(-2 x cos t) dt
    nodes, weights = np.polynomial.legendre.leggauss(min(max(n_points // 4, 16), 128))
    t = 0.25 * math.pi * (nodes + 1.0)  # [0, π/2]
    wt = 0.25 * math.pi * weights
    ct = np.cos(t)
    integrand = np.sin(t)[None, :] * ct[None, :] ** 2 * np.exp(-2.0 * xa[:, None] * ct[None, :])
    tau = 3.0 * (integrand @ wt)
    return np.asarray(tau.reshape(np.shape(x)))
