"""I(Q) → G(r) 変換の自己無撞着テスト。

Ni のナノ粒子(球、直径 24 Å)の Debye 散乱式で I(Q) を作り、緩やかな背景(Compton 散乱や
多重散乱の代わり)とスケールを掛け、容器の寄与を足した"測定"データから G(r) を復元し、
同じ構造・粒子径の実空間計算と比べる。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.pdf import PDFPhase, TransformSettings, calculate_pdf, parse_composition, transform
from kaisetsu.pdf.transform import (
    average_scattering,
    q_to_two_theta,
    sine_transform,
    two_theta_to_q,
)
from kaisetsu.scattering.xray import XrayScatteringFactors

A_NI = 3.524


def _nanoparticle(diameter: float) -> np.ndarray:
    """fcc Ni の球状粒子の直交座標 (N, 3)。"""
    n = math.ceil(diameter / A_NI) + 1
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.stack(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * A_NI
    return pos[np.linalg.norm(pos, axis=1) <= diameter / 2]


def _debye_iq(q: np.ndarray, pos: np.ndarray, uiso: float) -> np.ndarray:
    """I(Q) = N f² + 2 f² Σ_{i<j} sin(Q r_ij)/(Q r_ij) × exp(-Q² (U_i+U_j)/2)(1 原子あたり)。"""
    n = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    d = d[np.triu_indices(n, 1)]
    hist, edges = np.histogram(d, bins=np.arange(0.0, d.max() + 0.002, 0.001))
    centers = 0.5 * (edges[1:] + edges[:-1])
    keep = hist > 0
    centers, counts = centers[keep], hist[keep]
    f = XrayScatteringFactors().f0("Ni", q / (4 * math.pi))
    dw = np.exp(-(q**2) * (2 * uiso) / 2.0)  # 原子対の熱振動(等方、相関なし)
    qr = np.outer(q, centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(qr > 0, np.sin(qr) / qr, 1.0)
    return f**2 * (1.0 + 2.0 * dw * (sinc @ counts) / n)


def test_parse_composition() -> None:
    assert parse_composition("Ni") == {"Ni": 1.0}
    assert parse_composition("Na2Ca3Al2F14") == {"Na": 2, "Ca": 3, "Al": 2, "F": 14}
    assert parse_composition("La0.5Sr0.5MnO3") == pytest.approx(
        {"La": 0.5, "Sr": 0.5, "Mn": 1, "O": 3}
    )
    with pytest.raises(ValueError):
        parse_composition("ni2")


def test_q_two_theta_roundtrip() -> None:
    tt = np.array([5.0, 30.0, 120.0])
    q = two_theta_to_q(tt, 0.2)
    assert np.allclose(q_to_two_theta(q, 0.2), tt)
    assert q[1] == pytest.approx(4 * math.pi * math.sin(math.radians(15)) / 0.2)


def test_average_scattering_neutron_and_xray() -> None:
    q = np.array([0.0, 5.0])
    f_sq, f2 = average_scattering(q, {"Ni": 1.0}, "neutron")
    assert np.allclose(f_sq, 10.3**2, rtol=1e-3) and np.allclose(f2, 10.3**2, rtol=1e-3)
    f_sq, f2 = average_scattering(q, {"Ca": 1, "F": 2}, "xray")
    assert f2[0] > f_sq[0]  # <f²> ≥ <f>²(Q = 0 で 20, 9 の電子数)
    assert f_sq[0] == pytest.approx(((20 + 2 * 9) / 3) ** 2, rel=0.01)


def test_sine_transform_of_single_peak() -> None:
    """F(Q) = sin(Q r0)/... の逆: ガウスピーク 1 本の G(r) を順変換→逆変換で戻せる。"""
    r = np.arange(0.0, 20.0, 0.01)
    g = np.exp(-0.5 * ((r - 5.0) / 0.1) ** 2)
    q = np.arange(0.0, 60.0, 0.01)
    fq = (np.sin(np.outer(q, r)) @ g) * 0.01  # F(Q) = ∫ G(r) sin(Qr) dr
    g_back = sine_transform(q, fq, r, 60.0)
    assert np.allclose(g_back, g, atol=0.01)


@pytest.mark.slow
def test_transform_clean_debye_matches_absolute_scale() -> None:
    """誤差の無い Debye 散乱なら G(r) の絶対値まで 3% 以内で一致する。"""
    pos = _nanoparticle(24.0)
    q = np.arange(0.3, 28.0, 0.01)
    uiso = 0.005
    i_true = _debye_iq(q, pos, uiso)
    settings = TransformSettings(
        composition="Ni", radiation="xray", qmin=0.5, qmax=26.0, rpoly=0.0, rmax=22.0
    )
    res = transform(q, i_true, settings)
    st = Structure(
        Lattice.cubic(A_NI), SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=uiso)],
    )  # fmt: skip
    calc = calculate_pdf(res.r, [PDFPhase(st, spdiameter=24.0)], "xray", qmax=26.0, qmin=0.5)
    r = res.r
    for lo, hi in ((2.2, 2.8), (8.0, 9.0)):
        m = (r > lo) & (r < hi)
        assert res.gr[m].max() == pytest.approx(calc.g_calc[m].max(), rel=0.03)
    assert res.scale == pytest.approx(1.0, abs=0.01)


@pytest.mark.slow
def test_transform_recovers_nanoparticle_pdf() -> None:
    pos = _nanoparticle(24.0)
    assert 500 < len(pos) < 800
    q = np.arange(0.3, 28.0, 0.01)
    uiso = 0.005
    i_true = _debye_iq(q, pos, uiso)
    rng = np.random.default_rng(0)
    # 緩やかな乗法・加法誤差、容器の寄与、スケール、ノイズ
    smooth = 1.0 + 0.15 * np.exp(-(((q - 8.0) / 10.0) ** 2)) + 0.02 * q
    compton_like = 80.0 * (1 - np.exp(-((q / 6.0) ** 2)))
    container = 30.0 * np.exp(-q / 10.0) + 5.0
    i_meas = 3.7 * (i_true * smooth + compton_like) + 1.2 * container
    i_meas = i_meas * (1 + rng.normal(0, 0.002, len(q)))
    settings = TransformSettings(
        composition="Ni", radiation="xray", qmin=0.5, qmax=26.0, rpoly=0.9,
        bgscale=1.2, rmin=0.0, rmax=22.0, rstep=0.01,
    )  # fmt: skip
    res = transform(q, i_meas, settings, background=container)
    assert np.all(np.isfinite(res.gr))

    st = Structure(
        Lattice.cubic(A_NI), SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=uiso)],
    )  # fmt: skip
    calc = calculate_pdf(res.r, [PDFPhase(st, spdiameter=24.0)], "xray", qmax=26.0, qmin=0.5)
    r = res.r
    sel = (r > 1.8) & (r < 12.0)
    # 形で比較する。絶対スケールは、差し引けなかった Compton 散乱などの分だけ規格化係数が
    # 小さく出るので一致しない(PDFgetX3 と同じ。フィットのスケール因子が吸収する)
    corr = np.corrcoef(res.gr[sel], calc.g_calc[sel])[0, 1]
    assert corr > 0.97
    first = (r > 2.2) & (r < 2.8)
    far = (r > 8.0) & (r < 9.0)
    r_obs = r[first][np.argmax(res.gr[first])]
    r_calc = r[first][np.argmax(calc.g_calc[first])]
    assert abs(r_obs - r_calc) < 0.02
    ratio_first = res.gr[first].max() / calc.g_calc[first].max()
    ratio_far = res.gr[far].max() / calc.g_calc[far].max()
    assert 0.1 < ratio_first < 1.2
    assert ratio_far == pytest.approx(ratio_first, rel=0.15)
    assert any("規格化係数" in line for line in res.log)
