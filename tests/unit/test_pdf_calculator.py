"""PDF 計算器の単体テスト: 配位数、基線、異方性との等価性、終端効果、形状関数、データ読み込み。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.crystal.cif import read_structure
from kaisetsu.pdf import PDFPhase, calculate_pdf, pair_distribution, read_pdf_data
from kaisetsu.pdf.calculator import bandlimit, scattering_lengths, spherical_shape_factor


def _ni(u_iso: float = 0.005) -> Structure:
    st = Structure(
        Lattice.cubic(3.524),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni1", "Ni", [0, 0, 0], u_iso=u_iso)],
        name="Ni",
    )
    return st


def test_pairs_fcc_shells() -> None:
    """fcc の近接殻: 12 個 @ a/√2、6 個 @ a、24 個 @ a√(3/2)。"""
    st = _ni()
    pl = pair_distribution(st, 6.0, "xray")
    a = 3.524
    assert pl.n_atoms == pytest.approx(4.0)
    assert pl.rho0 == pytest.approx(4.0 / a**3)
    # まとめた後の対は (r, σ'²) ごとに 1 行。重みは起点 4 原子分の合計(= 4 × 配位数)
    shells = {round(float(d), 3): float(w) for d, w in zip(pl.distance, pl.weight, strict=True)}
    assert shells[round(a / math.sqrt(2), 3)] == pytest.approx(4 * 12)
    assert shells[round(a, 3)] == pytest.approx(4 * 6)
    assert shells[round(a * math.sqrt(1.5), 3)] == pytest.approx(4 * 24)
    assert np.allclose(pl.sigma2_prime, 2 * 0.005)


def test_coordination_number_and_baseline() -> None:
    st = _ni()
    r = np.arange(0.01, 12.0, 0.005)
    res = calculate_pdf(r, [PDFPhase(st)], "xray")
    rho0 = 4.0 / 3.524**3
    # 第一ピークより内側は -4π ρ0 r
    inner = r < 1.8
    assert np.allclose(res.g_calc[inner], -4 * math.pi * rho0 * r[inner], atol=1e-6)
    # RDF = r (G + 4π ρ0 r) の第一ピークの面積 = 配位数 12
    rdf = r * (res.g_calc + 4 * math.pi * rho0 * r)
    m = (r > 1.9) & (r < 3.0)
    assert np.trapezoid(rdf[m], r[m]) == pytest.approx(12.0, abs=0.02)
    # RDF のピーク位置は r_ij(G(r) のピークは 1/r と基線の分だけ僅かに低 r 側に寄る)
    k = np.argmax(np.where(m, rdf, -np.inf))
    assert r[k] == pytest.approx(3.524 / math.sqrt(2), abs=0.006)


def test_anisotropic_isotropic_equivalence() -> None:
    iso = _ni(0.006)
    aniso = Structure(
        Lattice.cubic(3.524),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni1", "Ni", [0, 0, 0], u_aniso=[0.006, 0.006, 0.006, 0, 0, 0])],
    )
    r = np.arange(1.0, 10.0, 0.01)
    g1 = calculate_pdf(r, [PDFPhase(iso)], "xray").g_calc
    g2 = calculate_pdf(r, [PDFPhase(aniso)], "xray").g_calc
    assert np.allclose(g1, g2, atol=1e-8)


def test_delta2_sharpens_near_peaks() -> None:
    """δ2 > 0 は近距離のピークを鋭くし、遠距離ではほとんど変えない。"""
    st = _ni()
    r = np.arange(1.0, 15.0, 0.01)
    g0 = calculate_pdf(r, [PDFPhase(st)], "xray").g_calc
    g1 = calculate_pdf(r, [PDFPhase(st, delta2=2.0)], "xray").g_calc
    first = (r > 2.3) & (r < 2.7)
    far = (r > 13.5) & (r < 14.5)
    assert g1[first].max() > g0[first].max() * 1.1
    assert abs(g1[far].max() - g0[far].max()) < 0.05 * g0[far].max()


def test_scattering_weights_two_species() -> None:
    """2 元素では重みが b_i b_j / <b>² になる(NaCl 型、X 線は電子数)。"""
    st = Structure(
        Lattice.cubic(5.64),
        SpaceGroup.from_symbol("F m -3 m"),
        [
            AtomSite("Na", "Na1+", [0, 0, 0], u_iso=0.01),
            AtomSite("Cl", "Cl1-", [0.5, 0.5, 0.5], u_iso=0.01),
        ],
    )
    pl = pair_distribution(st, 3.0, "xray")
    b_na, b_cl = (float(v) for v in scattering_lengths(("Na1+", "Cl1-"), "xray"))
    assert b_na == pytest.approx(10.0, abs=0.05) and b_cl == pytest.approx(18.0, abs=0.05)
    b_mean = (b_na + b_cl) / 2
    # 最近接 Na-Cl(2.82 Å): 4 Na × 6 Cl + 4 Cl × 6 Na = 48 対、重み b_Na b_Cl / <b>²
    k = np.argmin(np.abs(pl.distance - 2.82))
    assert pl.weight[k] == pytest.approx(48 * b_na * b_cl / b_mean**2, rel=1e-6)


def test_bandlimit_and_envelopes() -> None:
    st = _ni()
    r = np.arange(0.01, 10.0, 0.01)
    g_inf = calculate_pdf(r, [PDFPhase(st)], "xray").g_calc
    g_q = calculate_pdf(r, [PDFPhase(st)], "xray", qmax=25.0).g_calc
    # 終端効果で低 r にリップルが出る(基線からずれる)が、ピーク高さの差は 10% 以内
    rho0 = 4.0 / 3.524**3
    low = r < 1.5
    assert np.max(np.abs(g_q[low] + 4 * math.pi * rho0 * r[low])) > 0.01
    first = (r > 2.3) & (r < 2.7)
    assert g_q[first].max() == pytest.approx(g_inf[first].max(), rel=0.1)
    # Q_damp は遠距離を減衰させる
    g_d = calculate_pdf(r, [PDFPhase(st)], "xray", qdamp=0.1).g_calc
    far = (r > 8.5) & (r < 9.5)
    assert g_d[far].max() < 0.8 * g_inf[far].max()
    # 球状粒子: 直径の外では 0、内側では f(r,d) 倍
    g_s = calculate_pdf(r, [PDFPhase(st, spdiameter=8.0)], "xray").g_calc
    assert np.all(g_s[r > 8.05] == 0.0)
    f = spherical_shape_factor(r, 8.0)
    assert np.allclose(g_s[first], (g_inf * f)[first], atol=1e-8)
    # bandlimit は帯域内の正弦波を保つ
    x = np.arange(0, 2000) * 0.01
    s = np.sin(5.0 * x)
    assert np.allclose(bandlimit(s, 0.01, 0.0, 20.0)[300:-300], s[300:-300], atol=2e-2)
    # 帯域外の正弦波は消える
    s2 = np.sin(30.0 * x)
    assert np.max(np.abs(bandlimit(s2, 0.01, 0.0, 20.0)[300:-300])) < 5e-2


def test_scale_and_multi_phase() -> None:
    st = _ni()
    r = np.arange(1.0, 8.0, 0.02)
    g1 = calculate_pdf(r, [PDFPhase(st)], "xray").g_calc
    g2 = calculate_pdf(r, [PDFPhase(st, scale=0.5), PDFPhase(st, scale=0.5)], "xray", scale=2.0)
    assert np.allclose(g2.g_calc, 2.0 * g1)
    assert len(g2.g_phases) == 2


def test_read_gr_files(data_dir: Path) -> None:
    x = read_pdf_data(data_dir / "pdf" / "Ni-xray.gr")
    assert len(x) > 1000 and x.r[0] <= 0.01
    assert x.dg is not None and np.all(x.dg > 0)
    assert x.metadata.get("qmax") == pytest.approx(40.0)
    n = read_pdf_data(data_dir / "pdf" / "Ni-neutron.gr")
    assert len(n) > 1000 and n.metadata.get("qmax") == pytest.approx(27.0)


def test_read_plain_columns(tmp_path: Path) -> None:
    p = tmp_path / "a.gr"
    r = np.arange(0.0, 5.0, 0.1)
    g = np.sin(r)
    p.write_text(
        "# comment\nr G dG\n"
        + "\n".join(f"{a:.3f} {b:.5f} 0.1" for a, b in zip(r, g, strict=True)),
        encoding="utf-8",
    )
    d = read_pdf_data(p)
    assert len(d) == len(r)
    assert d.dg is not None and np.allclose(d.dg, 0.1)
    assert np.allclose(d.weight, 100.0)


def test_ni_cif_pairs(data_dir: Path) -> None:
    st = read_structure(data_dir / "pdf" / "Ni.cif")
    pl = pair_distribution(st, 5.0, "neutron")
    assert pl.n_atoms == pytest.approx(4.0)
    assert pl.weight[0] == pytest.approx(48.0)
