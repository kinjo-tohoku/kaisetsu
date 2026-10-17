"""PDF の終端効果まわり(外部レビュー 2026-10-17 精密化 #2、#3、#7)。

- Q_max はデータの上限で切り、フーリエ変換・Lorch 窓・多項式次数・メタデータに同じ値を使う
  (実験室 Cu Kα は 7.9 Å⁻¹ なのに既定の 25 Å⁻¹ のままだと U が 5 倍、scale が 1.9 倍になった)。
- Lorch 窓を掛けた G(r) は、モデル側も同じ窓で帯域制限する(掛けないと U が 2.2 倍)。
- U = 0 のサイトでもピークの面積(配位数)が保存される(ビン積分)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import least_squares

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.pdf import (
    PDFData,
    PDFPhase,
    TransformResult,
    TransformSettings,
    calculate_pdf,
    transform,
)
from kaisetsu.pdf.transform import average_scattering, sine_transform
from kaisetsu.refine.project import Project

A_NI = 3.524
U_NI = 0.006
QDAMP = 0.06
Q_CU = 7.879  # Cu Kα1、2θ = 150°
Q_MO = 17.08  # Mo Kα1、2θ = 150°


def _ni(a: float = A_NI, u: float = U_NI) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni1", "Ni", [0, 0, 0], u_iso=u)],
        name="Ni",
    )


@pytest.fixture(scope="module")
def g_full() -> tuple[np.ndarray, np.ndarray]:
    """終端のない Ni の G(r)(Q_damp で遠方を 0 にしてある)。"""
    rf = np.arange(0.01, 40.0, 0.01)
    g = calculate_pdf(rf, [PDFPhase(_ni())], "xray", qmax=0.0, qdamp=QDAMP).g_calc
    return rf, g


def _fq(q: np.ndarray, g_full: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """F(Q) = ∫ G(r) sin(Qr) dr。"""
    rf, g = g_full
    return np.asarray((np.sin(np.outer(q, rf)) @ g) * (rf[1] - rf[0]))


def _intensity(q: np.ndarray, g_full: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """S(Q) = 1 + F(Q)/Q から I(Q) = 1000 [(S - 1) <f>² + <f²>] を作る。"""
    fsq, f2 = average_scattering(q, {"Ni": 1.0}, "xray")
    return np.asarray(1000.0 * ((_fq(q, g_full) / q) * fsq + f2))


def _settings(qmax: float, lorch: bool = False, rmax: float = 15.0) -> TransformSettings:
    return TransformSettings(
        composition="Ni", qmin=0.5, qmax=qmax, rpoly=0.9, rmin=0.0, rmax=rmax, rstep=0.02,
        lorch=lorch,
    )  # fmt: skip


def _noisy(tr: TransformResult, name: str, rng: np.random.Generator) -> PDFData:
    pd = tr.to_pdf_data(name)
    pd.g = pd.g + rng.normal(0, 0.02, len(pd.g))
    pd.dg = np.full_like(pd.r, 0.02)
    return pd


def _fit_pdfdata(pd: PDFData, qmax: float | None) -> dict[str, float]:
    """Project でフィット(qmax=None ならメタデータから)。"""
    proj = Project()
    proj.add_phase(_ni(3.53, 0.004), name="Ni")
    proj.add_pdf(pd, "xray", qmax=qmax, qdamp=QDAMP, r_min=1.5, r_max=float(pd.r[-1]))
    proj.fix_all()
    proj.vary("Ni:cell:a", "Ni:Ni1:uiso", "g0:pdf:scale", "g0:pdf:qdamp")
    proj.refine(max_cycles=30)
    out = {p.name.split(":")[-1]: float(p.value) for p in proj.params if p.vary}
    out["rw"] = float(proj.rfactors()["g0"].rwp)
    out["qmax"] = float(proj.pdf_datasets[0].qmax)
    return out


def test_transform_clips_qmax_to_data(g_full: tuple[np.ndarray, np.ndarray]) -> None:
    """既定の Q_max = 25 のまま Cu Kα のデータを変換すると、データの上限 7.88 で切られる。"""
    q = np.linspace(0.3, Q_CU, 1200)
    inten = _intensity(q, g_full)
    default = transform(q, inten, _settings(25.0))
    explicit = transform(q, inten, _settings(Q_CU))
    assert default.qmax_eff == pytest.approx(Q_CU, abs=1e-6)
    assert any("Q_max をデータの上限" in line for line in default.log)
    # 多項式の次数も切った Q_max から(25 なら 7、7.88 なら 2)
    assert any("次数 2" in line for line in default.log)
    assert np.allclose(default.gr, explicit.gr, atol=1e-9)
    meta = default.to_pdf_data("cu").metadata
    assert meta["qmax"] == pytest.approx(Q_CU, abs=1e-6)
    assert meta["qmin"] == 0.5 and meta["lorch"] is False
    # Lorch 窓も切った Q_max で掛かる(25 のままだと窓がほとんど効かない)
    lorch = transform(q, inten, _settings(25.0, lorch=True))
    assert lorch.to_pdf_data().metadata["lorch"] is True
    first = (default.r > 2.2) & (default.r < 2.8)
    assert lorch.gr[first].max() < 0.8 * default.gr[first].max()


def test_fit_with_default_qmax_recovers_u(g_full: tuple[np.ndarray, np.ndarray]) -> None:
    """既定の Q_max 25 のまま変換・登録しても、U と scale が正しい Q_max を指定したときと同じ。

    修正前(Cu Kα): U = 0.029(真値の 4.8 倍)、scale = 1.88、Rw = 43%。
    scale の絶対値は Mo Kα で見る(Cu Kα では規格化に使える高 Q 側の窓が狭く、アドホックな
    規格化係数そのものが 1.4 倍ずれる。これは終端効果とは別の問題)。
    """
    rng = np.random.default_rng(0)
    # Cu Kα: 既定のまま == 正しい値を明示、U は真値の 1.2 倍以内(r ≤ 20 Å でフィット)
    q = np.linspace(0.3, Q_CU, 1200)
    pd = _noisy(transform(q, _intensity(q, g_full), _settings(25.0, rmax=20.0)), "cu", rng)
    fit_default = _fit_pdfdata(pd, None)
    fit_explicit = _fit_pdfdata(pd, Q_CU)
    assert fit_default["qmax"] == pytest.approx(Q_CU, abs=1e-6)
    assert fit_default["uiso"] == pytest.approx(fit_explicit["uiso"], rel=1e-3)
    assert fit_default["scale"] == pytest.approx(fit_explicit["scale"], rel=1e-3)
    assert U_NI / 1.2 < fit_default["uiso"] < U_NI * 1.2
    assert fit_default["rw"] < 0.05
    # Mo Kα: U は真値の 1.2 倍以内、scale は 1 ± 0.1
    q = np.linspace(0.3, Q_MO, 1200)
    pd = _noisy(transform(q, _intensity(q, g_full), _settings(25.0)), "mo", rng)
    fit = _fit_pdfdata(pd, None)
    assert fit["qmax"] == pytest.approx(Q_MO, abs=1e-6)
    assert U_NI / 1.2 < fit["uiso"] < U_NI * 1.2
    assert abs(fit["scale"] - 1.0) < 0.1
    assert fit["rw"] < 0.03


def test_lorch_window_in_model(g_full: tuple[np.ndarray, np.ndarray]) -> None:
    """Lorch 窓を掛けた G(r) は、lorch=True のモデルと一致し、U も回収できる(修正前は 2.2 倍)。"""
    q = np.linspace(0.0, Q_MO, 2000)
    fq = _fq(q, g_full)
    r = np.arange(0.5, 15.0, 0.02)
    g_obs = sine_transform(q, fq, r, Q_MO, lorch=True)
    model = calculate_pdf(r, [PDFPhase(_ni())], "xray", qmax=Q_MO, qdamp=QDAMP, lorch=True)
    first = (r > 2.2) & (r < 2.8)
    assert np.max(np.abs(model.g_calc - g_obs)) < 0.03 * g_obs[first].max()
    # 窓を知らないモデルは第 1 ピークが高すぎる
    plain = calculate_pdf(r, [PDFPhase(_ni())], "xray", qmax=Q_MO, qdamp=QDAMP)
    assert plain.g_calc[first].max() > 1.2 * g_obs[first].max()

    def fit_u(lorch: bool) -> float:
        def resid(p: np.ndarray) -> np.ndarray:
            a, u, s, qd = p
            res = calculate_pdf(
                r, [PDFPhase(_ni(a, u))], "xray", qmax=Q_MO, qdamp=qd, scale=s, lorch=lorch
            )
            return np.asarray(res.g_calc - g_obs)

        sol = least_squares(resid, [3.53, 0.004, 1.0, 0.05], diff_step=1e-4)
        return float(sol.x[1])

    assert U_NI / 1.2 < fit_u(True) < U_NI * 1.2
    assert fit_u(False) > 1.5 * U_NI


def test_u_zero_area_conserved() -> None:
    """U = 0 でも第 1 殻の配位数は 12(修正前は a によって 0.2〜46 と変わった)。"""
    r = np.arange(0.01, 4.0, 0.01)
    for a in np.arange(3.520, 3.5301, 0.001):
        res = calculate_pdf(r, [PDFPhase(_ni(float(a), 0.0))], "xray", qmax=0.0)
        rho0 = 4.0 / a**3
        rdf = r * (res.g_calc + 4.0 * math.pi * r * rho0)
        shell = (r > 2.2) & (r < 2.8)
        assert float(np.sum(rdf[shell]) * 0.01) == pytest.approx(12.0, abs=0.05)
    # 普通の U でも変わらない(ビン積分は点の値と 0.1% 以内で一致)
    res = calculate_pdf(r, [PDFPhase(_ni(A_NI, 0.005))], "xray", qmax=0.0)
    rho0 = 4.0 / A_NI**3
    rdf = r * (res.g_calc + 4.0 * math.pi * r * rho0)
    shell = (r > 2.2) & (r < 2.8)
    assert float(np.sum(rdf[shell]) * 0.01) == pytest.approx(12.0, abs=0.05)
