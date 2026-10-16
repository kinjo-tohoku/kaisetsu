"""残差の点検: 合成パターンに足した異物ピークとこぶを見つけ、相の反射は拾わない。"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.io.export import write_unexplained_peaks_csv
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.refine.residual import (
    find_residual_features,
    format_unexplained_peaks,
    observed_d_spacing,
    unexplained_peak_rows,
)

ROOT = Path(__file__).resolve().parents[2]
LAB6_CIF = ROOT / "data" / "public" / "lab6" / "LaB6_COD1000055.cif"
WIDTHS = {"u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02}


def _gauss(x: np.ndarray, c: float, fwhm: float, h: float) -> np.ndarray:
    s = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.asarray(h * np.exp(-0.5 * ((x - c) / s) ** 2))


def _project(extra: Callable[[np.ndarray, Project], np.ndarray] | None = None) -> Project:
    """真の装置で計算した LaB6 に extra を足し、同じ装置で scale と背景だけ合わせたプロジェクト。"""
    st = Structure.from_cif(str(LAB6_CIF))
    x = np.arange(20.0, 80.0, 0.02)
    inst = CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **WIDTHS)
    sim = Project("sim")
    sim.add_phase(st.copy(), name="L")
    sim.add_histogram(PowderData(x, np.ones_like(x)), inst, SampleGeometry(), n_background=2)
    sim.build_parameters()
    y_peaks = sim.calculate(0).y_bragg
    y_true = 3000.0 * y_peaks / y_peaks.max() + 20.0
    if extra is not None:
        y_true = y_true + extra(x, sim)
    y = np.random.default_rng(3).poisson(y_true).astype(np.float64)
    proj = Project("fit")
    proj.add_phase(st.copy(), name="L")
    proj.add_histogram(
        PowderData(x, y),
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **WIDTHS),
        SampleGeometry(),
        n_background=3,
    )
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:L:scale")
    proj.refine(max_cycles=8)
    return proj


def test_clean_fit_has_no_features() -> None:
    feats = find_residual_features(_project(), 0)
    assert feats.unindexed == [] and feats.humps == []
    assert feats.notes() == []


def test_finds_unindexed_peak_and_hump() -> None:
    def extra(x: np.ndarray, _sim: Project) -> np.ndarray:
        return _gauss(x, 47.3, 0.15, 600.0) + _gauss(x, 60.0, 5.0, 300.0)

    proj = _project(extra)
    feats = find_residual_features(proj, 0)
    assert feats.unindexed, feats
    assert all(abs(p.position - 47.3) < 0.2 for p in feats.unindexed)
    top = feats.unindexed[0]
    assert top.lo < 47.3 < top.hi and top.sigma > 8 and top.calc_fraction < 0.5
    assert feats.humps, feats
    hm = feats.humps[0]
    assert abs(hm.center - 60.0) < 1.0 and 3.0 < hm.fwhm < 8.0 and 150 < hm.height < 450
    notes = feats.notes()
    assert any("説明できないピーク" in n and "47.3" in n for n in notes)
    assert any("こぶ" in n for n in notes)
    # 精密化のあとの診断にも出る
    diag = proj.diagnostics()
    assert any("説明できないピーク" in n for n in diag) and any("こぶ" in n for n in diag)


def test_misfit_reflection_is_not_an_exclusion_candidate() -> None:
    def extra(x: np.ndarray, sim: Project) -> np.ndarray:
        pk = sim.calculate(0).peaks[0]
        k = int(np.argmax(pk.intensity_total))  # 最強反射(110)
        return _gauss(x, float(pk.two_theta[k]), float(pk.fwhm[k]), 0.5 * 3000.0)

    proj = _project(extra)
    feats = find_residual_features(proj, 0)
    assert feats.misfit, feats
    assert all(p.calc_fraction >= 0.5 for p in feats.misfit)
    strongest = proj.calculate(0).peaks[0]
    pos = float(strongest.two_theta[int(np.argmax(strongest.intensity_total))])
    assert any(abs(p.position - pos) < 0.1 for p in feats.misfit)
    assert not any(abs(p.position - pos) < 0.3 for p in feats.unindexed)
    assert any("観測が計算より強い" in n for n in feats.notes())


def test_unindexed_peak_has_d_spacing_and_can_be_exported(tmp_path: Path) -> None:
    """未同定ピークに d 値が付き、一覧(タブ区切り・CSV)に出る。"""

    def extra(x: np.ndarray, _sim: Project) -> np.ndarray:
        return _gauss(x, 47.3, 0.15, 600.0)

    proj = _project(extra)
    feats = find_residual_features(proj, 0)
    assert feats.unindexed, "47.3° のピークが未同定として見つかるはず"
    p = feats.unindexed[0]
    assert abs(p.position - 47.3) < 0.05
    lam = 1.54056
    d_expect = lam / (2.0 * math.sin(math.radians(p.position / 2.0)))
    assert abs(p.d - d_expect) < 1e-6 * d_expect  # ゼロ点 0、試料位置ずれ 0

    rows = unexplained_peak_rows([feats])
    assert rows and rows[0]["kind"] == "unindexed" and rows[0]["unit"] == "°"
    assert abs(float(rows[0]["d"]) - p.d) < 1e-12
    text = format_unexplained_peaks([feats])
    head, first = text.splitlines()[:2]
    assert "d (Å)" in head and "\t" in head
    assert first.split("\t")[1] == "未同定" and f"{p.d:.5f}" in first

    out = write_unexplained_peaks_csv(proj, tmp_path / "peaks.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    expect = "hist,kind,position,unit,d,height,rel_height_pct,sigma,nearest_reflection,lo,hi"
    assert lines[0] == expect
    assert any(ln.startswith("h0,unindexed,") and f",{p.d:.5f}," in ln for ln in lines[1:])
    only = write_unexplained_peaks_csv(proj, tmp_path / "u.csv", include_misfit=False)
    assert all("misfit" not in ln for ln in only.read_text(encoding="utf-8").splitlines())


def test_observed_d_spacing_removes_zero_and_displacement() -> None:
    """ゼロ点と試料変位の分を外した d になる(補正を足して戻すと元の位置)。"""
    proj = _project()
    h = proj.histograms[0]
    assert isinstance(h.instrument, CWInstrument)
    h.instrument.zero = 0.08
    h.sample.displacement_mm = 0.3  # 平板の変位(ゼロ点と逆向きに 0.1° 強ずらす)
    pos = np.array([30.0, 47.3, 75.0])
    d = observed_d_spacing(h, pos)
    tt = 2.0 * np.degrees(np.arcsin(h.instrument.radiation.wavelength / (2.0 * d)))
    back = tt + h.instrument.zero + h.sample.position_shift(tt)
    assert np.allclose(back, pos, atol=1e-6)
    assert np.all(np.abs(tt - pos) > 0.02)  # 補正は効いている(ゼロ点と変位は打ち消し合う向き)
