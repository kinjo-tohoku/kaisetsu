"""ピーク幅が負になる検出(width_check)、罰則残差、診断、最小二乗の n_penalty。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.refine.lsq import refine_least_squares
from kaisetsu.refine.parameters import Parameter, ParameterSet
from kaisetsu.refine.width_check import (
    G2_TOL,
    n_penalty_rows,
    width_penalty,
    width_violations,
)


def _project(data_dir: Path, inst: CWInstrument) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 60.0)
    proj = Project("pbso4-width")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def _good() -> CWInstrument:
    return CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), u=0.004, v=-0.003, w=0.004, x=0.02, y=0.02,
        s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip


def test_raw_widths_can_be_negative_but_fwhm_is_clamped() -> None:
    inst = CWInstrument(Radiation.xray(1.5406), u=-0.01, v=0.0, w=0.001, x=0.0, y=-0.1)
    tt = np.array([20.0, 60.0, 120.0])
    g2 = inst.gauss_fwhm2(tt)
    assert g2[0] > 0 and g2[-1] < 0
    assert np.all(inst.fwhm_gauss(tt) > 0)
    assert np.all(inst.lorentz_fwhm_raw(tt) < 0) and np.all(inst.fwhm_lorentz(tt) > 0)


def test_width_violations_reports_range(data_dir: Path) -> None:
    proj = _project(data_dir, _good())
    assert width_violations(proj) == []
    assert np.all(width_penalty(proj) == 0.0)
    assert len(width_penalty(proj)) == n_penalty_rows(proj) == 64
    # U を負にして高角で Γ_G² < 0 にする
    inst = proj.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    inst.u, inst.v, inst.w = -0.01, 0.0, 0.002
    v = width_violations(proj)
    assert len(v) == 1 and v[0].kind == "gauss" and v[0].hist == 0
    # tan²θ > 0.2 → 2θ > 48.2° あたりから負
    assert 47.0 < v[0].lo < 50.0 and v[0].hi == pytest.approx(60.0, abs=0.1)
    assert v[0].min_value < 0 and v[0].at == pytest.approx(60.0, abs=0.1)
    assert "Γ_G²" in v[0].message() and "h0:inst:u" in v[0].message()
    pen = width_penalty(proj)
    assert pen.shape == (64,) and pen[:32].max() > 0 and np.all(pen[32:] == 0.0)
    # 罰則は違反の大きさに比例(基準 G2_REF、重み sqrt(N))
    n = int(np.count_nonzero(proj.histograms[0].mask()))
    assert pen.max() == pytest.approx(np.sqrt(n) * (-v[0].min_value) / 1.0e-4, rel=1e-6)
    # 診断にも出る
    notes = proj.diagnostics()
    assert any("Γ_G²" in s for s in notes)
    # ローレンツ幅
    inst.u, inst.v, inst.w = 0.004, -0.003, 0.004
    inst.x, inst.y, inst.z = 0.0, 0.0, -0.01
    v = width_violations(proj)
    assert [w.kind for w in v] == ["lorentz"]
    assert "Γ_L" in v[0].message()


def test_width_violations_near_zero_counts_as_boundary(data_dir: Path) -> None:
    proj = _project(data_dir, _good())
    inst = proj.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    inst.u, inst.v, inst.w = 0.0, 0.0, 0.5 * G2_TOL
    v = width_violations(proj)
    assert len(v) == 1 and v[0].lo == pytest.approx(20.0, abs=0.1)
    assert np.all(width_penalty(proj) == 0.0)  # 正なので罰則は無い


def test_tof_sigma2_violation() -> None:
    inst = TOFInstrument(difc=5000.0, sig0=-100.0, sig1=0.0, sig2=0.0)
    d = np.array([1.0, 2.0])
    assert np.all(inst.sigma2_raw(d) < 0) and np.all(inst.sigma2_of(d) > 0)


def test_lsq_excludes_penalty_rows_from_chi2_and_n_data() -> None:
    ps = ParameterSet()
    ps.add(Parameter("a", 0.0, vary=True, step=1e-3))

    def resid() -> np.ndarray:
        a = ps["a"].value
        return np.array([a - 1.0, a - 3.0, 10.0 * max(0.0, a - 1.5)])

    # 罰則なしなら a = 2(データの平均)、罰則行を含めれば a ≈ 1.5 に押し戻される
    res = refine_least_squares(ps, resid, max_cycles=30, n_penalty=1)
    assert res.n_data == 2
    assert ps["a"].value == pytest.approx(1.5, abs=0.02)
    assert res.chi2 == pytest.approx(((0.5) ** 2 + (1.5) ** 2) / 1, rel=0.05)


def test_refine_keeps_gauss_width_nonnegative(data_dir: Path) -> None:
    """U を負から出発させても、罰則があればデータ範囲で Γ_G² < 0 にならない。"""
    inst = _good()
    inst.u = -0.02  # 60° で Γ_G² < 0 になる出発点
    proj = _project(data_dir, inst)
    assert width_violations(proj)
    proj.vary("h0:bkg:*", "h0:PbSO4:scale", "h0:inst:u", "h0:inst:v", "h0:inst:w")
    proj.refine(max_cycles=15)
    xs = np.linspace(20.0, 60.0, 400)
    inst2 = proj.histograms[0].instrument
    assert isinstance(inst2, CWInstrument)
    assert inst2.gauss_fwhm2(xs).min() > -1e-6
    # 罰則行は χ² とデータ点数に入らない
    assert proj.last_fit is not None
    assert proj.last_fit.n_data == int(np.count_nonzero(proj.histograms[0].mask()))


def test_penalty_can_be_disabled(data_dir: Path) -> None:
    proj = _project(data_dir, _good())
    assert proj.n_penalty_rows() == 64
    proj.penalize_negative_widths = False
    assert proj.n_penalty_rows() == 0
    assert len(proj.residuals_with_penalty()) == len(proj.residuals())
