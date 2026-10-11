"""TOF 中性子: リーダー、装置パラメータ、パターン計算、Project への組み込み、保存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    Lattice,
    PhaseContribution,
    Project,
    SpaceGroup,
    Structure,
    TOFInstrument,
    calculate_pattern_tof,
    read_powder_data,
)
from kaisetsu.crystal.cif import read_structure
from kaisetsu.io.project import load_project, save_project


def test_read_tof_gsa(data_dir: Path) -> None:
    d = read_powder_data(data_dir / "nac" / "PG3_22048.gsa")
    assert d.metadata["xunit"] == "tof_us" and d.metadata["binning"] == "SLOG"
    # ビン端 → ビン中心(最後の点は落ちる)、強度と esd はビン幅で割る(GSAS-II の読み方と同じ)
    assert len(d) == 6824 and d.x[0] == pytest.approx(6778.4370, abs=0.001)
    assert d.x[-1] > 100000
    assert np.all(np.diff(d.x) > 0)
    cw0 = 6779.79237452 - 6777.0815419
    assert d.y[0] == pytest.approx(117.13886376 / cw0, rel=1e-6)
    assert d.weight is not None and d.weight[0] == pytest.approx(cw0**2 / 5.2528**2, rel=1e-3)
    assert d.metadata["yunit"] == "counts/us"


def test_instrument_from_instprm(data_dir: Path) -> None:
    inst = TOFInstrument.from_instprm(data_dir / "nac" / "POWGEN_1066.instprm")
    assert inst.difc == pytest.approx(22600.2479, abs=1e-3)
    assert inst.two_theta == 90.0 and inst.beta1 == pytest.approx(0.00272732, rel=1e-5)
    d = np.array([1.0, 2.0])
    t = inst.position(d)
    assert np.allclose(inst.d_from_tof(t), d)
    assert inst.to_gsas2()["sig-q"] == pytest.approx(-3.7297, abs=1e-3)
    with pytest.raises(ValueError):
        TOFInstrument.from_instprm(data_dir / "pbso4" / "INST_XRY.PRM")


def test_pattern_peaks_at_predicted_tof() -> None:
    st = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("P m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.005)],
    )
    inst = TOFInstrument(
        difc=20000.0, alpha=0.2, beta0=0.05, beta1=0.002, sig1=40.0, two_theta=90.0
    )
    x = np.arange(10000.0, 90000.0, 5.0)
    res = calculate_pattern_tof(x, [PhaseContribution(st, scale=100.0)], inst)
    tbl = res.peaks[0]
    assert len(tbl.hkl) > 5
    # (100) の位置 = difc × 4.0 = 80000。減衰側に裾を引くので最大は位置の少し前後
    k = int(np.argmin(np.abs(tbl.d - 4.0)))
    assert tbl.two_theta[k] == pytest.approx(80000.0)
    a, b = int(tbl.window_start[k]), int(tbl.window_stop[k])
    peak_x = x[a:b][np.argmax(res.y_bragg[a:b])]
    assert abs(peak_x - 80000.0) < 200.0
    assert res.y_bragg.sum() > 0
    # 強度は d⁴ sinθ の Lorentz 因子を含む: (100) と (200) の比
    k2 = int(np.argmin(np.abs(tbl.d - 2.0)))
    assert tbl.intensity[k] / tbl.intensity[k2] == pytest.approx(
        16.0 * tbl.multiplicity[k] * tbl.f2[k] / (tbl.multiplicity[k2] * tbl.f2[k2]), rel=1e-6
    )


def test_project_with_tof_histogram(data_dir: Path, tmp_path: Path) -> None:
    proj = Project("NAC TOF")
    proj.add_phase(read_structure(data_dir / "nac" / "NAC.cif"), name="NAC")
    data = read_powder_data(data_dir / "nac" / "PG3_22048.gsa")
    inst = TOFInstrument.from_instprm(data_dir / "nac" / "POWGEN_1066.instprm")
    h = proj.add_histogram(data, inst, n_background=4, x_min=10000.0, x_max=60000.0)
    assert h.is_tof
    names = proj.params.names()
    assert "h0:inst:difc" in names and "h0:inst:sig1" in names and "h0:inst:zero" in names
    assert "h0:inst:u" not in names
    assert 0.3 < h.effective_d_min() < 0.5 and h.effective_d_max() is not None
    res = proj.calculate(0)
    assert len(res.peaks[0].hkl) > 50 and np.all(np.isfinite(res.y_calc))
    # スケールと背景だけ動かして Rwp が下がる
    proj.phase_contribution(0, "NAC").scale = 0.01
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:NAC:scale")
    rwp0 = proj.rfactors()["total"].rwp
    fit = proj.refine(max_cycles=6)
    assert fit.success
    rwp1 = proj.rfactors()["total"].rwp
    assert rwp1 < rwp0 and rwp1 < 0.6
    rep = proj.reflection_report(0, "NAC")
    assert len(rep.hkl) == len(res.peaks[0].hkl)
    path = save_project(proj, tmp_path / "tof.json", external_arrays=True)
    p2 = load_project(path)
    assert isinstance(p2.histograms[0].instrument, TOFInstrument)
    assert p2.histograms[0].instrument.difc == pytest.approx(inst.difc)
    assert p2.params["h0:NAC:scale"].value == pytest.approx(proj.params["h0:NAC:scale"].value)
