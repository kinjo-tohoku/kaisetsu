"""MEM 用の観測構造因子(mpf.observed_structure_factors)。

X 線では異常分散 f′, f″ を除いて f0 だけの構造因子に合わせる(外部レビュー 2026-10-17 精密化 #4)。
y_obs = y_calc の合成データなら |F_obs| = |F_calc(f0)| になるはず。修正前は tbl.f2(f′, f″ 込み)が
そのまま残り、Cu Kα の PbSO4 で |F_obs| の中央値が |F(f0)| の 0.94 倍、Pb の球内電子数が 4%
少なかった。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import CWInstrument, PowderData, Project, Radiation, SampleGeometry
from kaisetsu.crystal.cif import read_structure
from kaisetsu.mem import mem_from_project, observed_structure_factors, run_mpf
from kaisetsu.mem.mpf import anomalous_correction
from kaisetsu.reflections.structure_factor import (
    powder_structure_factor_squared,
    structure_factors,
)


def _synthetic_pbso4(data_dir: Path, radiation: Radiation) -> Project:
    """PbSO4 の計算パターンを「観測」にしたプロジェクト(y_obs = y_calc)。"""
    st = read_structure(data_dir / "pbso4" / "PbSO4-Wyckoff.cif")
    inst = CWInstrument(
        radiation, u=0.002, v=-0.002, w=0.004, x=0.02, y=0.02, s_l=0.01, h_l=0.01,
        polarization=0.7,
    )  # fmt: skip
    x = np.arange(10.0, 150.0, 0.02)
    p0 = Project("tmp")
    p0.add_phase(st.copy(), name="PbSO4")
    p0.add_histogram(PowderData(x, np.ones_like(x)), inst, SampleGeometry(), n_background=1)
    y = p0.calculate(0).y_calc + 1e-6
    proj = Project("syn")
    proj.add_phase(st.copy(), name="PbSO4")
    proj.add_histogram(
        PowderData(x, y, weight=1.0 / np.maximum(y, 1.0)), inst, SampleGeometry(), n_background=1
    )
    # 背景パラメータの初期値の分だけ y_calc がずれるので、もう一度 y_calc を観測にする
    y = proj.calculate(0).y_calc + 1e-6
    proj.histograms[0].data = PowderData(x, y, weight=1.0 / np.maximum(y, 1.0))
    return proj


def _well_observed(proj: Project, hkl: np.ndarray, f_ref: np.ndarray) -> np.ndarray:
    """範囲内(2θ < 150°)で、|F| が最大値の 5% 以上の反射。"""
    two_theta = proj.calculate(0).peaks[0].two_theta
    return (two_theta < 148.0) & (np.abs(f_ref) > 0.05 * np.abs(f_ref).max())


def test_xray_fobs_matches_f0_only_structure_factors(data_dir: Path) -> None:
    proj = _synthetic_pbso4(data_dir, Radiation.xray(1.5405, 1.5443, 0.5))
    obs = observed_structure_factors(proj, 0, "PbSO4")
    st = proj.phases["PbSO4"].structure
    rad = proj.histograms[0].instrument.radiation
    f0 = structure_factors(st, obs.hkl, rad, anomalous=dict.fromkeys(st.species_list(), 0j))
    sel = _well_observed(proj, obs.hkl, f0)
    assert np.count_nonzero(sel) > 100
    # 振幅は f0 だけの |F|(修正前は f′, f″ 込みで中央値 0.94 倍)
    ratio = np.abs(obs.f_obs[sel]) / np.abs(f0[sel])
    assert np.median(ratio) == pytest.approx(1.0, abs=2e-3)
    assert np.max(np.abs(ratio - 1.0)) < 0.02
    # 位相も f0 だけの F_calc と同じ(PbSO4 は中心対称なので符号)
    assert np.allclose(
        obs.f_obs[sel] / np.abs(obs.f_obs[sel]), f0[sel] / np.abs(f0[sel]), atol=1e-6
    )
    # F000 は電子数(Pb 82 + S 16 + O 8 × 4 = 130 e × Z = 4)
    assert obs.f000 == pytest.approx(4 * 130.0, rel=1e-3)
    # 補正比は Pb の f′ ≈ −4 e を反映して 1 より大きい側に偏る
    _fc0, corr = anomalous_correction(st, obs.hkl, rad)
    assert np.median(corr[sel]) > 1.05
    assert np.all(corr > 0)
    # 中心が測定範囲(2θ < 150°)の外の反射は σ = inf(拘束に入れない)、中は有限
    two_theta = proj.calculate(0).peaks[0].two_theta
    outside = ~np.isfinite(two_theta) | (two_theta > 150.0)
    assert np.count_nonzero(outside) > 0
    assert np.all(np.isinf(obs.sigma[outside]))
    assert np.all(np.isfinite(obs.sigma[~outside]))


def test_neutron_fobs_unchanged(data_dir: Path) -> None:
    proj = _synthetic_pbso4(data_dir, Radiation.neutron(1.5))
    obs = observed_structure_factors(proj, 0, "PbSO4")
    st = proj.phases["PbSO4"].structure
    rad = proj.histograms[0].instrument.radiation
    fc = structure_factors(st, obs.hkl, rad)
    sel = _well_observed(proj, obs.hkl, fc)
    assert np.count_nonzero(sel) > 100
    ratio = np.abs(obs.f_obs[sel]) / np.abs(fc[sel])
    assert np.max(np.abs(ratio - 1.0)) < 0.02
    assert obs.radiation == "neutron"


@pytest.mark.slow
def test_xray_mem_pb_electrons(data_dir: Path) -> None:
    """合成 PbSO4 の MEM で Pb の球内電子数が f0 だけの参照値(76.4 e)に一致する(修正前 73.3 e)。

    MPF のパターンフィッティングでは |F_MEM|²(f0 だけ)に異常分散を戻すので、構造因子表は
    モデルの |F(f0 + f′ + if″)|² と一致する(戻さないと中央値で 1.13 倍、Rwp 7.0%。戻すと
    Rwp 4.8%。残りは MEM が σ の範囲で |F| を動かした分)。
    """
    proj = _synthetic_pbso4(data_dir, Radiation.xray(1.5405, 1.5443, 0.5))
    st = proj.phases["PbSO4"].structure
    pb = st.site("Pb").position.copy()
    mem = mem_from_project(proj, 0, "PbSO4", grid=(48, 32, 40), max_outer=15)
    assert mem.chi2 == pytest.approx(1.0, abs=0.05)
    assert mem.integrate_sphere(pb, 1.3) == pytest.approx(76.4, abs=1.0)
    res = run_mpf(proj, 0, "PbSO4", grid=(48, 32, 40), n_cycles=1, max_outer=15)
    assert res.densities[0].integrate_sphere(pb, 1.3) == pytest.approx(76.4, abs=1.0)
    rad = proj.histograms[0].instrument.radiation
    f2_anom = powder_structure_factor_squared(st, mem.hkl, rad)
    table = proj.phases["PbSO4"].f2_tables[0]
    f2_tbl = np.array([table[(int(a), int(b), int(c))] for a, b, c in mem.hkl])
    sel = _well_observed(proj, mem.hkl, mem.f_calc)
    assert np.median(f2_tbl[sel] / f2_anom[sel]) == pytest.approx(1.0, abs=0.02)
    assert res.rwp_history[-1] < 0.06
