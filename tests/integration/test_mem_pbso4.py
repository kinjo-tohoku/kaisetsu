"""実データでの MEM / MPF: PbSO4 の実験室 X 線 Rietveld → MEM 電子密度 → MPF。

Pb の位置に最大の密度があり、S、O の位置にも極大が出ること、MPF で Rwp が
Rietveld 単独より悪くならないことを確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, read_powder_data
from kaisetsu.crystal.cif import read_structure
from kaisetsu.mem import mem_from_project, run_mpf
from kaisetsu.refine.recipe import standard_recipe

pytestmark = pytest.mark.slow


def _refined_pbso4(data_dir: Path) -> Project:
    proj = Project("PbSO4 MEM")
    proj.add_phase(read_structure(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"), name="PbSO4")
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), u=0.002, v=-0.002, w=0.004, x=0.02, y=0.02,
        s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip
    proj.add_histogram(
        read_powder_data(data_dir / "pbso4" / "PBSO4.XRA"), inst, SampleGeometry(), n_background=8
    )
    standard_recipe("PbSO4", "h0", sample_displacement=True).run(proj)
    return proj


def test_mem_and_mpf_on_pbso4(data_dir: Path) -> None:
    proj = _refined_pbso4(data_dir)
    rwp0 = proj.rfactors()["total"].rwp
    assert rwp0 < 0.15
    st = proj.phases["PbSO4"].structure
    pb = st.site("Pb").position.copy()

    mem = mem_from_project(proj, 0, "PbSO4", grid=(48, 32, 40), max_outer=12)
    assert mem.chi2 == pytest.approx(1.0, abs=0.1)
    assert np.all(mem.density >= 0)
    peaks = mem.maxima(6, min_separation=1.0)
    top, _ = peaks[0]
    # 最大密度は Pb の位置(等価位置のどれか)
    sg = st.spacegroup
    eq, _ = sg.equivalent_positions(pb)
    diffs = [(top - e) - np.round(top - e) for e in eq]
    d = np.array([float(st.lattice.distance(np.zeros(3), v)) for v in diffs])
    assert d.min() < 0.3
    # Pb の球内電子数は 60〜90(Pb2+: 80 e、分解能と重なりで減る)
    n_pb = mem.integrate_sphere(pb, 1.3)
    assert 55 < n_pb < 95

    res = run_mpf(proj, 0, "PbSO4", grid=(48, 32, 40), n_cycles=2, max_outer=12)
    assert len(res.densities) == 2
    # MPF のパターンフィッティングは Rietveld より悪くならない(|F_MEM| は観測に合わせてある)
    assert res.rwp_history[-1] <= rwp0 * 1.02
    assert proj.phases["PbSO4"].le_bail and not proj.phases["PbSO4"].f2_update
