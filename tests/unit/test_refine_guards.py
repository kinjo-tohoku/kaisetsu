"""精密化の入力チェック(点が無い・動かす量が多すぎる)、TOF でのピーク位置、旧形式の粗さ。"""

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
from kaisetsu.io.project import _sample_from_dict
from kaisetsu.pattern.instrument import GeometryKind
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.refine.recipe import standard_recipe


def _project(data_dir: Path) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
    )
    proj = Project("guards")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def test_refine_rejects_empty_fit_range(data_dir: Path) -> None:
    proj = _project(data_dir)
    h = proj.histograms[0]
    h.set_excluded([(0.0, 200.0)])
    assert proj.n_fit_points() == 0
    proj.vary("h0:bkg:*")
    with pytest.raises(ValueError, match="データ点がありません"):
        proj.refine(max_cycles=2)
    h.set_excluded([])
    h.x_min = h.x_max = 30.0
    proj.vary("h0:bkg:*", "h0:PbSO4:scale")
    with pytest.raises(ValueError, match="データ点"):
        proj.refine(max_cycles=2)


def test_refine_rejects_more_parameters_than_points(data_dir: Path) -> None:
    proj = _project(data_dir)
    h = proj.histograms[0]
    h.x_min, h.x_max = 30.0, 30.3
    proj.vary("*")
    n = proj.n_fit_points()
    assert 0 < n < len(proj.params.free())
    with pytest.raises(ValueError, match="データ点"):
        proj.refine(max_cycles=2)


def test_standard_recipe_uses_zero_for_tof_and_shift_for_cw(data_dir: Path) -> None:
    """試料変位を選んでも、TOF では disp_x/y(意味が無く esd が発散する)でなく zero を動かす。"""
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    tof_data = read_powder_data(data_dir / "nac" / "PG3_22048.gsa")
    proj = Project("tof-recipe")
    proj.add_phase(st, name="P")
    proj.add_histogram(
        tof_data, TOFInstrument(difc=22000.0), SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER)
    )
    cw = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    proj.add_histogram(
        cw, CWInstrument(Radiation.xray(1.5406), w=0.003), SampleGeometry(), n_background=3
    )
    proj.build_parameters()
    r = standard_recipe("*", "h*", sample_displacement=True)
    step = next(s for s in r.steps if s.name == "格子定数とピーク位置")
    chosen = {
        p.name
        for pat in step.patterns
        for p in proj.params.select(pat)
        if step.filter is None or step.filter(proj, p)
    }
    assert "h0:inst:zero" in chosen and "h0:sample:disp_x" not in chosen
    assert "h1:sample:displacement" in chosen and "h1:inst:zero" not in chosen
    assert "P:cell:a" in chosen


def test_legacy_roughness_zero_becomes_off_on_load() -> None:
    smp = _sample_from_dict({"kind": "bragg-brentano", "roughness_a": 0.0, "roughness_b": 0.0})
    assert (smp.roughness_a, smp.roughness_b) == (1.0, 1.0)
    kept = _sample_from_dict({"kind": "bragg-brentano", "roughness_a": 0.4, "roughness_b": 2.0})
    assert (kept.roughness_a, kept.roughness_b) == (0.4, 2.0)


def test_n_fit_points_counts_masked_points(data_dir: Path) -> None:
    proj = _project(data_dir)
    h = proj.histograms[0]
    assert proj.n_fit_points() == int(np.count_nonzero(h.mask()))
    h.set_excluded([(25.0, 26.0)])
    assert proj.n_fit_points() == int(np.count_nonzero(h.mask())) < len(h.data)


def test_width_guard_fixes_tof_sigma_terms(data_dir: Path) -> None:
    """TOF で σ² が負になったら、動いている sig0, sig2(無ければ sig1)を段階前の値に固定する。"""
    from kaisetsu.refine.recipe import Recipe, Step

    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    tof_data = read_powder_data(data_dir / "nac" / "PG3_22048.gsa")
    proj = Project("tof-guard")
    proj.add_phase(st, name="P")
    proj.add_histogram(
        tof_data,
        TOFInstrument(difc=22000.0, sig1=50.0),
        SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER),
        n_background=2,
    )
    proj.build_parameters()
    recipe = Recipe("t")
    step = Step("幅", ("h0:inst:sig0", "h0:inst:sig2"), 1)
    snapshot = {q.name: q.value for q in proj.params}
    # 段階の精密化で sig0 が流れて σ² が全域で負になった、という状況を作る
    proj.vary("h0:bkg:*", "h0:inst:sig0", "h0:inst:sig2")
    res = proj.refine(max_cycles=1)
    proj.params["h0:inst:sig0"].value = -1.0e5
    proj.params.apply_all()
    recipe._guard_widths(proj, step, snapshot, res, None)
    assert not proj.params["h0:inst:sig0"].vary and not proj.params["h0:inst:sig2"].vary
    assert proj.params["h0:inst:sig0"].value == 0.0  # 段階前の値に戻っている
    assert proj.params["h0:bkg:0"].vary  # 他は動いたまま
    assert any("sig0, h0:inst:sig2 を段階前の値に固定" in line for line in proj.log)
    # sig0, sig2 が固定で sig1 だけ動いているなら sig1 を固定する
    proj.vary("h0:inst:sig1")
    proj.params["h0:inst:sig1"].value = -1.0e4
    proj.params.apply_all()
    recipe._guard_widths(proj, step, snapshot, res, None)
    assert not proj.params["h0:inst:sig1"].vary
    assert proj.params["h0:inst:sig1"].value == 50.0


def test_standard_recipe_varies_tof_alpha(data_dir: Path) -> None:
    """標準手順は TOF の立ち上がり α と減衰 β を sig1 と同じ段階で動かす(装置ごとに桁で違う)。"""
    r = standard_recipe("*", "h0")
    step = next(s for s in r.steps if s.name == "ピーク幅 W")
    assert {"h0:inst:sig1", "h0:inst:alpha", "h0:inst:beta0", "h0:inst:beta1"} <= set(step.patterns)
    step2 = next(s for s in r.steps if s.name.startswith("ピーク幅 U"))
    assert "h0:inst:sig0" in step2.patterns and "h0:inst:alpha" not in step2.patterns
