"""表面粗さの 1 パラメータ化、SH/L の統合、詳細パラメータ(P, Z, rough_b)、旧プロジェクトの移行。"""

from __future__ import annotations

import json
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
from kaisetsu.io.project import load_project, save_project
from kaisetsu.profile.corrections import surface_roughness_suortti
from kaisetsu.refine.recipe import standard_recipe


def test_suortti_off_conventions_and_derivative() -> None:
    tt = np.linspace(10.0, 150.0, 50)
    assert np.allclose(surface_roughness_suortti(1.0, 1.0, tt), 1.0)
    assert np.allclose(surface_roughness_suortti(0.0, 0.0, tt), 1.0)  # 旧来の「なし」
    f = surface_roughness_suortti(0.5, 1.0, tt)
    assert f[0] < f[-1] <= 1.0 + 1e-12  # 低角ほど強度が落ちる
    # a = 1(補正なし)から動かし始めても微分が 0 でない(精密化が進む)
    eps = 1e-6
    da = (surface_roughness_suortti(1.0 - eps, 1.0, tt) - 1.0) / -eps
    assert np.max(np.abs(da)) > 1e-3


def test_sample_geometry_default_is_off_and_legacy_zero_maps_to_one() -> None:
    smp = SampleGeometry()
    assert (smp.roughness_a, smp.roughness_b) == (1.0, 1.0)
    legacy = SampleGeometry(roughness_a=0.0, roughness_b=0.0)
    assert (legacy.roughness_a, legacy.roughness_b) == (1.0, 1.0)
    gs = SampleGeometry.from_gsas2({"SurfRoughA": [0.0, False], "SurfRoughB": [0.0, False]})
    assert (gs.roughness_a, gs.roughness_b) == (1.0, 1.0)
    kept = SampleGeometry(roughness_a=0.3, roughness_b=2.0)
    assert (kept.roughness_a, kept.roughness_b) == (0.3, 2.0)


def test_sh_l_property_splits_equally() -> None:
    inst = CWInstrument(Radiation.xray(1.5406), s_l=0.03, h_l=0.01)
    assert inst.sh_l == pytest.approx(0.04)
    inst.sh_l = 0.05
    assert inst.s_l == pytest.approx(0.025) and inst.h_l == pytest.approx(0.025)


def _project(data_dir: Path) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
    )
    proj = Project("pbso4-shl")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def test_parameter_set_uses_sh_l_and_marks_advanced(data_dir: Path) -> None:
    proj = _project(data_dir)
    p = proj.params
    assert "h0:inst:sh_l" in p and "h0:inst:s_l" not in p and "h0:inst:h_l" not in p
    assert p["h0:inst:sh_l"].value == pytest.approx(0.02)
    for name in ("h0:inst:p", "h0:inst:z"):
        assert p[name].advanced and p[name].neutral_value == 0.0
    assert p["h0:sample:rough_b"].advanced and p["h0:sample:rough_b"].neutral_value is None
    assert not p["h0:sample:rough_a"].advanced and not p["h0:inst:u"].advanced
    # 値を書くと装置に反映される(S/L = H/L = SH/L / 2)
    p["h0:inst:sh_l"].value = 0.06
    p["h0:inst:sh_l"].apply()
    inst = proj.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    assert inst.s_l == pytest.approx(0.03) and inst.h_l == pytest.approx(0.03)


def test_standard_recipe_uses_single_asymmetry_and_roughness_parameter() -> None:
    r = standard_recipe("*", "h0", asymmetry=True, roughness=True)
    pats = {name: step.patterns for name, step in ((s.name, s) for s in r.steps)}
    assert pats["非対称"] == ("h0:inst:sh_l",)
    assert pats["表面粗さ"] == ("h0:sample:rough_a",)


def test_standard_recipe_structure_steps_skip_instrument_xyz(data_dir: Path) -> None:
    """phase='*' の「原子座標」が装置の h0:inst:x, y, z を巻き込まない。"""
    proj = _project(data_dir)
    r = standard_recipe("*", "h0")
    step = next(s for s in r.steps if s.name == "原子座標")
    chosen = [
        p.name
        for pat in step.patterns
        for p in proj.params.select(pat)
        if step.filter is None or step.filter(proj, p)
    ]
    assert "PbSO4:O3:x" in chosen
    assert not any(n.startswith("h0:") for n in chosen)
    step = next(s for s in r.steps if s.name == "変位パラメータ")
    chosen = [
        p.name
        for pat in step.patterns
        for p in proj.params.select(pat)
        if step.filter is None or step.filter(proj, p)
    ]
    assert "PbSO4:Pb:uiso" in chosen and not any(n.startswith("h0:") for n in chosen)


def test_load_migrates_old_s_l_h_l_vary_flags(data_dir: Path, tmp_path: Path) -> None:
    proj = _project(data_dir)
    proj.vary("h0:inst:sh_l")
    path = tmp_path / "old.kaisetsu.json"
    save_project(proj, path)
    d = json.loads(path.read_text(encoding="utf-8"))
    entry = d["params"].pop("h0:inst:sh_l")
    d["params"]["h0:inst:s_l"] = {**entry, "vary": False}
    d["params"]["h0:inst:h_l"] = {**entry, "vary": True}
    path.write_text(json.dumps(d), encoding="utf-8")
    loaded = load_project(path)
    q = loaded.params["h0:inst:sh_l"]
    assert q.vary and "h0:inst:s_l" not in loaded.params
    # 上下限は現在の既定が残る(旧ファイルの s_l の範囲で上書きしない)
    assert q.min == 0.0 and q.max == pytest.approx(0.4)
    assert q.value == pytest.approx(0.02)
