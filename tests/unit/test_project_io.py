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
from kaisetsu.refine.recipe import Recipe, standard_recipe


def _project(data_dir: Path) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 60.0)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
    )
    proj = Project("pbso4-test")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(
        data, inst, SampleGeometry(displacement_mm=0.02), n_background=4, excluded=[(30.0, 31.0)]
    )
    pc = proj.phase_contribution(0, "PbSO4")
    pc.scale, pc.size_um, pc.strain_e6 = 0.02, 1.0, 1000.0
    proj.build_parameters()
    return proj


def test_parameter_names_and_constraints(data_dir: Path) -> None:
    proj = _project(data_dir)
    p = proj.params
    assert "PbSO4:cell:a" in p and "h0:inst:zero" in p and "h0:bkg:3" in p and "h0:PbSO4:scale" in p
    # Pnma: 直方晶なので角度は固定、長さは独立
    assert p["PbSO4:cell:alpha"].locked and not p["PbSO4:cell:a"].locked
    # 4c (x, 1/4, z): y は固定
    assert p["PbSO4:Pb:y"].locked and not p["PbSO4:Pb:x"].locked
    assert not p["PbSO4:O3:y"].locked  # 一般位置
    assert proj.vary("PbSO4:*:y") == ["PbSO4:O3:y"]
    # 計算と残差
    res = proj.calculate(0)
    assert res.y_calc.shape == proj.histograms[0].data.x.shape
    r = proj.residuals()
    m = proj.histograms[0].mask()
    assert len(r) == int(m.sum())
    assert not np.any((proj.histograms[0].data.x[m] > 30.0) & (proj.histograms[0].data.x[m] < 31.0))
    rf = proj.rfactors()
    assert "h0" in rf or proj.histograms[0].name in rf
    assert "total" in rf
    rep = proj.reflection_report(0, "PbSO4")
    assert len(rep.hkl) > 10 and np.isfinite(rep.r_bragg)


def test_save_load_roundtrip(data_dir: Path, tmp_path: Path) -> None:
    proj = _project(data_dir)
    proj.vary("h0:bkg:*", "h0:PbSO4:scale", "PbSO4:cell:*")
    proj.params["h0:inst:zero"].value = 0.01
    proj.params.apply_all()
    proj.tie("PbSO4:S:uiso", "PbSO4:Pb:uiso")
    y0 = proj.calculate(0).y_calc
    path = tmp_path / "p.kaisetsu.json"
    save_project(proj, path)
    back = load_project(path)
    assert back.name == proj.name
    assert list(back.phases) == ["PbSO4"]
    assert back.params.free_names() == proj.params.free_names()
    assert back.params["PbSO4:S:uiso"].tie is not None
    assert back.histograms[0].instrument.zero == pytest.approx(0.01)
    assert back.histograms[0].excluded == [(30.0, 31.0)]
    assert back.histograms[0].sample.displacement_mm == pytest.approx(0.02)
    assert np.allclose(back.calculate(0).y_calc, y0)


def test_recipe_runs_steps(data_dir: Path) -> None:
    proj = _project(data_dir)
    recipe = (
        Recipe("short")
        .add("bg", ["h0:bkg:*", "h0:PbSO4:scale"], 5)
        .add("cell", ["PbSO4:cell:*", "h0:inst:zero"], 5)
    )
    results = recipe.run(proj)
    assert len(results) == 2
    assert set(proj.params.free_names()) >= {"PbSO4:cell:a", "h0:inst:zero", "h0:bkg:0"}
    rw = [float(line.split("Rwp = ")[1].split("%")[0]) for line in proj.log if "レシピ" in line]
    assert rw[1] <= rw[0] + 1e-9
    std = standard_recipe("PbSO4", "h0")
    assert [s.name for s in std.steps][:2] == ["スケールと背景", "格子定数とピーク位置"]
