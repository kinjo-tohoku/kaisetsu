"""標準手順の改善: Kα2 なしの試行、軽原子 U_iso の共通化、放射光の非対称、対象の無い段階の省略。"""

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
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pattern.instrument import GeometryKind
from kaisetsu.refine.recipe import (
    LIGHT_ATOM_FRACTION,
    Recipe,
    Step,
    _is_bragg_brentano,
    _tie_light_atom_uiso,
    light_atom_uiso_groups,
    standard_recipe,
)

ROOT = Path(__file__).resolve().parents[2]
LAB6_CIF = ROOT / "data" / "public" / "lab6" / "LaB6_COD1000055.cif"


def _synthetic(st: Structure, truth: CWInstrument, *, seed: int = 1) -> PowderData:
    """truth の装置で計算した LaB6 のパターンにポアソン雑音を乗せる(最大 3000、背景 20)。"""
    x = np.arange(20.0, 75.0, 0.02)
    sim = Project("sim")
    sim.add_phase(st.copy(), name="L")
    sim.add_histogram(PowderData(x, np.ones_like(x)), truth, SampleGeometry(), n_background=2)
    sim.build_parameters()
    y_peaks = sim.calculate(0).y_bragg
    y_true = 3000.0 * y_peaks / y_peaks.max() + 20.0
    y = np.random.default_rng(seed).poisson(y_true).astype(np.float64)
    return PowderData(x, y)


def _fit_project(data: PowderData, inst: CWInstrument) -> Project:
    st = Structure.from_cif(str(LAB6_CIF))
    proj = Project("fit")
    proj.add_phase(st.copy(), name="L")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def _widths() -> dict[str, float]:
    return {"u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02}


def test_kalpha2_trial_drops_kalpha2_for_monochromatic_data() -> None:
    """Kα1 だけのデータに Kα2 入りの装置で始めても、試行で Kα2 が外れる。"""
    st = Structure.from_cif(str(LAB6_CIF))
    data = _synthetic(st, CWInstrument(Radiation.xray(1.54056), **_widths()))
    proj = _fit_project(data, CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **_widths()))
    standard_recipe("*", "h0", coordinates=False, adp=False).run(proj)
    inst = proj.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    assert inst.radiation.ratio2 == 0.0 and inst.radiation.wavelength2 is not None
    assert any("Kα2 を外しました" in line for line in proj.log)
    assert proj.rfactors()["total"].chi2 < 1.5  # 雑音だけが残る(Rwp ≈ R_exp)


def test_kalpha2_trial_keeps_kalpha2_when_data_has_it() -> None:
    """Kα2 入りのデータでは試行しても戻し、比も値も元のまま。"""
    st = Structure.from_cif(str(LAB6_CIF))
    truth = CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **_widths())
    data = _synthetic(st, truth)
    proj = _fit_project(data, CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **_widths()))
    r = standard_recipe("*", "h0", coordinates=False, adp=False)
    names = [s.name for s in r.steps]
    k = names.index("Kα2 なしを試す")
    Recipe("前半", r.steps[:k]).run(proj)
    before = proj.snapshot()
    rwp_before = proj.rfactors()["total"].rwp
    Recipe("試行", [r.steps[k]]).run(proj, reset=False)
    inst = proj.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    assert inst.radiation.ratio2 == 0.5
    assert any("Kα2 ありのまま進めます" in line for line in proj.log)
    after = proj.snapshot()
    assert all(abs(after[n] - before[n]) < 1e-12 for n in before)
    assert proj.rfactors()["total"].rwp == pytest.approx(rwp_before, rel=1e-9)


def test_kalpha2_trial_is_skipped_without_kalpha2() -> None:
    st = Structure.from_cif(str(LAB6_CIF))
    data = _synthetic(st, CWInstrument(Radiation.xray(0.5), w=0.001, x=0.005, y=0.005))
    proj = _fit_project(data, CWInstrument(Radiation.xray(0.5), w=0.001, x=0.005, y=0.005))
    proj.vary("h0:bkg:*", "h0:L:scale")
    r = standard_recipe("*", "h0")
    step = next(s for s in r.steps if s.kind == "kalpha2_trial")
    assert Recipe("試行", [step]).run(proj, reset=False) == []
    assert any("Kα2 の無い装置なので飛ばします" in line for line in proj.log)


def test_light_atom_uiso_is_tied_when_heavy_atoms_dominate(data_dir: Path) -> None:
    """PbSO4(Pb が散乱の 3/4)では O の U_iso が共通になり、LaB6(B が 1/3)ではならない。"""
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    proj = Project("pbso4")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, CWInstrument(Radiation.xray(1.5405, 1.5443, 0.5), w=0.003))
    proj.build_parameters()
    groups = light_atom_uiso_groups(proj)
    assert len(groups) == 1
    phase, labels, frac = groups[0]
    assert phase == "PbSO4" and len(labels) == 3 and all(lab.startswith("O") for lab in labels)
    assert 0.2 < frac < LIGHT_ATOM_FRACTION
    msgs = _tie_light_atom_uiso(proj, Step("変位パラメータ", ()))
    assert msgs and "U_iso を共通にします" in msgs[0]
    master = f"PbSO4:{labels[0]}:uiso"
    for lab in labels[1:]:
        p = proj.params[f"PbSO4:{lab}:uiso"]
        assert p.tie is not None and p.tie.masters == (master,)
        assert p.value == pytest.approx(proj.params[master].value)
    assert proj.params[master].value > 0
    # 2 回目は何もしない(すでに従属)
    assert _tie_light_atom_uiso(proj, Step("変位パラメータ", ())) == []
    # 標準手順の「変位パラメータ」段階が準備として呼ぶ
    r = standard_recipe("*", "h0")
    step = next(s for s in r.steps if s.name == "変位パラメータ")
    assert step.prepare is _tie_light_atom_uiso
    assert standard_recipe("*", "h0", light_atom_uiso=False).steps[-2].prepare is None

    lab6 = Project("lab6")
    lab6.add_phase(Structure.from_cif(str(LAB6_CIF)), name="L")
    lab6.add_histogram(data, CWInstrument(Radiation.xray(1.5405, 1.5443, 0.5), w=0.003))
    lab6.build_parameters()
    (_, b_labels, b_frac) = light_atom_uiso_groups(lab6)[0]
    assert b_frac >= LIGHT_ATOM_FRACTION
    assert _tie_light_atom_uiso(lab6, Step("変位パラメータ", ())) == []
    assert all(lab6.params[f"L:{lab}:uiso"].tie is None for lab in b_labels)


def test_asymmetry_step_is_automatic_for_synchrotron_only() -> None:
    st = Structure.from_cif(str(LAB6_CIF))
    x = np.arange(5.0, 30.0, 0.01)
    proj = Project("mixed")
    proj.add_phase(st, name="L")
    proj.add_histogram(
        PowderData(x, np.ones_like(x)),
        CWInstrument(Radiation.xray(0.5), w=0.001, s_l=0.005, h_l=0.005),
        SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER),
        n_background=2,
    )
    xl = np.arange(20.0, 60.0, 0.02)
    proj.add_histogram(
        PowderData(xl, np.ones_like(xl)),
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), w=0.004, s_l=0.02, h_l=0.02),
        SampleGeometry(),
        n_background=2,
    )
    proj.build_parameters()
    r = standard_recipe("*", "h*")
    step = next(s for s in r.steps if s.name == "非対称(放射光)")
    assert step.filter is not None
    chosen = [
        p.name for pat in step.patterns for p in proj.params.select(pat) if step.filter(proj, p)
    ]
    assert chosen == ["h0:inst:sh_l"]
    names = [s.name for s in standard_recipe("*", "h*", asymmetry=True).steps]
    assert "非対称" in names and "非対称(放射光)" not in names
    names = [s.name for s in standard_recipe("*", "h*", asymmetry=False).steps]
    assert "非対称" not in names and "非対称(放射光)" not in names


def test_recipe_skips_steps_without_target_parameters(data_dir: Path) -> None:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 30.0)
    proj = Project("ds")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(
        data,
        CWInstrument(Radiation.xray(1.5405, 1.5443, 0.5), w=0.003),
        SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER),
        n_background=3,
    )
    proj.build_parameters()
    r = (
        Recipe("t")
        .add("bg", ["h0:bkg:*", "h0:PbSO4:scale"], 3)
        .add("表面粗さ", ["h0:sample:rough_a"], 3, filter=_is_bragg_brentano)
    )
    results = r.run(proj)
    assert len(results) == 1
    assert any("段階 '表面粗さ': 対象のパラメータが無いので飛ばします" in line for line in proj.log)


def test_standard_recipe_step_names_and_options() -> None:
    names = [s.name for s in standard_recipe("*", "h0").steps]
    assert names == [
        "スケールと背景",
        "格子定数とピーク位置",
        "ピーク幅 W",
        "ピーク幅 U, V, X, Y",
        "残差のこぶを背景に試す",
        "Kα2 なしを試す",
        "非対称(放射光)",
        "原子座標",
        "変位パラメータ",
        "最終(全パラメータ)",
    ]
    names = [s.name for s in standard_recipe("*", "h0", try_no_kalpha2=False).steps]
    assert "Kα2 なしを試す" not in names
    names = [s.name for s in standard_recipe("*", "h0", humps=False).steps]
    assert "残差のこぶを背景に試す" not in names
    # 人が足したハンプは「スケールと背景」で動く
    assert "h0:hump*:*" in standard_recipe("*", "h0").steps[0].patterns
