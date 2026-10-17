"""標準手順のピーク位置の補正の選択(ゼロ点のみ / 試料変位のみ / 両方)。

レビュー 2026-10-17 精密化 #1: 平板(Bragg-Brentano)の標準手順がゼロ点しか動かさず、試料高さの
ずれ(Δ2θ ∝ cos θ)を格子定数が吸っていた(CeO2 の合成データで高さ 0.15 mm → a が −213 ppm、
esd の 575 倍。χ² 1.27 で気付けない)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    Lattice,
    PhaseContribution,
    PowderData,
    Project,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.crystal.cif import read_structure
from kaisetsu.io import read_powder_data
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.refine.lsq import FitResult, position_correlations
from kaisetsu.refine.recipe import (
    POSITION_TAILS,
    Recipe,
    _equivalent_position_start,
    standard_recipe,
)

A0 = 5.4112


def _ceo2() -> Structure:
    return Structure(
        Lattice.cubic(A0),
        SpaceGroup.from_symbol("Fm-3m"),
        [
            AtomSite("Ce", "Ce", [0, 0, 0], u_iso=0.004),
            AtomSite("O", "O", [0.25, 0.25, 0.25], u_iso=0.008),
        ],
        name="CeO2",
    )


def _synthetic(disp_mm: float, zero: float) -> PowderData:
    """Cu Kα1+2、R = 240 mm の平板。真の試料高さずれ disp_mm とゼロ点 zero を入れる。"""
    inst = CWInstrument(
        Radiation.characteristic("Cu"), zero=zero, u=0.004, v=-0.003, w=0.003, x=0.02, y=0.01,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    x = np.arange(15.0, 110.0, 0.03)
    bkg = ChebyshevBackground([150.0, -30.0, 10.0, 0, 0, 0], 15.0, 110.0)
    sample = SampleGeometry(displacement_mm=disp_mm, radius_mm=240.0)
    tr = calculate_pattern(x, [PhaseContribution(_ceo2(), scale=3.0)], inst, sample, bkg)
    y = np.random.default_rng(1).poisson(tr.y_calc).astype(float)
    return PowderData(x, y, name="syn")


def _project(data: PowderData) -> Project:
    """初期値: a を +0.05% ずらし、幅・ゼロ点・変位は既定(未知)から。"""
    st0 = _ceo2()
    st0.lattice = Lattice.cubic(A0 * 1.0005)
    inst0 = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.0, u=0.0, v=0.0, w=0.005, x=0.0, y=0.0,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    p = Project("t")
    p.add_phase(st0, name="CeO2")
    p.add_histogram(data, inst0, SampleGeometry(radius_mm=240.0), n_background=6)
    return p


def _recipe(**kw: object) -> Recipe:
    return standard_recipe(
        coordinates=False,
        adp=False,
        humps=False,
        try_no_kalpha2=False,
        **kw,  # type: ignore[arg-type]
    )


def _adopted(project: Project) -> str:
    line = next(ln for ln in project.log if "ピーク位置の補正は" in ln)
    return line.split("ピーク位置の補正は ")[1].split("(")[0]


def test_auto_position_recovers_lattice_with_sample_height_error() -> None:
    data = _synthetic(0.15, 0.0)
    p = _project(data)
    _recipe().run(p)
    a = p.params["CeO2:cell:a"]
    assert a.esd is not None and abs(a.value - A0) < 5.0 * a.esd
    assert abs(a.value - A0) / A0 < 2e-5
    assert _adopted(p) == "試料変位のみ"
    assert p.params["h0:sample:displacement"].is_free and not p.params["h0:inst:zero"].is_free
    assert p.params["h0:sample:displacement"].value == pytest.approx(0.15, abs=0.01)
    assert p.params["h0:inst:zero"].value == 0.0  # 動かさない補正は開始時の値のまま
    # 旧来の「ゼロ点だけ」では格子定数がずれを吸う(修正前の既定の挙動)
    p0 = _project(data)
    _recipe(position="zero").run(p0)
    a0 = p0.params["CeO2:cell:a"]
    assert a0.esd is not None and abs(a0.value - A0) > 50.0 * a0.esd
    assert (a0.value - A0) / A0 < -1e-4


def test_auto_position_recovers_lattice_with_zero_shift() -> None:
    p = _project(_synthetic(0.0, 0.05))
    _recipe().run(p)
    a = p.params["CeO2:cell:a"]
    assert a.esd is not None and abs(a.value - A0) < 5.0 * a.esd
    assert abs(a.value - A0) / A0 < 2e-5
    assert _adopted(p) == "ゼロ点のみ"
    assert p.params["h0:inst:zero"].value == pytest.approx(0.05, abs=0.003)
    assert p.params["h0:sample:displacement"].value == 0.0


def test_fixed_position_choices_and_legacy_argument() -> None:
    # sample_displacement=True は position="displacement" と同じ(後方互換)
    r_old = standard_recipe("*", "h0", sample_displacement=True)
    r_new = standard_recipe("*", "h0", position="displacement")
    assert [s.name for s in r_old.steps] == [s.name for s in r_new.steps]
    assert r_old.steps[2].patterns == r_new.steps[2].patterns
    assert "ピーク位置の補正を選ぶ" not in [s.name for s in r_new.steps]
    # "both" は平板でゼロ点と高さずれの両方を動かす
    p = _project(_synthetic(0.0, 0.0))
    r = standard_recipe("*", "h0", position="both")
    step = next(s for s in r.steps if s.name == "格子定数とピーク位置")
    assert step.filter is not None
    chosen = [q.name for pat in step.patterns for q in p.params.select(pat) if step.filter(p, q)]
    assert "h0:inst:zero" in chosen and "h0:sample:displacement" in chosen
    assert "h0:sample:disp_x" not in chosen


def test_equivalent_start_carries_zero_shift_into_displacement() -> None:
    """前の段階でゼロ点が吸ったずれを、試料変位の候補はそれと同じピーク位置になる値から始める。"""
    p = _project(_synthetic(0.0, 0.0))
    h = p.histograms[0]
    initial = {"h0:inst:zero": 0.0, "h0:sample:displacement": 0.0}
    start = {"h0:inst:zero": -0.05, "h0:sample:displacement": 0.0}
    eq = _equivalent_position_start(p, ["h0:sample:displacement"], initial, start)
    s = eq["h0:sample:displacement"]
    assert s > 0.0  # ゼロ点 −0.05° に相当する正の高さずれ(Δ2θ = −2 s cos θ / R)
    x = h.data.x[h.mask()]
    from dataclasses import replace

    shift_eq = replace(h.sample, displacement_mm=s).position_shift(x)
    assert float(np.sqrt(np.mean((shift_eq + 0.05) ** 2))) < 0.01  # 範囲全体で 0.01° 以内
    # 候補が「両方」なら今の値のまま。候補がゼロ点だけで変位が吸っていたなら定数に戻す
    both = _equivalent_position_start(p, list(start), initial, start)
    assert both["h0:inst:zero"] == pytest.approx(-0.05, abs=1e-9)
    assert both["h0:sample:displacement"] == pytest.approx(0.0, abs=1e-9)
    back = _equivalent_position_start(
        p, ["h0:inst:zero"], initial, {"h0:inst:zero": 0.0, "h0:sample:displacement": s}
    )
    assert back["h0:inst:zero"] == pytest.approx(-0.05, abs=0.01)


def test_trial_is_skipped_for_tof_only_project(data_dir: Path) -> None:
    proj = Project("NAC TOF")
    proj.add_phase(read_structure(data_dir / "nac" / "NAC.cif"), name="NAC")
    data = read_powder_data(data_dir / "nac" / "PG3_22048.gsa")
    inst = TOFInstrument.from_instprm(data_dir / "nac" / "POWGEN_1066.instprm")
    proj.add_histogram(data, inst, n_background=4, x_min=10000.0, x_max=60000.0)
    recipe = Recipe("t").add(
        "ピーク位置の補正を選ぶ",
        ["*:cell:*", *(f"h0:{tail}" for tail in POSITION_TAILS)],
        5,
        kind="position_trial",
    )
    results = recipe.run(proj, reset=False)
    assert results == [] and proj.last_fit is None
    assert any("試す候補が無い" in line for line in proj.log)


def test_position_correlation_is_reported_in_diagnostics() -> None:
    p = _project(_synthetic(0.0, 0.0))
    names = ["CeO2:cell:a", "h0:inst:zero", "h0:sample:displacement"]
    corr = np.eye(3)
    corr[1, 2] = corr[2, 1] = 0.99
    fit = FitResult(
        names=names, values=np.zeros(3), esd=np.ones(3), covariance=np.eye(3), correlation=corr,
        chi2=1.0, n_data=100, n_params=3, n_evaluations=1, success=True, message="ok",
        elapsed=0.0, status=2,
    )  # fmt: skip
    assert position_correlations(fit) == [("h0:inst:zero", "h0:sample:displacement", 0.99)]
    assert position_correlations(None) == []
    p.last_fit = fit
    assert any("ゼロ点と試料変位は区別できていません" in n for n in p.diagnostics())
    corr[1, 2] = corr[2, 1] = 0.5
    assert position_correlations(fit) == []
    assert not any("区別できていません" in n for n in p.diagnostics())
