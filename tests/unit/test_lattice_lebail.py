"""Le Bail で格子定数だけを出す(放射光マイクロ XRD 向け)。

合成データで格子定数と距離ずれを回収する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    GeometryKind,
    Lattice,
    PhaseContribution,
    PowderData,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.refine.lattice import (
    LatticeBatch,
    format_with_esd,
    le_bail_recipe,
    refine_lattice,
    refine_lattice_batch,
)

A_TRUE = 5.4113  # CeO2
LAM = 0.5  # 放射光 25 keV 相当


def _ceo2(a: float) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("F m -3 m"),
        [
            AtomSite("Ce", "Ce", [0, 0, 0], u_iso=0.004),
            AtomSite("O", "O", [0.25, 0.25, 0.25], u_iso=0.008),
        ],
    )


def _instrument() -> CWInstrument:
    # 2 次元検出器を積分した放射光データの典型: ほぼガウス型で幅 0.03° 程度
    return CWInstrument(
        Radiation.xray(LAM), u=0.0002, v=-0.0001, w=0.0008, x=0.003, y=0.002, polarization=0.98
    )


def _simulate(
    a: float, disp_y_mm: float = 0.0, seed: int = 0, peak_counts: float = 2e4
) -> PowderData:
    x = np.arange(4.0, 36.0, 0.01)
    inst = _instrument()
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=300.0, displace_y_mm=disp_y_mm)
    bkg = ChebyshevBackground([200.0, -40.0, 10.0], 4.0, 36.0)
    y = calculate_pattern(x, [PhaseContribution(_ceo2(a), scale=1.0)], inst, smp, bkg).y_calc
    peaks = y - bkg(x)
    y = bkg(x) + peaks * (peak_counts / peaks.max())
    y = np.random.default_rng(seed).poisson(y).astype(float)
    return PowderData(x, y, name=f"sim_a{a:.4f}")


def test_format_with_esd() -> None:
    assert format_with_esd(5.41134, 0.00012) == "5.41134(12)"
    assert format_with_esd(5.41134, 0.0009) == "5.4113(9)"
    assert format_with_esd(158.43, 0.012, 3) == "158.430(12)"
    assert format_with_esd(90.0, None, 3) == "90.000"
    assert format_with_esd(10.2, 1.5) == "10.2(15)"
    assert format_with_esd(4.0, 0.0001) == "4.00000(10)"


def test_recovers_lattice_from_wrong_initial_cell() -> None:
    """初期値 5.40 Å(0.2% ずれ)から 1e-4 Å 以内で戻る。校正済みテンプレート(位置固定)。"""
    data = _simulate(A_TRUE)
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=300.0)
    msgs: list[str] = []
    res = refine_lattice(
        data, [_ceo2(5.40)], _instrument(), smp, names=["CeO2"], position="none",
        widths="sample", progress=msgs.append,
    )  # fmt: skip
    assert res.ok, res.message
    c = res.cells[0]
    assert c.phase == "CeO2" and c.n_reflections > 15
    assert abs(c.values["a"] - A_TRUE) < 1e-4, c.text()
    assert c.esds["a"] is not None and 0 < c.esds["a"] < 1e-4
    assert c.esds["b"] == c.esds["a"]  # 立方晶の拘束
    expect_v = 3 * c.values["a"] ** 2 * c.esds["a"]
    assert c.volume_esd is not None and abs(c.volume_esd - expect_v) < 1e-6 * expect_v
    assert res.chi2 < 2.0 and res.position == {}
    assert "格子定数とピーク位置" in "".join(msgs) and "CeO2:" in msgs[-1]
    assert res.project is not None and res.project.phases["CeO2"].le_bail
    # 抽出強度が 0 に落ちて戻れなくなった反射が無い((533) が消えて χ² 2.4 になっていた)
    assert min(res.project.phases["CeO2"].f2_tables[0].values()) > 0.0
    # 試料の広がりはテンプレートより小さくならない
    inst_fit = res.project.histograms[0].instrument
    assert isinstance(inst_fit, CWInstrument) and inst_fit.x >= 0.003 and inst_fit.y >= 0.002


def test_distance_error_is_absorbed_by_disp_y() -> None:
    """ビーム方向に 0.1 mm ずれたデータ: 位置を固定すると a が 3e-4 ずれ、disp_y を動かすと戻る。"""
    data = _simulate(A_TRUE, disp_y_mm=0.1, seed=3)
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=300.0)
    fixed = refine_lattice(data, [_ceo2(5.41)], _instrument(), smp, position="none")
    moved = refine_lattice(data, [_ceo2(5.41)], _instrument(), smp, position="distance")
    assert fixed.ok and moved.ok
    a_fixed = fixed.cells[0].values["a"]
    a_moved = moved.cells[0].values["a"]
    assert abs(a_fixed - A_TRUE) > 2e-4  # 距離の誤差は格子定数に化ける(Δa/a ≈ Δ/R)
    assert abs(a_moved - A_TRUE) < 1e-4
    ((dy, dy_esd),) = moved.position.values()
    assert abs(dy - 0.1) < 0.02 and dy_esd is not None
    assert any("相関" in w for w in moved.warnings)  # 距離と格子定数の相関を知らせる


def test_batch_to_csv(tmp_path: Path) -> None:
    paths = []
    a_values = [5.4100, 5.4113, 5.4126]
    for i, a in enumerate(a_values):
        d = _simulate(a, seed=10 + i)
        p = tmp_path / f"map_{i}.xy"
        np.savetxt(p, np.column_stack([d.x, d.y]))
        paths.append(p)
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=300.0)
    batch = refine_lattice_batch(
        paths, [_ceo2(5.41)], _instrument(), smp, position="none", names=["CeO2"]
    )
    assert isinstance(batch, LatticeBatch) and len(batch.results) == 3
    assert all(r.ok for r in batch.results)
    got = [r.cells[0].values["a"] for r in batch.results]
    assert np.allclose(got, a_values, atol=1e-4)
    assert all(r.project is None for r in batch.results)
    out = batch.to_csv(tmp_path / "lattice.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("name,ok,Rwp%,chi2,n_points,CeO2:a,esd(CeO2:a),")
    assert "CeO2:V" in lines[0] and len(lines) == 4
    assert "map_1" in lines[2] and "5.4113" in lines[2]
    assert "Rwp" in batch.summary()


def test_failure_is_reported_not_raised(tmp_path: Path) -> None:
    bad = tmp_path / "missing.xy"
    res = refine_lattice(bad, [_ceo2(5.41)], _instrument())
    assert not res.ok and "missing" in res.message
    assert "失敗" in res.summary()


def test_recipe_position_and_width_choices() -> None:
    r = le_bail_recipe("h0", position="distance", widths="instrument",
                       geometry=GeometryKind.DEBYE_SCHERRER)  # fmt: skip
    names = [s.name for s in r.steps]
    assert names == [
        "格子定数の初期値を振る",
        "背景",
        "格子定数とピーク位置",
        "ピーク幅 W",
        "ピーク幅 U, V, X, Y",
        "最終(全パラメータ)",
    ]
    assert "h0:sample:disp_y" in r.steps[2].patterns
    r2 = le_bail_recipe("h0", position="distance", geometry=GeometryKind.BRAGG_BRENTANO)
    assert "h0:sample:displacement" in r2.steps[2].patterns
    r3 = le_bail_recipe("h0", position="distance", is_tof=True, widths="none")
    assert "h0:inst:zero" in r3.steps[2].patterns and len(r3.steps) == 4


@pytest.mark.slow
def test_real_synchrotron_nac(data_dir: Path) -> None:
    """11-BM の NAC(λ 0.4139 Å)を 1〜20° だけで Le Bail: a ≈ 10.25 Å。"""
    from kaisetsu.io import read_powder_data

    data = read_powder_data(data_dir / "nac" / "11BM_NAC.fxye")
    inst = CWInstrument.from_instrument_file(data_dir / "nac" / "11bm_gsas.prm")
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=1000.0)
    res = refine_lattice(
        data, [data_dir / "nac" / "NAC.cif"], inst, smp, x_min=1.0, x_max=20.0,
        position="zero", widths="instrument",
    )  # fmt: skip
    assert res.ok, res.message
    a = res.cells[0].values["a"]
    assert abs(a - 10.25) < 0.01, res.summary()
    assert res.rwp < 0.15  # 装置の幅も動かして 13.3%(.prm の幅のままだと 21%)
