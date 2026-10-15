"""標準手順の「残差のこぶを背景に試す」: ガラスキャピラリのような広いこぶを背景のハンプにして採る。

合成 LaB6(最大 3000、背景 20)に幅 10° のガウスのこぶ(高さ 900)を乗せたデータで、
(1) 標準手順がこぶを見つけてハンプを採用し雑音だけが残ること、(2) こぶの無いデータでは
ハンプを足さないこと、(3) 人が「背景の設定」で足したハンプが「スケールと背景」で動くことを確かめる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.refine.recipe import standard_recipe

ROOT = Path(__file__).resolve().parents[2]
LAB6_CIF = ROOT / "data" / "public" / "lab6" / "LaB6_COD1000055.cif"
HUMP_CENTER, HUMP_FWHM, HUMP_HEIGHT = 40.0, 10.0, 900.0


def _widths() -> dict[str, float]:
    return {"u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02}


def _synthetic(with_hump: bool, seed: int = 3) -> PowderData:
    st = Structure.from_cif(str(LAB6_CIF))
    truth = CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **_widths())
    x = np.arange(20.0, 75.0, 0.02)
    sim = Project("sim")
    sim.add_phase(st.copy(), name="L")
    sim.add_histogram(PowderData(x, np.ones_like(x)), truth, SampleGeometry(), n_background=2)
    sim.build_parameters()
    y_peaks = sim.calculate(0).y_bragg
    y_true = 3000.0 * y_peaks / y_peaks.max() + 20.0 + 0.3 * x
    if with_hump:
        sigma = HUMP_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        y_true = y_true + HUMP_HEIGHT * np.exp(-0.5 * ((x - HUMP_CENTER) / sigma) ** 2)
    y = np.random.default_rng(seed).poisson(y_true).astype(np.float64)
    return PowderData(x, y)


def _project(data: PowderData, n_background: int = 4) -> Project:
    proj = Project("fit")
    proj.add_phase(Structure.from_cif(str(LAB6_CIF)).copy(), name="L")
    proj.add_histogram(
        data,
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), **_widths()),
        SampleGeometry(),
        n_background=n_background,
    )
    proj.build_parameters()
    return proj


def _recipe(**kw: bool) -> object:
    return standard_recipe("*", "h0", coordinates=False, adp=False, try_no_kalpha2=False, **kw)


def _hump_values(proj: Project) -> dict[str, float]:
    return {q.name.split(":")[-1]: q.value for q in proj.params if ":hump0:" in q.name}


def test_hump_trial_adopts_glass_hump() -> None:
    proj = _project(_synthetic(with_hump=True))
    _recipe().run(proj)  # type: ignore[attr-defined]
    assert any("背景のガウスのハンプにしました" in line for line in proj.log)
    hv = _hump_values(proj)
    assert abs(hv["center"] - HUMP_CENTER) < 1.0
    assert abs(hv["fwhm"] - HUMP_FWHM) < 2.0
    assert abs(hv["height"] - HUMP_HEIGHT) < 0.2 * HUMP_HEIGHT
    assert proj.rfactors()["total"].chi2 < 1.6  # 雑音だけが残る
    # 採用したハンプは以後の段階でも動く(最終段階で自由のまま)
    assert all(q.vary for q in proj.params if ":hump0:" in q.name)


def test_hump_trial_leaves_clean_data_alone() -> None:
    proj = _project(_synthetic(with_hump=False))
    _recipe().run(proj)  # type: ignore[attr-defined]
    assert not any(":hump" in q.name for q in proj.params)
    assert any("広いこぶは無いので飛ばします" in line for line in proj.log) or any(
        "ハンプにしても良くなりません" in line for line in proj.log
    )
    assert proj.rfactors()["total"].chi2 < 1.6


def test_manual_hump_is_refined_in_background_step() -> None:
    """「背景の設定」で粗く足したハンプが、標準手順で正しい位置・幅に動く(以前は動かなかった)。"""
    proj = _project(_synthetic(with_hump=True))
    proj.add_hump(0, HUMP_CENTER - 3.0, HUMP_FWHM * 0.6, 0.5 * HUMP_HEIGHT)
    _recipe(humps=False).run(proj)  # type: ignore[attr-defined]
    hv = _hump_values(proj)
    assert abs(hv["center"] - HUMP_CENTER) < 1.0
    assert abs(hv["fwhm"] - HUMP_FWHM) < 2.0
    assert proj.rfactors()["total"].chi2 < 1.6
