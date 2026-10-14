"""標準手順で FAP を精密化しても、ガウス幅 Γ_G² がデータの端で負にならない。

修正前は U = −0.0022, V = +0.0040, W = 0.0006 となり 2θ ≥ 125.4° で負のまま「収束」していた。
原因は「原子座標」の段階の `*:*:z` が h0:inst:z(ローレンツ幅 Z)まで動かしていたこと。
修正後は U = −0.0011, V = +0.0024, W = 0.0010 で全範囲で正。負になれば標準手順が U を固定して
やり直す(Recipe._guard_widths)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.pattern.instrument import CWInstrument
from kaisetsu.refine.width_check import width_violations

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]


def test_fap_standard_recipe_keeps_gauss_width_positive() -> None:
    sys.path.insert(0, str(ROOT / "examples"))
    from refine_sample_data import setup_fap

    project, recipe = setup_fap()
    recipe.run(project)
    inst = project.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    h = project.histograms[0]
    xs = np.linspace(float(h.data.x[0]), float(h.data.x[-1]), 500)
    assert inst.gauss_fwhm2(xs).min() > -1e-6
    assert [v for v in width_violations(project) if v.kind == "gauss"] == []
    # 自動対処が働いたなら U は固定されてログに残る
    guarded = any("を固定してやり直します" in line for line in project.log)
    if guarded:
        assert not project.params["h0:inst:u"].vary
    rf = project.rfactors()["total"]
    assert rf.rwp < 0.09  # 修正前 8.48%。妥当な解のまま


def test_guard_cascades_to_v_and_w_for_lorentzian_peaks() -> None:
    """ガウス成分の無い(ローレンツだけの)ピークでは、標準手順が U → V → W の順に固定して
    ガウス幅を負のまま終わらせない。

    Z-Rietveld サンプルの CaF2 実験室データで V = −1.8, W = −0.43 まで流れて全範囲で負になり、
    V, W の微分が 0 で共分散が特異になって |ρ| > 1 の相関が表示されていた。
    """
    from kaisetsu.api import Project, Radiation, SampleGeometry, Structure
    from kaisetsu.pattern.histogram import PowderData
    from kaisetsu.refine.recipe import standard_recipe

    st = Structure.from_cif(str(ROOT / "data" / "public" / "lab6" / "LaB6_COD1000055.cif"))
    x = np.arange(20.0, 90.0, 0.02)
    truth = CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5), u=0.0, v=0.0, w=1e-6, x=0.08, y=0.02)
    sim = Project("sim")
    sim.add_phase(st.copy(), name="L")
    sim.add_histogram(PowderData(x, np.ones_like(x)), truth, SampleGeometry(), n_background=2)
    sim.build_parameters()
    y_peaks = sim.calculate(0).y_calc
    y_true = 3000.0 * y_peaks / y_peaks.max() + 20.0
    y = np.random.default_rng(1).poisson(y_true).astype(np.float64)

    proj = Project("fit")
    proj.add_phase(st.copy(), name="L")
    proj.add_histogram(
        PowderData(x, y),
        CWInstrument(Radiation.xray(1.54056, 1.5444, 0.5)),
        SampleGeometry(),
        n_background=4,
    )
    proj.build_parameters()
    standard_recipe("*", "h0", coordinates=False, adp=False).run(proj)
    inst = proj.histograms[0].instrument
    assert isinstance(inst, CWInstrument)
    assert inst.gauss_fwhm2(x).min() >= 0.0
    assert [v for v in width_violations(proj) if v.kind == "gauss"] == []
    guards = [line for line in proj.log if "を固定してやり直します" in line]
    assert guards, proj.log
    assert not proj.params["h0:inst:u"].vary
    assert proj.rfactors()["total"].chi2 < 1.5
    fit = proj.last_fit
    assert fit is not None
    assert np.all(np.abs(fit.correlation) <= 1.0 + 1e-12)
    assert fit.insensitive == []
