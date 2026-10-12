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
