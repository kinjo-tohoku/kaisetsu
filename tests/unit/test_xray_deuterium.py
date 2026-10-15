"""X 線の散乱因子: 重水素 D は H と同じ係数を使う(中性子用の CIF を X 線で読んだとき落ちない)。"""

from __future__ import annotations

import numpy as np

from kaisetsu.scattering.xray import XrayScatteringFactors


def test_deuterium_uses_hydrogen_coefficients() -> None:
    tab = XrayScatteringFactors()
    assert tab.resolve("D") == "H"
    assert tab.resolve("T") == "H"
    s = np.array([0.0, 0.3, 0.6])
    assert np.allclose(tab.f0("D", s), tab.f0("H", s))
    assert XrayScatteringFactors.atomic_number("D") == 1
