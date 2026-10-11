"""Sabine 粉末消衰補正の GSAS-II 突き合わせ(GSASIIstrMath.GetPwdrExt と同じ値になる)。

参照値は `uv run python validation/gsas2/extinction_reference.py`。
GSAS-II の |F|² は (10⁻¹² cm)² 単位(中性子は b を 10⁻¹² cm で持つ)なので、
こちらの fm² 単位には 100 倍して渡す。X 線は電子単位で同じ。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.profile.corrections import sabine_extinction
from tests.conftest import load_reference

pytestmark = pytest.mark.validation


def test_sabine_matches_gsas2() -> None:
    ref = load_reference("sabine_extinction")
    for r in ref["cases"]:
        f2 = r["f2"] if r["xray"] else r["f2"] * 100.0
        e = sabine_extinction(
            np.array([f2]), r["lam"], r["vol"], np.array([r["tt"]]), r["ext"], r["xray"]
        )[0]
        assert e == pytest.approx(r["e"], rel=1e-10), r
