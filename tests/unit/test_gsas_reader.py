"""GSAS 形式リーダーの細部: CONS 略記、FXYE の centidegree、STD の計数回数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.io import read_powder_data

FXYE = """NAC 11-BM test
# comment line
BANK  1    4   4 CONS      50.00      0.10 0 0 FXYE
             50        239.933       16.08
           50.1        244.123       16.18
           50.2         224.02        15.6
           50.3        202.751       14.79
"""

STD = """garnet test
BANK 1 4 1 CONST 2400 5 0 0
 1   162 1   178 2   310 1   166
"""


def test_fxye_with_cons_abbreviation(tmp_path: Path) -> None:
    p = tmp_path / "a.fxye"
    p.write_text(FXYE, encoding="utf-8")
    d = read_powder_data(p)
    assert len(d) == 4
    assert np.allclose(d.x, [0.5, 0.501, 0.502, 0.503])
    assert d.y[0] == pytest.approx(239.933)
    assert d.weight[0] == pytest.approx(1 / 16.08**2)


def test_std_with_count_prefix(tmp_path: Path) -> None:
    p = tmp_path / "g.raw"
    p.write_text(STD, encoding="utf-8")
    d = read_powder_data(p)
    assert np.allclose(d.x, [24.0, 24.05, 24.10, 24.15])
    assert np.allclose(d.y, [162, 178, 310, 166])
    assert d.weight[2] == pytest.approx(2 / 310)
