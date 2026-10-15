"""実データ(COD 2240592 の Ba(ClO4)2、pdCIF)で標準手順: 非晶質のこぶを背景のハンプにして採り、
公表値(GSAS、Rwp 6.00%、背景 36 項・選択配向・異方的広がり)に近い Rwp と格子定数になる。

背景 12 項の標準手順は、こぶを吸えないと Rwp 11.5%。「残差のこぶを背景に試す」で 7% 前後まで下がる。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.refine.recipe import standard_recipe

pytestmark = pytest.mark.slow


def test_standard_recipe_adopts_hump_on_baclo4(data_dir: Path) -> None:
    cif = data_dir / "baclo4" / "BaClO4_COD2240592.cif"
    data = read_powder_data(cif)
    inst = CWInstrument(
        Radiation.xray(1.54059), u=0.004, v=-0.003, w=0.004, x=0.02, y=0.02, polarization=0.5
    )
    inst.s_l = inst.h_l = 0.02
    proj = Project("Ba(ClO4)2")
    proj.add_phase(Structure.from_cif(str(cif)), name="B")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=12)
    proj.build_parameters()
    standard_recipe("*", "h0", try_no_kalpha2=False).run(proj)
    assert any("背景のガウスのハンプにしました" in line for line in proj.log)
    hv = {q.name.split(":")[-1]: q.value for q in proj.params if ":hump0:" in q.name}
    assert 15.0 < hv["center"] < 25.0 and 3.0 < hv["fwhm"] < 15.0 and hv["height"] > 300
    rf = proj.rfactors()["total"]
    assert rf.rwp < 0.08  # 公表 6.00% は背景 36 項 + 選択配向 + 異方的広がり
    cell = proj.phases["B"].structure.lattice
    assert (
        abs(cell.a - 14.304) < 0.02 and abs(cell.b - 11.688) < 0.02 and abs(cell.c - 7.2857) < 0.01
    )
