"""同じ名前のヒストグラムが R 因子の辞書で衝突しないこと(レビュー 2026-10-17 精密化 #6)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kaisetsu.api import CWInstrument, PowderData, Project, Radiation, SampleGeometry, Structure
from kaisetsu.io import read_powder_data
from kaisetsu.pattern.instrument import GeometryKind


def _data(name: str, n: int = 400, **meta: object) -> PowderData:
    x = np.linspace(20.0, 60.0, n)
    y = 100.0 + 10.0 * np.sin(x)
    return PowderData(x, y, name=name, metadata=dict(meta))


def test_same_file_twice_gets_distinct_rfactor_keys(data_dir: Path, pbso4_cif: Path) -> None:
    p = Project("t")
    p.add_phase(Structure.from_cif(str(pbso4_cif)), name="PbSO4")
    xra = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA")
    inst = CWInstrument(Radiation.characteristic("Cu"), w=0.01)
    p.add_histogram(xra, inst, SampleGeometry(), x_max=40.0)
    p.add_histogram(xra, inst, SampleGeometry(), x_max=60.0)
    names = [h.name for h in p.histograms]
    assert names[0] == "PBSO4" and names[1] == "PBSO4 (2)"
    rf = p.rfactors()
    assert set(rf) == {"PBSO4", "PBSO4 (2)", "total"}
    assert rf["PBSO4"].n_points == int(p.histograms[0].mask().sum())
    assert rf["PBSO4 (2)"].n_points == int(p.histograms[1].mask().sum())
    assert rf["PBSO4"].n_points < rf["PBSO4 (2)"].n_points
    assert "PBSO4 (2)" in p.summary()


def test_same_stem_different_radiation(data_dir: Path, pbso4_cif: Path) -> None:
    p = Project("t")
    p.add_phase(Structure.from_cif(str(pbso4_cif)), name="PbSO4")
    p.add_histogram(
        read_powder_data(data_dir / "pbso4" / "PBSO4.XRA"),
        CWInstrument(Radiation.characteristic("Cu"), w=0.01),
        SampleGeometry(),
    )
    p.add_histogram(
        read_powder_data(data_dir / "pbso4" / "PBSO4.CWN"),
        CWInstrument(Radiation.neutron(1.909), w=0.1),
        SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER),
    )
    assert [h.name for h in p.histograms] == ["PBSO4", "PBSO4 (2)"]
    rf = p.rfactors()
    for h in p.histograms:
        assert rf[h.name].n_points == int(h.mask().sum())


def test_reserved_names_and_gsas_banks() -> None:
    p = Project("t")
    inst = CWInstrument(Radiation.characteristic("Cu"), w=0.01)
    # "total" と "g0" 形式は全体の R 因子・PDF データセットのキーなので避ける
    assert p.add_histogram(_data("total"), inst).name == "total (2)"
    assert p.add_histogram(_data("g0"), inst).name == "g0 (2)"
    assert p.add_histogram(_data("g12"), inst).name == "g12 (2)"
    assert p.add_histogram(_data("gx"), inst).name == "gx"
    # GSAS の複数 BANK は metadata["bank"] で "{stem} bank{k}"
    assert p.add_histogram(_data("multi", bank=1), inst).name == "multi"
    assert p.add_histogram(_data("multi", bank=2), inst).name == "multi bank2"
    assert p.add_histogram(_data("multi", bank=2), inst).name == "multi (2)"
    # 明示した名前も一意にする
    assert p.add_histogram(_data("x"), inst, name="total").name == "total (3)"
    assert len({h.name for h in p.histograms}) == len(p.histograms)
    assert set(p.rfactors()) == {h.name for h in p.histograms} | {"total"}
