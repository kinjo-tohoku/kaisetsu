"""装置テンプレート(*.kaisetsu-inst.json)の保存と読み込み、プリセットからの装置生成。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaisetsu.io.project import (
    INSTRUMENT_TEMPLATE_SUFFIX,
    load_instrument_template,
    save_instrument_template,
)
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.scattering import Radiation


def test_cw_template_round_trip(tmp_path: Path) -> None:
    inst = CWInstrument(
        Radiation.xray(0.4959), zero=0.0012, u=0.0003, v=-0.0002, w=0.0009, x=0.004, y=0.001,
        s_l=0.005, h_l=0.005, polarization=0.98,
    )  # fmt: skip
    smp = SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER, radius_mm=350.0, displace_y_mm=0.07)
    path = save_instrument_template(tmp_path / f"beamline{INSTRUMENT_TEMPLATE_SUFFIX}", inst, smp)
    assert path.exists() and "kaisetsu-instrument-template" in path.read_text(encoding="utf-8")
    inst2, smp2 = load_instrument_template(path)
    assert isinstance(inst2, CWInstrument)
    assert inst2 == inst
    assert smp2 == smp


def test_tof_template_round_trip(tmp_path: Path) -> None:
    inst = TOFInstrument(difc=22000.0, zero=-1.5, sig1=50.0)
    path = save_instrument_template(tmp_path / "tof.json", inst, None)
    inst2, smp2 = load_instrument_template(path)
    assert isinstance(inst2, TOFInstrument) and inst2.difc == 22000.0 and inst2.zero == -1.5
    assert smp2.kind is GeometryKind.BRAGG_BRENTANO  # 省略時の既定


def test_template_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_instrument_template(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="instrument"):
        load_instrument_template(bad)


def test_instrument_from_preset() -> None:
    pytest.importorskip("PySide6")
    from kaisetsu.pattern.presets import PRESETS, instrument_from_preset

    for name in PRESETS:
        inst, smp = instrument_from_preset(name)
        assert isinstance(inst, CWInstrument) and inst.w > 0
    inst, smp = instrument_from_preset("放射光 X 線 (キャピラリ, Debye-Scherrer)")
    assert inst.radiation.wavelength == 0.5 and inst.radiation.wavelength2 is None
    assert smp.kind is GeometryKind.DEBYE_SCHERRER
    inst, smp = instrument_from_preset("実験室 X 線 Cu Kα (Bragg-Brentano, Ni フィルタ)")
    assert inst.radiation.wavelength2 is not None and smp.kind is GeometryKind.BRAGG_BRENTANO
