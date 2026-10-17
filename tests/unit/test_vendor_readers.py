# ruff: noqa: E501
import zipfile
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.io import read_powder_data

RAS = """*RAS_DATA_START
*RAS_HEADER_START
*FILE_SAMPLE "test sample"
*HW_XG_TARGET_NAME "Cu"
*MEAS_COND_XG_WAVE_LENGTH_ALPHA1 "1.540593"
*MEAS_COND_XG_WAVE_LENGTH_ALPHA2 "1.544414"
*MEAS_SCAN_AXIS_X "TwoThetaTheta"
*MEAS_SCAN_STEP "0.0200"
*RAS_HEADER_END
*RAS_INT_START
10.0000 120.0000 1.0000
10.0200 118.0000 1.0000
10.0400 125.0000 1.0000
10.0600 300.0000 2.0000
*RAS_INT_END
*RAS_DATA_END
"""

XRDML = """<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5" status="Completed">
  <xrdMeasurement measurementType="Scan" status="Completed">
    <usedWavelength intended="K-Alpha 1">
      <kAlpha1 unit="Angstrom">1.5405980</kAlpha1>
      <kAlpha2 unit="Angstrom">1.5444260</kAlpha2>
      <kBeta unit="Angstrom">1.3922500</kBeta>
      <ratioKAlpha2KAlpha1>0.5000</ratioKAlpha2KAlpha1>
    </usedWavelength>
    <incidentBeamPath><xRayTube><anodeMaterial>Cu</anodeMaterial></xRayTube></incidentBeamPath>
    <scan appendNumber="0" mode="Continuous" scanAxis="Gonio" status="Completed">
      <dataPoints>
        <positions axis="2Theta" unit="deg"><startPosition>10.0</startPosition><endPosition>10.08</endPosition></positions>
        <positions axis="Omega" unit="deg"><startPosition>5.0</startPosition><endPosition>5.04</endPosition></positions>
        <commonCountingTime unit="seconds">1.00</commonCountingTime>
        <intensities unit="counts">100 110 120 130 140</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""


def test_read_ras(tmp_path: Path) -> None:
    p = tmp_path / "a.ras"
    p.write_text(RAS, encoding="utf-8")
    d = read_powder_data(p)
    assert len(d) == 4
    assert d.x[0] == pytest.approx(10.0) and d.x[1] - d.x[0] == pytest.approx(0.02)
    assert d.y[0] == 120.0
    assert d.y[3] == 600.0  # 3 列目の係数 2.0 を掛ける
    assert d.metadata["target"] == "Cu"
    assert d.metadata["wavelength"] == pytest.approx(1.540593)
    assert d.metadata["wavelength2"] == pytest.approx(1.544414)
    assert d.metadata["sample"] == "test sample"


def test_read_rasx(tmp_path: Path) -> None:
    p = tmp_path / "b.rasx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(
            "Data0/Profile0.txt", "10.0000\t50.0\t1.0\n10.0200\t55.0\t1.0\n10.0400\t60.0\t1.0\n"
        )
        zf.writestr(
            "Data0/MesurementConditions0.xml",
            '<?xml version="1.0"?><MeasurementConditions><HardwareConfig><XrayGenerator>'
            "<TargetName>Co</TargetName><WavelengthKalpha1>1.788965</WavelengthKalpha1>"
            "<WavelengthKalpha2>1.792850</WavelengthKalpha2></XrayGenerator></HardwareConfig></MeasurementConditions>",
        )
    d = read_powder_data(p)
    assert len(d) == 3 and d.y[1] == 55.0
    assert d.metadata["target"] == "Co" and d.metadata["wavelength"] == pytest.approx(1.788965)


def test_read_xrdml(tmp_path: Path) -> None:
    p = tmp_path / "c.xrdml"
    p.write_text(XRDML, encoding="utf-8")
    d = read_powder_data(p)
    assert len(d) == 5
    assert np.allclose(d.x, [10.0, 10.02, 10.04, 10.06, 10.08])
    assert d.y[-1] == 140.0
    assert d.metadata["target"] == "Cu"
    assert d.metadata["wavelength"] == pytest.approx(1.540598)
    # intended="K-Alpha 1" は Kα1 単色化のデータ: kAlpha2 が書かれていても Kα2 は出さない
    # (レビュー 2026-10-17 GUI #9)
    assert "wavelength2" not in d.metadata and "ratio2" not in d.metadata
    assert d.metadata["monochromatic"] is True
    # 拡張子が違っても中身で判定する
    q = tmp_path / "c.xml"
    q.write_text(XRDML, encoding="utf-8")
    assert len(read_powder_data(q)) == 5
    # intended="K-Alpha"(Kα1 + Kα2)なら Kα2 の波長と強度比を出す
    r = tmp_path / "d.xrdml"
    r.write_text(XRDML.replace('intended="K-Alpha 1"', 'intended="K-Alpha"'), encoding="utf-8")
    e = read_powder_data(r)
    assert e.metadata["wavelength2"] == pytest.approx(1.544426)
    assert e.metadata["ratio2"] == pytest.approx(0.5) and "monochromatic" not in e.metadata


def test_txt_with_header_and_comma(tmp_path: Path) -> None:
    p = tmp_path / "d.txt"
    p.write_text("Angle,Intensity\n10.00,100\n10.02,105\n10.04,98\n", encoding="utf-8")
    d = read_powder_data(p)
    assert len(d) == 3 and d.y[1] == 105.0
