"""他のプログラムの例題の形式: RIETAN-FP、FullProf、GSAS ALT、Z-Rietveld、GSAS 装置ファイル。

実ファイルは配布物(RIETAN-FP、FullProf、Z-Rietveld、GSAS-II チュートリアル)で確認済み。
ここでは同じ構造の小さなファイルを作って読む(形式判定は 5 点以上を前提にしている)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.io import read_powder_data
from kaisetsu.io.readers.fullprof import detect_fullprof
from kaisetsu.io.readers.rietan import is_rietan_block
from kaisetsu.pattern.instrument import CWInstrument


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_rietan_block_format(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "Cu3Fe4P6.int",
        "* Cu3Fe4(PO4)6, CuKa1\n   12    10.000     0.010\n"
        "   9104   9375   9335   9396   9149   9450   9180   9274   9158   9382   9192\n"
        "   9115\n",
    )
    assert is_rietan_block(p.read_text())
    d = read_powder_data(p)
    assert len(d) == 12
    assert d.x[0] == pytest.approx(10.0) and d.x[-1] == pytest.approx(10.11)
    assert d.y[1] == 9375 and d.metadata["format"] == "rietan"
    assert d.metadata["comment"].startswith("Cu3Fe4(PO4)6")
    # 1 行 1 個でも読める
    body = "".join(f" {100 + i}\n" for i in range(12))
    q = _write(tmp_path, "BiCoO3.int", "12   2.5   0.05\n" + body)
    e = read_powder_data(q)
    assert len(e) == 12 and e.x[1] == pytest.approx(2.55) and e.y[11] == 111


def test_rietan_general_and_igor_still_go_to_xy(tmp_path: Path) -> None:
    gen = _write(
        tmp_path, "Fap.int", "GENERAL\n 4\n 15.00 609\n 15.02 612\n 15.04 600\n 15.06 630\n"
    )
    d = read_powder_data(gen)
    assert len(d) == 4 and d.x[0] == pytest.approx(15.0) and d.y[3] == 630
    igor = _write(
        tmp_path,
        "LaB6.histogramIgor",
        "IGOR\nWAVES/O twotheta, yint\nBEGIN\n10.0 155.\n10.01 152.\n10.02 154.\nEND\n",
    )
    e = read_powder_data(igor)
    assert len(e) == 3 and e.y[0] == 155
    tof = _write(
        tmp_path,
        "CeO2.histogramIgor",
        "IGOR\nWAVES tof, yint, yerr, nc\nBEGIN\n"
        " 1501 0.0719 0.0203 1\n 1503 0.1148 0.0212 1\n 1505 0.0756 0.0203 1\nEND\n",
    )
    f = read_powder_data(tof)
    assert len(f) == 3 and f.x[0] == 1501 and f.weight[0] == pytest.approx(1 / 0.0203**2)


def test_fullprof_free_format(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "b8p.dat",
        "   5.000  0.02   5.18  title of the sample\n"
        "    422.    424.    397.    377.    412.    424.    397.\n    401.    390.    380.\n",
    )
    assert detect_fullprof(p.read_text()) == "free"
    assert not is_rietan_block(p.read_text())
    d = read_powder_data(p)
    assert len(d) == 10 and d.x[-1] == pytest.approx(5.18) and d.y[0] == 422
    assert d.metadata["comment"] == "title of the sample"
    # コメント・空行が先にあり、題名が別行でも読める(ceo2.dat、HRPT の形)
    q = _write(
        tmp_path,
        "ceo2.dat",
        "\n! Diffraction pattern from CeO2\n#\n     25.000   0.025  25.150\n"
        "      32      39      32      32      32      30      31\n",
    )
    e = read_powder_data(q)
    assert len(e) == 7 and e.x[0] == pytest.approx(25.0) and e.x[-1] == pytest.approx(25.15)
    r = _write(
        tmp_path,
        "hrpt.dat",
        "HRPT, K2NaAg3(CN)6\nlambda= 1.8856, T= 1.5\n"
        '   4.950   0.050   5.250 4500000., sample="x"\n'
        "    514.    545.    580.    578.    572.    560.    571.\n",
    )
    f = read_powder_data(r)
    assert len(f) == 7 and f.x[2] == pytest.approx(5.05)


def test_fullprof_detector_pairs(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "Ca2MnO4.dat",
        "    6.00    0.05  6.30\n 3   123 3   119 3   114 3   116 3   128 3   130 3   120\n",
    )
    assert detect_fullprof(p.read_text()) == "pairs"
    d = read_powder_data(p)
    assert len(d) == 7
    # 強度は I のまま、分散は I/n(GSAS の STD と同じ扱い)
    assert d.y[0] == pytest.approx(123) and d.weight[0] == pytest.approx(3 / 123)
    old = _write(
        tmp_path,
        "pbso4.dat",
        " PbSO4 D1A(ILL) Round Robin\n     100       0   0.050      10       0       0\n  10.000\n"
        "  10000.      0.    0.00    0.00\n"
        " 1   220 1   214 1   219 1   224 1   198 1   200 1   210\n",
    )
    assert detect_fullprof(old.read_text()) == "ill_d1a"
    e = read_powder_data(old)
    assert len(e) == 7 and e.x[0] == pytest.approx(10.0) and e.x[1] == pytest.approx(10.05)
    assert e.y[1] == 214 and e.metadata["comment"].startswith("PbSO4")


def test_gsas_alt_format(tmp_path: Path) -> None:
    """ALT は (TOF×32 F8.0, 強度 F7.4, σ F5.4) × 4 / 行。TOF は他の GSAS 形式と同じくビン端 → 中心、
    強度はビン幅あたりに直す。"""
    p = _write(
        tmp_path,
        "arg_si.dat",
        " Si Std\nBANK 1  6 2 CONST   2000.     5.     0.     0. ALT\n"
        "  64000.  213.8  3.3  64160.  213.8  3.3  64320.  210.1  3.3  64480.  213.3  3.3\n"
        "  64640.  211.9  3.3  64800.  212.8  3.3\n",
    )
    d = read_powder_data(p)
    assert d.metadata["xunit"] == "tof_us"
    assert len(d) == 5  # 6 個のビン端 → 5 個の中心
    assert d.x[0] == pytest.approx(2002.5) and d.x[1] == pytest.approx(2007.5)
    assert d.y[2] == pytest.approx(210.1 / 5.0)
    # RALF + 小数点なし(暗黙の小数桁 4)
    q = _write(
        tmp_path,
        "pearl.dat",
        "CeO2 PEARL\nBANK 1  4 1 RALF    48008   64    64331 0.0010 ALT\n"
        "   48008  69433 1704   48072  67465 1682   48136  66487 1671   48200  71298 1739\n",
    )
    e = read_powder_data(q)
    assert len(e) == 3
    assert e.x[0] == pytest.approx((48008 + 48072) / 2 / 32)
    assert e.y[0] == pytest.approx(6.9433 / 2.0)


def test_xy_merges_duplicate_x(tmp_path: Path) -> None:
    p = _write(tmp_path, "joined.xy", "! two scans joined\n10.0 100\n10.5 110\n10.5 130\n11.0 90\n")
    d = read_powder_data(p)
    assert len(d) == 3 and d.x[1] == pytest.approx(10.5) and d.y[1] == pytest.approx(120.0)


_PRM_11BM = """            123456789012345678901234567890123456789012345678901234567890
INS   BANK      1
INS   HTYPE   PXCR
INS  1 ICONS 0.4132590    0.0000    0.0000               0.990    0     0.500
INS  1 IRAD     0
INS  1PRCF1     3   19   0.00100
INS  1PRCF11       1.163000      -0.126000       0.063000   0.000000E+00
INS  1PRCF12       0.173000       0.000000       0.001100       0.001100
"""
_INST_BT1 = """INS   BANK      1
INS   HTYPE   PNCR
INS  1 ICONS   1.54030   0.00000   0.04000         0
INS  1PRCF1     1    6   0.00500    0NNNNNN
INS  1PRCF11   0.239700E+03  -0.298200E+03   0.180800E+03   0.500000E+01
INS  1PRCF12   0.000000E+00   0.000000E+00
"""
_PRM_LAB = """INS   HTYPE   PXCR
INS  1 ICONS  1.540500  1.544300    0.0000               0.500    0     0.780
INS  1PRCF1     3   19   0.00100
INS  1PRCF11       2.000000      -2.000000       5.000000   0.000000E+00
INS  1PRCF12       1.000000       2.000000       0.010000       0.020000
"""


def test_gsas_prm_cw_xray_synchrotron(tmp_path: Path) -> None:
    inst = CWInstrument.from_instrument_file(_write(tmp_path, "11bmb.prm", _PRM_11BM))
    assert inst.radiation.is_xray and inst.radiation.wavelength2 is None
    assert inst.radiation.wavelength == pytest.approx(0.413259)
    assert inst.polarization == pytest.approx(0.99)
    assert inst.zero == 0.0
    eight_ln2 = 8 * np.log(2)
    assert inst.u == pytest.approx(eight_ln2 * 1.163 / 1e4)
    assert inst.w == pytest.approx(eight_ln2 * 0.063 / 1e4)
    # GSAS の X(1/cosθ、サイズ)は本プログラムの y、Y(tanθ、歪み)は x
    assert inst.y == pytest.approx(0.173 / 100) and inst.x == 0.0
    assert inst.sh_l == pytest.approx(0.0022)


def test_gsas_prm_cw_neutron_type1(tmp_path: Path) -> None:
    inst = CWInstrument.from_instrument_file(_write(tmp_path, "BT1.inst", _INST_BT1))
    assert inst.radiation.is_neutron and inst.radiation.wavelength == pytest.approx(1.5403)
    assert inst.zero == pytest.approx(0.0004)  # 0.04 cdeg
    assert inst.polarization == 0.0
    assert inst.u == pytest.approx(8 * np.log(2) * 239.7 / 1e4)
    # 型 1(ガウス)では X = Y = 0、SH/L は既定の 0.002
    assert inst.x == 0.0 and inst.y == 0.0 and inst.sh_l == pytest.approx(0.002)


def test_gsas_prm_lab_kalpha2(tmp_path: Path) -> None:
    inst = CWInstrument.from_instrument_file(_write(tmp_path, "lab.prm", _PRM_LAB))
    rad = inst.radiation
    assert rad.wavelength == pytest.approx(1.5405) and rad.wavelength2 == pytest.approx(1.5443)
    assert rad.ratio2 == pytest.approx(0.5)
    assert inst.polarization == pytest.approx(0.78)
    assert inst.sh_l == pytest.approx(0.03)
    assert inst.x == pytest.approx(0.02) and inst.y == pytest.approx(0.01)


def test_gsas_instprm_cw_and_tof_rejection(tmp_path: Path) -> None:
    cw = _write(
        tmp_path,
        "cw.instprm",
        "#GSAS-II instrument parameter file\nType:PXC\nLam:0.413\nZero:0.01\nPolariz.:0.99\n"
        "U:1.2\nV:-0.1\nW:0.06\nX:0.0\nY:0.17\nZ:0.0\nSH/L:0.002\nAzimuth:0.0\n",
    )
    inst = CWInstrument.from_instrument_file(cw)
    assert inst.radiation.wavelength == pytest.approx(0.413) and inst.zero == pytest.approx(0.01)
    assert inst.x == pytest.approx(0.0017) and inst.y == 0.0  # GSAS-II の Y(tanθ)は x
    tof = _write(tmp_path, "tof.instprm", "Type:PNT\ndifC:22600\nsig-1:50\n")
    with pytest.raises(ValueError, match="定波長"):
        CWInstrument.from_instrument_file(tof)
