"""Bruker .raw リーダーのテスト。レイアウトどおりの合成ファイルと公開実データで確認する。"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.io import read_powder_data
from kaisetsu.io.readers.bruker import is_bruker_raw, read_bruker_raw

ROOT = Path(__file__).resolve().parents[2]


def _put(buf: bytearray, off: int, fmt: str, *vals: object) -> None:
    struct.pack_into(fmt, buf, off, *vals)


def _put_str(buf: bytearray, off: int, s: str) -> None:
    b = s.encode("latin-1")
    buf[off : off + len(b)] = b


def build_raw101(ranges: list[tuple[float, float, list[float], int]], anode: str = "Cu") -> bytes:
    """RAW1.01 を組み立てる。ranges = [(2θ開始, 刻み, 強度列, 補助ヘッダ長)]。"""
    head = bytearray(712)
    _put_str(head, 0, "RAW1.01")
    _put(head, 8, "<i", 1)
    _put(head, 12, "<i", len(ranges))
    _put_str(head, 16, "01/22/18")
    _put_str(head, 26, "16:34:23")
    _put_str(head, 326, "test sample")
    _put(head, 564, "<f", 250.0)
    _put_str(head, 608, anode)
    _put(head, 616, "<5d", 1.54184, 1.5406, 1.54439, 1.39222, 0.5)
    out = bytes(head)
    for start, step, ys, supp in ranges:
        rh = bytearray(304)
        _put(rh, 0, "<i", 304)
        _put(rh, 4, "<i", len(ys))
        _put(rh, 8, "<d", start / 2)
        _put(rh, 16, "<d", start)
        _put(rh, 176, "<d", step)
        _put(rh, 192, "<f", 1.5)
        _put(rh, 240, "<d", 1.5406)
        _put(rh, 256, "<i", supp)
        out += bytes(rh) + bytes(supp) + struct.pack(f"<{len(ys)}f", *ys)
    return out


def build_raw4(start: float, step: float, ys: list[float]) -> bytes:
    """RAW4.00 を組み立てる(ヘッダセグメント 10 と 30、レンジ 1 つ)。"""
    out = bytearray(61)
    _put_str(out, 0, "RAW4.00")
    _put_str(out, 12, "05/06/2024")
    _put_str(out, 24, "10:20:30")
    # 種別 10: 名前付き情報
    seg = bytearray(36 + 16)
    _put(seg, 0, "<I", 10)
    _put(seg, 4, "<I", len(seg))
    _put_str(seg, 12, "SampleName")
    _put_str(seg, 36, "raw4 sample")
    out += seg
    # 種別 30: X 線源
    seg = bytearray(124)
    _put(seg, 0, "<I", 30)
    _put(seg, 4, "<I", len(seg))
    _put(seg, 72, "<5d", 1.54184, 1.5406, 1.54439, 1.39222, 0.5)
    _put_str(seg, 116, "Co")
    out += seg
    # レンジヘッダ 160 バイト + 補助ヘッダ(種別 50 の駆動軸 1 つ、64 バイト)
    sub = bytearray(64)
    _put(sub, 0, "<I", 50)
    _put(sub, 4, "<I", 64)
    _put_str(sub, 12, "Theta")
    _put(sub, 56, "<d", start / 2)
    rh = bytearray(160)
    _put(rh, 0, "<I", 160)
    _put_str(rh, 32, "Locked Coupled")
    _put(rh, 72, "<d", start)
    _put(rh, 80, "<d", step)
    _put(rh, 88, "<I", len(ys))
    _put(rh, 92, "<f", 0.5)
    _put(rh, 100, "<f", 40.0)
    _put(rh, 104, "<f", 40.0)
    _put(rh, 112, "<d", 1.5406)
    _put(rh, 136, "<I", 4)
    _put(rh, 140, "<I", len(sub))
    out += rh + sub + struct.pack(f"<{len(ys)}f", *ys)
    return bytes(out)


def build_raw2(start: float, step: float, ys: list[float]) -> bytes:
    head = bytearray(256)
    _put_str(head, 0, "RAW2")
    _put(head, 4, "<H", 1)
    _put_str(head, 168, "22-MAR-95 10:00")
    _put_str(head, 188, "Cu")
    _put(head, 190, "<3f", 1.5406, 1.54439, 0.5)
    rh = bytearray(48)
    _put(rh, 0, "<H", 48)
    _put(rh, 2, "<H", len(ys))
    _put(rh, 8, "<f", 1.0)
    _put(rh, 12, "<f", step)
    _put(rh, 16, "<f", start)
    return bytes(head) + bytes(rh) + struct.pack(f"<{len(ys)}f", *ys)


def build_raw1(start: float, step: float, ys: list[float]) -> bytes:
    out = bytearray(4 + 152)
    _put_str(out, 0, "RAW ")
    _put(out, 4, "<I", len(ys))
    _put(out, 8, "<f", 1.0)
    _put(out, 12, "<f", step)
    _put(out, 24, "<f", start)
    _put_str(out, 40, "old sample")
    _put(out, 72, "<2f", 1.5406, 1.54439)
    _put(out, 152, "<I", 0)
    return bytes(out) + struct.pack(f"<{len(ys)}f", *ys)


def test_is_bruker_raw() -> None:
    assert is_bruker_raw(b"RAW1.01\0\0")
    assert is_bruker_raw(b"RAW4.00\0")
    assert is_bruker_raw(b"RAW2xxxx")
    assert is_bruker_raw(b"RAW \0\0\0\0")
    assert not is_bruker_raw(b"BANK 1 ...")
    assert not is_bruker_raw(b"RAW5.00")


def test_raw101_single_range(tmp_path: Path) -> None:
    ys = [100.0, 120.0, 90.0, 300.0, 110.0]
    p = tmp_path / "a.raw"
    p.write_bytes(build_raw101([(10.0, 0.02, ys, 40)]))
    d = read_powder_data(p)
    assert len(d) == 5
    assert np.allclose(d.x, [10.0, 10.02, 10.04, 10.06, 10.08])
    assert np.allclose(d.y, ys)
    assert d.metadata["target"] == "Cu"
    assert d.metadata["wavelength"] == pytest.approx(1.5406)
    assert d.metadata["wavelength2"] == pytest.approx(1.54439)
    assert d.metadata["ratio2"] == pytest.approx(0.5)
    assert d.metadata["sample"] == "test sample"
    assert d.metadata["n_scans"] == 1
    assert d.name == "a"


def test_raw101_multi_range(tmp_path: Path) -> None:
    p = tmp_path / "b.raw"
    p.write_bytes(build_raw101([(10.0, 0.1, [1.0, 2.0, 3.0], 0), (30.0, 0.2, [4.0, 5.0], 0)], "CU"))
    d1 = read_bruker_raw(p, scan=1)
    d2 = read_bruker_raw(p, scan=2)
    assert np.allclose(d1.x, [10.0, 10.1, 10.2]) and np.allclose(d1.y, [1, 2, 3])
    assert np.allclose(d2.x, [30.0, 30.2]) and np.allclose(d2.y, [4, 5])
    assert d1.metadata["target"] == "Cu"
    assert d2.metadata["n_scans"] == 2 and d2.name == "b scan 2"
    with pytest.raises(ValueError):
        read_bruker_raw(p, scan=3)
    # kwargs 経由でも選べる
    assert len(read_powder_data(p, scan=2)) == 2


def test_raw4(tmp_path: Path) -> None:
    ys = [10.0, 20.0, 30.0, 40.0]
    p = tmp_path / "c.raw"
    p.write_bytes(build_raw4(20.0, 0.01, ys))
    d = read_powder_data(p)
    assert np.allclose(d.x, [20.0, 20.01, 20.02, 20.03])
    assert np.allclose(d.y, ys)
    assert d.metadata["target"] == "Co"
    assert d.metadata["wavelength"] == pytest.approx(1.5406)
    assert d.metadata["sample"] == "raw4 sample"
    assert d.metadata["scan_type"] == "Locked Coupled"
    assert d.metadata["drives_start"]["Theta"] == pytest.approx(10.0)


def test_raw2_and_raw1(tmp_path: Path) -> None:
    p2 = tmp_path / "d.raw"
    p2.write_bytes(build_raw2(5.0, 0.05, [7.0, 8.0, 9.0]))
    d = read_powder_data(p2)
    assert np.allclose(d.x, [5.0, 5.05, 5.1]) and np.allclose(d.y, [7, 8, 9])
    assert d.metadata["target"] == "Cu" and d.metadata["wavelength"] == pytest.approx(1.5406)
    p1 = tmp_path / "e.raw"
    p1.write_bytes(build_raw1(3.0, 0.5, [1.0, 1.0]))
    d = read_powder_data(p1)
    assert np.allclose(d.x, [3.0, 3.5]) and d.metadata["sample"] == "old sample"


def test_truncated_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "f.raw"
    p.write_bytes(build_raw101([(10.0, 0.02, [1.0] * 10, 0)])[:-8])
    with pytest.raises(ValueError):
        read_powder_data(p)


@pytest.mark.skipif(
    not (ROOT / "data/public/lab6/LaB6_Jan2018.raw").exists(), reason="公開データが無い"
)
def test_real_lab6_file() -> None:
    d = read_powder_data(ROOT / "data/public/lab6/LaB6_Jan2018.raw")
    assert len(d) == 3040
    assert d.x[0] == pytest.approx(10.0)
    assert d.x[1] - d.x[0] == pytest.approx(0.0197448, abs=1e-6)
    assert d.y[0] == pytest.approx(7393.0)
    assert d.metadata["target"] == "Cu"
    assert d.metadata["wavelength"] == pytest.approx(1.5406)


@pytest.mark.skipif(
    not (ROOT / "data/public/kaolinite/Keokuk kaolinite.RAW").exists(), reason="公開データが無い"
)
def test_real_two_range_file() -> None:
    p = ROOT / "data/public/kaolinite/Keokuk kaolinite.RAW"
    d1 = read_powder_data(p)
    d2 = read_powder_data(p, scan=2)
    assert d1.metadata["n_scans"] == 2
    assert len(d1) == 4001 and d1.x[0] == pytest.approx(10.0)
    assert len(d2) == 3501 and d2.x[0] == pytest.approx(80.0)
    assert d1.metadata["sample"].startswith("Kaolinite")
