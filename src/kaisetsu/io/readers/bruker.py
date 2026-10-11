"""Bruker / Siemens DIFFRAC の .raw(バイナリ)リーダー。

対応する版:
    - ``RAW1.01``(DIFFRACplus、いわゆる version 3)。公開されている実ファイル 7 件
      (単一レンジ・複数レンジ)で読めることを確認済み。
    - ``RAW4.00``(DIFFRAC.SUITE / EVA 4 以降)、``RAW2``(DIFFRAC-AT V3)、
      ``"RAW "``(DIFFRAC-AT V1/V2)は公開されているレイアウト記述に従って実装したが、
      実ファイルでは未確認。読めなければファイルをテキスト(.xy / .txt)に書き出して使う。

レイアウトの出典:
    - Bruker, DIFFRACplus XCH V5.0 User's Manual, Appendix A(RAW1.01)
    - M. Wojdyr, xylib ``bruker_raw.cpp``(各版の解釈)
    - GSAS-II ``G2pwd_BrukerRAW.py``(RAW4.00 のレンジヘッダ)

読めた情報は metadata に入れる(vendors.py と同じキー):
    wavelength: λ1 [Å]、wavelength2: λ2 [Å]、ratio2: Kα2/Kα1、target: 'Cu' など、
    n_scans: ファイル内のレンジ(スキャン)数、scan: 読んだレンジ番号(1 始まり)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kaisetsu.pattern.histogram import PowderData


def is_bruker_raw(head: bytes) -> bool:
    """先頭バイト列が Bruker .raw のマジックで始まるか。"""
    return head.startswith((b"RAW1.01", b"RAW4.00")) or head[:4] in (b"RAW2", b"RAW ")


@dataclass
class _Scan:
    start: float
    step: float
    y: np.ndarray
    meta: dict[str, object] = field(default_factory=dict)


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("latin-1").strip()


def _f32(b: bytes, off: int) -> float:
    return float(struct.unpack_from("<f", b, off)[0])


def _f64(b: bytes, off: int) -> float:
    return float(struct.unpack_from("<d", b, off)[0])


def _i32(b: bytes, off: int) -> int:
    return int(struct.unpack_from("<i", b, off)[0])


def _u32(b: bytes, off: int) -> int:
    return int(struct.unpack_from("<I", b, off)[0])


def _u16(b: bytes, off: int) -> int:
    return int(struct.unpack_from("<H", b, off)[0])


def _target(anode: str) -> str:
    """'CU' → 'Cu'。"""
    a = anode.strip()
    return a[:1].upper() + a[1:].lower() if a else ""


def _counts(b: bytes, offset: int, n: int, dtype: str, name: str) -> np.ndarray:
    size = np.dtype(dtype).itemsize
    if n <= 0 or offset < 0 or offset + size * n > len(b):
        raise ValueError(
            f"{name}: データ部がファイルの大きさと合いません(壊れているかもしれません)"
        )
    return np.frombuffer(b, dtype=dtype, count=n, offset=offset).astype(float)


def _wavelength_meta(meta: dict[str, object], ka1: float, ka2: float, ratio: float) -> None:
    if 0.1 < ka1 < 10.0:
        meta["wavelength"] = ka1
    if 0.1 < ka2 < 10.0:
        meta["wavelength2"] = ka2
    if 0.0 < ratio <= 1.0:
        meta["ratio2"] = ratio


# ----------------------------------------------------------------------
# RAW1.01(version 3)
# ----------------------------------------------------------------------
def _read_v3(b: bytes, name: str) -> tuple[dict[str, object], list[_Scan]]:
    meta: dict[str, object] = {"format": "Bruker RAW", "version": "RAW1.01"}
    n_ranges = _i32(b, 12)
    meta["date"] = _cstr(b[16:26])
    meta["time"] = _cstr(b[26:36])
    sample = _cstr(b[326:386])
    if sample:
        meta["sample"] = sample
    comment = _cstr(b[386:546])
    if comment:
        meta["comment"] = comment
    radius = _f32(b, 564)
    if radius > 0:
        meta["goniometer_radius_mm"] = radius
    anode = _target(_cstr(b[608:612]))
    if anode:
        meta["target"] = anode
    _ka_mean, ka1, ka2, _kb, ratio = struct.unpack_from("<5d", b, 616)
    _wavelength_meta(meta, ka1, ka2, ratio)

    scans: list[_Scan] = []
    pos = 712
    for _ in range(max(n_ranges, 1)):
        if pos + 304 > len(b):
            break
        header_len = _i32(b, pos)
        n_steps = _i32(b, pos + 4)
        if header_len < 304 or n_steps <= 0:
            break
        start = _f64(b, pos + 16)
        step = _f64(b, pos + 176)
        supplement = max(_i32(b, pos + 256), 0)
        smeta: dict[str, object] = {
            "theta_start": _f64(b, pos + 8),
            "time_per_step_s": _f32(b, pos + 192),
        }
        used = _f64(b, pos + 240)
        if 0.1 < used < 10.0:
            smeta["used_wavelength"] = used
        data_start = pos + header_len + supplement
        y = _counts(b, data_start, n_steps, "<f4", name)
        scans.append(_Scan(start, step, y, smeta))
        pos = data_start + 4 * n_steps
    return meta, scans


# ----------------------------------------------------------------------
# RAW4.00(version 4)
# ----------------------------------------------------------------------
def _read_v4(b: bytes, name: str) -> tuple[dict[str, object], list[_Scan]]:
    meta: dict[str, object] = {"format": "Bruker RAW", "version": "RAW4.00"}
    meta["date"] = _cstr(b[12:24])
    meta["time"] = _cstr(b[24:34])
    pos = 61
    info: dict[str, str] = {}
    # ファイル全体のヘッダ: (種別, 長さ) のセグメント列。種別 0 / 160 でレンジが始まる
    while pos + 8 <= len(b):
        seg_type = _u32(b, pos)
        if seg_type in (0, 160):
            break
        seg_len = max(_u32(b, pos + 4), 8)
        if seg_type == 30 and seg_len >= 120:  # X 線源
            _ka_mean, ka1, ka2, _kb, ratio = struct.unpack_from("<5d", b, pos + 72)
            _wavelength_meta(meta, ka1, ka2, ratio)
            anode = _target(_cstr(b[pos + 116 : pos + 120]))
            if anode:
                meta["target"] = anode
        elif seg_type == 10 and seg_len >= 36:  # 名前付き情報
            key = _cstr(b[pos + 12 : pos + 36])
            val = _cstr(b[pos + 36 : pos + seg_len])
            if key:
                info[key] = val
        pos += seg_len
    if info:
        meta["info"] = info
        for key in ("SampleName", "Sample Name", "SAMPLE_NAME", "SampleID"):
            if info.get(key):
                meta["sample"] = info[key]
                break

    scans: list[_Scan] = []
    while pos + 160 <= len(b) and _u32(b, pos) in (0, 160):
        scan_type = _cstr(b[pos + 32 : pos + 56])
        start = _f64(b, pos + 72)
        step = _f64(b, pos + 80)
        n_steps = _u32(b, pos + 88)
        smeta: dict[str, object] = {
            "scan_type": scan_type,
            "time_per_step_s": _f32(b, pos + 92),
            "generator_kv": _f32(b, pos + 100),
            "generator_ma": _f32(b, pos + 104),
        }
        used = _f64(b, pos + 112)
        if 0.1 < used < 10.0:
            smeta["used_wavelength"] = used
        datum_size = _u32(b, pos + 136)
        header_size = _u32(b, pos + 140)
        # レンジ内の補助ヘッダ(駆動軸の開始値など)
        p2 = pos + 160
        end_hdr = p2 + header_size
        drives: dict[str, float] = {}
        while p2 + 8 <= end_hdr:
            st = _u32(b, p2)
            ln = max(_u32(b, p2 + 4), 8)
            if st == 50 and ln >= 64:
                drives[_cstr(b[p2 + 12 : p2 + 36])] = _f64(b, p2 + 56)
            p2 += ln
        if drives:
            smeta["drives_start"] = drives
        if "Coupled" not in scan_type and "Detector" not in scan_type and "2Theta" not in scan_type:
            smeta["warning"] = f"スキャン種別 {scan_type!r} は 2θ 走査ではないかもしれません"
        dtype = "<f8" if datum_size == 8 else "<f4"
        data_start = pos + 160 + header_size
        y = _counts(b, data_start, n_steps, dtype, name)
        scans.append(_Scan(start, step, y, smeta))
        pos = data_start + (8 if datum_size == 8 else 4) * n_steps
    return meta, scans


# ----------------------------------------------------------------------
# RAW2(version 2)
# ----------------------------------------------------------------------
def _read_v2(b: bytes, name: str) -> tuple[dict[str, object], list[_Scan]]:
    meta: dict[str, object] = {"format": "Bruker RAW", "version": "RAW2"}
    n_ranges = _u16(b, 4)
    meta["date"] = _cstr(b[168:188])
    anode = _target(_cstr(b[188:190]))
    if anode:
        meta["target"] = anode
    _wavelength_meta(meta, _f32(b, 190), _f32(b, 194), _f32(b, 198))
    scans: list[_Scan] = []
    pos = 256
    for _ in range(max(n_ranges, 1)):
        if pos + 48 > len(b):
            break
        header_len = _u16(b, pos)
        n_steps = _u16(b, pos + 2)
        if header_len < 48 or n_steps == 0:
            break
        smeta: dict[str, object] = {"time_per_step_s": _f32(b, pos + 8)}
        step = _f32(b, pos + 12)
        start = _f32(b, pos + 16)
        data_start = pos + header_len
        y = _counts(b, data_start, n_steps, "<f4", name)
        scans.append(_Scan(start, step, y, smeta))
        pos = data_start + 4 * n_steps
    return meta, scans


# ----------------------------------------------------------------------
# "RAW "(version 1)
# ----------------------------------------------------------------------
def _read_v1(b: bytes, name: str) -> tuple[dict[str, object], list[_Scan]]:
    meta: dict[str, object] = {"format": "Bruker RAW", "version": "RAW"}
    magic = struct.unpack("<I", b"RAW ")[0]
    scans: list[_Scan] = []
    pos = 4
    while pos + 152 <= len(b):
        n_steps = _u32(b, pos)
        if scans and n_steps == magic:  # 古いファイルは 2 レンジ目以降にも "RAW " を書くことがある
            pos += 4
            n_steps = _u32(b, pos)
        if n_steps == 0:
            break
        smeta: dict[str, object] = {"time_per_step_s": _f32(b, pos + 4)}
        step = _f32(b, pos + 8)
        start = _f32(b, pos + 20)
        sample = _cstr(b[pos + 36 : pos + 68])
        if sample:
            smeta["sample"] = sample
        _wavelength_meta(meta, _f32(b, pos + 68), _f32(b, pos + 72), 0.0)
        following = _u32(b, pos + 148)
        data_start = pos + 152
        y = _counts(b, data_start, n_steps, "<f4", name)
        scans.append(_Scan(start, step, y, smeta))
        pos = data_start + 4 * n_steps
        if following == 0:
            break
    return meta, scans


# ----------------------------------------------------------------------
def read_bruker_raw(path: str | Path, scan: int = 1) -> PowderData:
    """Bruker .raw を読む。複数レンジ(スキャン)があれば scan 番号(1 始まり)で選ぶ。"""
    p = Path(path)
    b = p.read_bytes()
    head = b[:7]
    if head == b"RAW1.01":
        meta, scans = _read_v3(b, p.name)
    elif head == b"RAW4.00":
        meta, scans = _read_v4(b, p.name)
    elif head[:4] == b"RAW2":
        meta, scans = _read_v2(b, p.name)
    elif head[:4] == b"RAW ":
        meta, scans = _read_v1(b, p.name)
    else:
        raise ValueError(f"{p.name}: Bruker .raw ではありません(先頭 {head!r})")
    if not scans:
        raise ValueError(f"{p.name}: 読めるスキャンがありません")
    if not 1 <= scan <= len(scans):
        raise ValueError(f"{p.name}: スキャン {scan} はありません(スキャン数 {len(scans)})")
    s = scans[scan - 1]
    if not (0.0 < abs(s.step) < 10.0):
        raise ValueError(f"{p.name}: 刻み幅 {s.step} が不正です")
    x = s.start + s.step * np.arange(len(s.y), dtype=float)
    y = s.y
    with np.errstate(divide="ignore"):
        w = np.where(y > 0, 1.0 / np.maximum(y, 1e-300), 0.0)
    meta.update(s.meta)
    meta["n_scans"] = len(scans)
    meta["scan"] = scan
    meta["path"] = str(p)
    name = p.stem if len(scans) == 1 else f"{p.stem} scan {scan}"
    if s.step < 0:
        order = np.argsort(x, kind="stable")
        x, y, w = x[order], y[order], w[order]
    return PowderData(x, y, w, name=name, metadata=meta)
