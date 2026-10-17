"""装置メーカー形式のリーダー: Rigaku .ras / .rasx、PANalytical .xrdml。

読める情報はメタデータ(metadata)に入れる:
    wavelength: λ1 [Å]、wavelength2: λ2 [Å]、ratio2: Kα2/Kα1、target: 'Cu' など
GUI はこれを見て装置設定の初期値を決める。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from kaisetsu.pattern.histogram import PowderData

_NUM_LINE = re.compile(r"^\s*[-+]?\d")


def _powder(x: list[float], y: list[float], name: str, meta: dict[str, object]) -> PowderData:
    if len(x) < 2:
        raise ValueError("数値データが読めませんでした")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    order = np.argsort(xa, kind="stable")
    with np.errstate(divide="ignore"):
        w = np.where(ya > 0, 1.0 / np.maximum(ya, 1e-300), 0.0)
    return PowderData(xa[order], ya[order], w[order], name=name, metadata=meta)


# ----------------------------------------------------------------------
# Rigaku .ras(テキスト)
# ----------------------------------------------------------------------
def is_ras_text(head: str) -> bool:
    return "*RAS_DATA_START" in head or "*RAS_HEADER_START" in head


def parse_ras_text(text: str, name: str = "") -> PowderData:
    """Rigaku SmartLab / MiniFlex の .ras 本文を解釈する(最初のスキャンのみ)。"""
    meta: dict[str, object] = {"format": "Rigaku RAS"}
    x: list[float] = []
    y: list[float] = []
    in_data = False
    done = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*"):
            if line.startswith("*RAS_INT_START"):
                in_data = True
                continue
            if line.startswith("*RAS_INT_END"):
                in_data = False
                done = True
                continue
            if done:
                break
            key, _, val = line[1:].partition(" ")
            val = val.strip().strip('"')
            if key == "MEAS_COND_XG_WAVE_LENGTH_ALPHA1":
                meta["wavelength"] = float(val)
            elif key == "MEAS_COND_XG_WAVE_LENGTH_ALPHA2":
                meta["wavelength2"] = float(val)
            elif key == "HW_XG_TARGET_NAME":
                meta["target"] = val
            elif key == "MEAS_SCAN_AXIS_X":
                meta["scan_axis"] = val
            elif key == "FILE_SAMPLE":
                meta["sample"] = val
            elif key == "MEAS_SCAN_STEP":
                meta["step"] = float(val)
            continue
        if in_data and _NUM_LINE.match(line):
            parts = line.replace(",", " ").split()
            try:
                xi, yi = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            # 3 列目は減衰器などの係数(通常 1.0)。1 以外なら掛ける。
            if len(parts) >= 3:
                try:
                    f = float(parts[2])
                    if f > 0 and abs(f - 1.0) > 1e-9:
                        yi *= f
                except ValueError:
                    pass
            x.append(xi)
            y.append(yi)
    if "wavelength" in meta and "wavelength2" in meta:
        meta["ratio2"] = 0.5
    return _powder(x, y, name, meta)


def read_ras(path: str | Path) -> PowderData:
    p = Path(path)
    text = p.read_bytes().decode("utf-8", errors="replace")
    if not is_ras_text(text[:4000]):
        text = p.read_bytes().decode("shift_jis", errors="replace")
    return parse_ras_text(text, p.stem)


# ----------------------------------------------------------------------
# Rigaku .rasx(zip: Data0/Profile0.txt + MesurementConditions0.xml)
# ----------------------------------------------------------------------
def read_rasx(path: str | Path) -> PowderData:
    p = Path(path)
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        profiles = sorted(n for n in names if re.search(r"Profile\d*\.txt$", n))
        if not profiles:
            raise ValueError(
                f"{p.name}: Profile*.txt が見つかりません(.rasx ではないかもしれません)"
            )
        text = zf.read(profiles[0]).decode("utf-8", errors="replace")
        meta: dict[str, object] = {"format": "Rigaku RASX"}
        conds = [
            n
            for n in names
            if re.search(r"MesurementConditions\d*\.xml$|MeasurementConditions\d*\.xml$", n)
        ]
        if conds:
            try:
                root = ElementTree.fromstring(zf.read(conds[0]))
                for el in root.iter():
                    tag = el.tag.split("}")[-1]
                    if tag == "WavelengthKalpha1" and el.text:
                        meta["wavelength"] = float(el.text)
                    elif tag == "WavelengthKalpha2" and el.text:
                        meta["wavelength2"] = float(el.text)
                    elif tag == "TargetName" and el.text:
                        meta["target"] = el.text.strip()
            except ElementTree.ParseError:
                pass
    x: list[float] = []
    y: list[float] = []
    for line in text.splitlines():
        if not _NUM_LINE.match(line):
            continue
        parts = line.replace(",", " ").split()
        try:
            xi, yi = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        if len(parts) >= 3:
            try:
                f = float(parts[2])
                if f > 0 and abs(f - 1.0) > 1e-9:
                    yi *= f
            except ValueError:
                pass
        x.append(xi)
        y.append(yi)
    if "wavelength" in meta and "wavelength2" in meta:
        meta["ratio2"] = 0.5
    return _powder(x, y, p.stem, meta)


# ----------------------------------------------------------------------
# PANalytical / Malvern Panalytical .xrdml(XML)
# ----------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _xrdml_from_root(root: ElementTree.Element, name: str) -> PowderData:
    """XRDML の木から最初の <scan> を読む(read_xrdml と parse_xrdml_text で共用)。

    `<usedWavelength intended="K-Alpha 1">` は Kα1 単色化(Ge(111) モノクロメータなど)の
    データなので、ファイルに kAlpha2 の波長が書かれていても Kα2 の情報は出さない
    (出すと GUI が Kα2 ありで始めてしまう。レビュー 2026-10-17 GUI #9)。
    """
    meta: dict[str, object] = {"format": "PANalytical XRDML"}
    monochromatic = False
    for el in root.iter():
        t = _local(el.tag)
        if t == "kAlpha1" and el.text:
            meta["wavelength"] = float(el.text)
        elif t == "kAlpha2" and el.text:
            meta["wavelength2"] = float(el.text)
        elif t == "ratioKAlpha2KAlpha1" and el.text:
            meta["ratio2"] = float(el.text)
        elif t == "anodeMaterial" and el.text:
            meta["target"] = el.text.strip()
        elif t == "usedWavelength":
            intended = el.attrib.get("intended", "")
            monochromatic = "1" in intended and "2" not in intended
    if monochromatic:
        meta.pop("wavelength2", None)
        meta.pop("ratio2", None)
        meta["monochromatic"] = True
    scans = [el for el in root.iter() if _local(el.tag) == "scan"]
    if not scans:
        raise ValueError(f"{name}: <scan> が見つかりません")
    scan = scans[0]
    start = end = None
    for pos in scan.iter():
        if _local(pos.tag) == "positions" and pos.attrib.get("axis", "").lower() in (
            "2theta",
            "2θ",
        ):
            for c in pos:
                if _local(c.tag) == "startPosition" and c.text:
                    start = float(c.text)
                elif _local(c.tag) == "endPosition" and c.text:
                    end = float(c.text)
            if start is not None and end is not None:
                break
    inten = None
    for c in scan.iter():
        if _local(c.tag) in ("intensities", "counts") and c.text:
            inten = np.array([float(v) for v in c.text.split()])
            break
    if inten is None or start is None or end is None:
        raise ValueError(f"{name}: 2θ の範囲か強度が見つかりません")
    x = np.linspace(start, end, len(inten))
    return _powder(list(x), list(inten), name, meta)


def read_xrdml(path: str | Path) -> PowderData:
    p = Path(path)
    root = ElementTree.parse(p).getroot()
    return _xrdml_from_root(root, p.stem)


def read_rasx_or_ras(path: str | Path) -> PowderData:
    p = Path(path)
    if zipfile.is_zipfile(p):
        return read_rasx(p)
    return read_ras(p)


def parse_xrdml_text(text: str, name: str = "") -> PowderData:
    """文字列から .xrdml を読む(テスト用)。"""
    tmp = io.BytesIO(text.encode("utf-8"))
    root = ElementTree.parse(tmp).getroot()
    return _xrdml_from_root(root, Path(name or "xrdml").stem)
