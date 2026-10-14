"""回折データのリーダー。拡張子と中身から形式を判定する。"""

from __future__ import annotations

import zipfile
from pathlib import Path

from kaisetsu.io.readers.bruker import is_bruker_raw, read_bruker_raw
from kaisetsu.io.readers.fullprof import detect_fullprof, read_fullprof
from kaisetsu.io.readers.gsas import gsas_bank_count, is_gsas_file, read_gsas
from kaisetsu.io.readers.opxrd import is_opxrd, read_opxrd
from kaisetsu.io.readers.rietan import is_rietan_block, read_rietan_block
from kaisetsu.io.readers.vendors import is_ras_text, read_ras, read_rasx, read_xrdml
from kaisetsu.io.readers.xy import read_xy
from kaisetsu.pattern.histogram import PowderData

SUPPORTED_EXTENSIONS = (
    ".xy", ".xye", ".chi", ".dat", ".txt", ".csv", ".asc",
    ".gsa", ".gsas", ".fxye", ".raw", ".xra", ".cwn",
    ".ras", ".rasx", ".xrdml", ".int", ".histogramigor",
)  # fmt: skip


def bank_count(path: str | Path) -> int:
    """複数バンク(GSAS 形式)なら BANK の数、それ以外は 1。"""
    p = Path(path)
    if p.exists() and is_gsas_file(p):
        return max(gsas_bank_count(p), 1)
    return 1


def read_powder_data(path: str | Path, **kwargs: object) -> PowderData:
    """粉末回折データを読む。

    対応形式:
        - ASCII 2〜3 列(.xy/.xye/.dat/.txt/.csv: 2θ, 強度[, σ])。先頭のヘッダ行は読み飛ばす。
          Fit2D / Dioptas / pyFAI が 2 次元画像を積分して書く .chi / .xy(ヘッダ付き)も同じ
        - GSAS 標準形式(.gsa/.gsas/.fxye/.raw/.xra/.cwn 等、BANK 行を持つもの)
        - Bruker / Siemens DIFFRAC .raw(バイナリ。RAW1.01 は実ファイルで確認済み)
        - Rigaku .ras(テキスト)、.rasx(zip)
        - PANalytical / Malvern Panalytical .xrdml
        - RIETAN-FP の .int(GENERAL / IGOR / GSAS 形式と「点数 開始角 刻み」形式)
        - FullProf の自由形式(開始角 刻み 終了角 + 強度)、D1A/D2B の検出器数と強度の組
        - Z-Rietveld の .histogramIgor(IGOR テキスト。TOF は 4 列目の計数を無視)

    kwargs は形式ごとの選択肢(GSAS: bank=、Bruker: scan=)に渡す。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {p}")
    suffix = p.suffix.lower()
    if suffix == ".xrdml":
        return read_xrdml(p)
    if suffix == ".rasx" or (suffix == ".ras" and zipfile.is_zipfile(p)):
        return read_rasx(p)
    raw_head = p.read_bytes()[:4000]
    if is_bruker_raw(raw_head):
        return read_bruker_raw(p, **kwargs)  # type: ignore[arg-type]
    head = raw_head.decode("utf-8", errors="replace")
    if suffix == ".ras" or is_ras_text(head):
        return read_ras(p)
    if head.lstrip().startswith("<?xml") and "xrdMeasurement" in head:
        return read_xrdml(p)
    if is_gsas_file(p):
        return read_gsas(p, **kwargs)  # type: ignore[arg-type]
    if is_rietan_block(head):
        return read_rietan_block(p)
    if is_opxrd(head):
        return read_opxrd(p).data
    if detect_fullprof(head) is not None:
        return read_fullprof(p)
    return read_xy(p)
