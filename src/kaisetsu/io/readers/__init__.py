"""回折データのリーダー。拡張子と中身から形式を判定する。"""

from __future__ import annotations

import zipfile
from pathlib import Path

from kaisetsu.io.readers.bruker import is_bruker_raw, read_bruker_raw
from kaisetsu.io.readers.fullprof import detect_fullprof, read_fullprof
from kaisetsu.io.readers.gsas import gsas_bank_count, is_gsas_file, read_gsas
from kaisetsu.io.readers.opxrd import is_opxrd, read_opxrd
from kaisetsu.io.readers.pdcif import read_pdcif
from kaisetsu.io.readers.rietan import is_rietan_block, read_rietan_block
from kaisetsu.io.readers.vendors import is_ras_text, read_ras, read_rasx, read_xrdml
from kaisetsu.io.readers.xy import binary_message, decode_text, looks_binary, read_xy
from kaisetsu.pattern.histogram import PowderData

SUPPORTED_EXTENSIONS = (
    ".xy", ".xye", ".chi", ".dat", ".txt", ".csv", ".asc",
    ".gsa", ".gsas", ".fxye", ".raw", ".xra", ".cwn",
    ".ras", ".rasx", ".xrdml", ".int", ".histogramigor",
)  # fmt: skip

#: これより点数が少ないデータは、形式を読み違えている可能性が高い(GUI は読み込みを拒む)
MIN_POINTS = 50


def bank_count(path: str | Path) -> int:
    """複数バンク(GSAS 形式)なら BANK の数、それ以外は 1。"""
    p = Path(path)
    if p.exists() and is_gsas_file(p):
        return max(gsas_bank_count(p), 1)
    return 1


def too_few_points_message(data: PowderData, name: str) -> str | None:
    """点数が極端に少ない(MIN_POINTS 未満)ときの日本語の説明。問題なければ None。"""
    if len(data) >= MIN_POINTS:
        return None
    return (
        f"{name}: 読み込めた点数が {len(data)} 点しかありません。ファイルの形式を正しく読めて"
        "いない可能性があります(ヘッダの書き方、小数点のカンマ、対応していない形式など)。"
        "2θ と強度を空白かカンマで区切ったテキストで保存し直してください"
    )


def read_powder_data(path: str | Path, **kwargs: object) -> PowderData:
    """粉末回折データを読む。

    対応形式:
        - ASCII 2〜3 列(.xy/.xye/.dat/.txt/.csv: 2θ, 強度[, σ])。先頭のヘッダ行は読み飛ばす。
          Fit2D / Dioptas / pyFAI が 2 次元画像を積分して書く .chi / .xy(ヘッダ付き)も同じ。
          文字コードは UTF-8(BOM 付き可)と UTF-16(BOM 付き)
        - GSAS 標準形式(.gsa/.gsas/.fxye/.raw/.xra/.cwn 等、BANK 行を持つもの)
        - Bruker / Siemens DIFFRAC .raw(バイナリ。RAW1.01 は実ファイルで確認済み)
        - Rigaku .ras(テキスト)、.rasx(zip)
        - PANalytical / Malvern Panalytical .xrdml
        - RIETAN-FP の .int(GENERAL / IGOR / GSAS 形式と「点数 開始角 刻み」形式)
        - FullProf の自由形式(開始角 刻み 終了角 + 強度)、D1A/D2B の検出器数と強度の組
        - Z-Rietveld の .histogramIgor(IGOR テキスト。TOF は 4 列目の計数を無視)

    対応していないバイナリ(古い装置の .raw、Excel の .xlsx など)は 2 点のゴミとして読まずに
    エラーにする。kwargs は形式ごとの選択肢(GSAS: bank=、Bruker: scan=)に渡す。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {p}")
    suffix = p.suffix.lower()
    if suffix == ".xrdml":
        return read_xrdml(p)
    if suffix == ".rasx" or (suffix == ".ras" and zipfile.is_zipfile(p)):
        return read_rasx(p)
    raw = p.read_bytes()
    raw_head = raw[:4000]
    if is_bruker_raw(raw_head):
        return read_bruker_raw(p, **kwargs)  # type: ignore[arg-type]
    if zipfile.is_zipfile(p):
        raise ValueError(
            f"{p.name}: zip 形式のファイルです(Excel の .xlsx をそのまま選んでいませんか?)。"
            "テキスト(CSV)で保存し直してください"
        )
    head = decode_text(raw_head)
    if suffix == ".ras" or is_ras_text(head):
        return read_ras(p)
    if head.lstrip().startswith("<?xml") and "xrdMeasurement" in head:
        return read_xrdml(p)
    if suffix == ".cif":
        return read_pdcif(p)  # pdCIF(COD や Acta E に付く GSAS2CIF 出力の粉末データ)
    if is_gsas_file(p):
        return read_gsas(p, **kwargs)  # type: ignore[arg-type]
    if looks_binary(raw_head):
        raise ValueError(binary_message(p.name))
    if is_rietan_block(decode_text(raw), complete=True):
        return read_rietan_block(p)
    if is_opxrd(head):
        return read_opxrd(p).data
    if detect_fullprof(head) is not None:
        return read_fullprof(p)
    return read_xy(p)
