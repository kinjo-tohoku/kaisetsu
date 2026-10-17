"""ASCII 2〜3 列(2θ, 強度[, σ])のリーダー。

'#', '!', '/', ';' で始まる行とヘッダ文字列は読み飛ばす。区切りは空白・カンマ・タブ。
文字コードは UTF-8(BOM 付きも可)と UTF-16(BOM で判定。Excel の「Unicode テキスト」)。
cp932 などの日本語ヘッダは読めない文字として読み飛ばす(数値の行には影響しない)。
"""

from __future__ import annotations

import codecs
import re
from pathlib import Path

import numpy as np

from kaisetsu.pattern.histogram import PowderData

_SPLIT = re.compile(r"[\s,;]+")
_DECIMAL_COMMA = re.compile(r"\d,\d")
_DECIMAL_POINT = re.compile(r"\d\.\d")
_UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


def decode_text(raw: bytes) -> str:
    """バイト列を文字列にする。UTF-16 は BOM で判定し、UTF-8 の BOM は取り除く。

    BOM 付き UTF-8 を utf-8 で読むと先頭行が "\\ufeff10.000 …" になって最初の 1 点が落ちる
    (レビュー 2026-10-17 GUI #17)。
    """
    if raw.startswith(_UTF16_BOMS):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def looks_binary(raw: bytes) -> bool:
    """テキストではなくバイナリらしいか(UTF-16 以外で NUL バイトを含む)。"""
    if raw.startswith(_UTF16_BOMS):
        return False
    return b"\x00" in raw


def binary_message(name: str) -> str:
    return (
        f"{name}: テキストではなくバイナリのファイルのようです(対応していない装置の形式か、"
        "Excel などのファイル)。装置のソフトでテキスト(2θ, 強度)に書き出してください"
    )


def parse_xy_text(text: str, name: str, path: str = "") -> PowderData:
    """文字列から 2〜3 列(2θ, 強度[, σ])を読む。"""
    xs: list[float] = []
    ys: list[float] = []
    es: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#!/;'":
            continue
        parts = [t for t in _SPLIT.split(line) if t]
        try:
            vals = [float(t) for t in parts[:3]]
        except ValueError:
            continue  # ヘッダ行
        if len(vals) < 2:
            continue
        xs.append(vals[0])
        ys.append(vals[1])
        es.append(vals[2] if len(vals) >= 3 else float("nan"))
    x = np.array(xs)
    n_unique = len(np.unique(x)) if x.size else 0
    if _DECIMAL_COMMA.search(text) and not _DECIMAL_POINT.search(text):
        # 「10,250 1234」や「10,250;1234,0」はカンマを区切りと解釈してしまい、x が整数だけ
        # (同じ値が何十回も)になる。小数点がピリオドの普通のデータなら \d\.\d がある
        if len(xs) < 2 or n_unique < 0.5 * len(xs):
            raise ValueError(
                f"{name}: 小数点がカンマ(10,250 のような書き方)になっているようです。"
                "小数点をピリオド(10.250)にして保存し直してください"
            )
    if len(xs) < 2:
        raise ValueError(f"{name}: 数値データが読めませんでした")
    if n_unique < 0.2 * len(xs):
        # 走査を 2〜3 回継いだファイルは平均するが、同じ 2θ が平均 5 回以上なら列の読み違い
        raise ValueError(
            f"{name}: 同じ 2θ の点が多すぎます({len(xs)} 点中 {n_unique} 種類)。"
            "列の区切りや小数点の書き方が想定と違うかもしれません"
            "(2θ, 強度[, σ] を空白かカンマで区切ったテキストにしてください)"
        )
    y = np.array(ys)
    e = np.array(es)
    if np.all(np.isfinite(e)) and np.any(e > 0):
        with np.errstate(divide="ignore"):
            w = np.where(e > 0, 1.0 / e**2, 0.0)
    else:
        with np.errstate(divide="ignore"):
            w = np.where(y > 0, 1.0 / np.maximum(y, 1e-300), 0.0)
    order = np.argsort(x, kind="stable")
    x, y, w = x[order], y[order], w[order]
    if np.any(np.diff(x) == 0.0):
        # 同じ 2θ が重複している(複数の走査を継いだファイルなど)→ 平均して 1 点にする
        ux, inv = np.unique(x, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        y = np.bincount(inv, weights=y) / cnt
        w = np.bincount(inv, weights=w)  # 分散は 1/Σw(重みの和)
        x = ux
    return PowderData(x, y, w, name=name, metadata={"format": "xy", "path": path})


def read_xy(path: str | Path) -> PowderData:
    p = Path(path)
    raw = p.read_bytes()
    if looks_binary(raw[:4000]):
        raise ValueError(binary_message(p.name))
    return parse_xy_text(decode_text(raw), p.stem, str(p))
