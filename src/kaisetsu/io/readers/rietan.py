"""RIETAN-FP の強度データ(*.int)のうち、「点数 開始角 刻み」形式。

RIETAN-FP が受け付ける強度データのうち、
- GENERAL 形式(1 行目 "GENERAL"、2 行目 点数、以降 "2θ 強度 [σ]")と IGOR テキスト形式は `read_xy`、
- GSAS 形式(BANK 行)は `read_gsas`
で読める。ここでは残りの、RIETAN-94 以来の標準形式を扱う:

    * コメント(任意。'*' か '#' で始まる行が何行でも)
    点数 開始角 刻み
    強度 強度 強度 …(1 行に何個でも。点数ぶん続く)

2θ は 開始角 + 刻み × i で作る。σ は無いので重みは 1/強度(計数統計)。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from kaisetsu.io.readers.vendors import _powder
from kaisetsu.io.readers.xy import decode_text
from kaisetsu.pattern.histogram import PowderData

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _is_num(tok: str) -> bool:
    return bool(_NUM.match(tok))


def _body_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("*", "#")):
            continue
        out.append(line)
    return out


def is_rietan_block(head: str, *, complete: bool = False) -> bool:
    """「点数 開始角 刻み + 強度の羅列」形式かを判定する。

    ヘッダの無い 3 列の .xye(「10.000 1234 35.1」で始まるもの)も 1 行目だけ見れば
    「点数 10、開始角 1234、刻み 35.1」に見えるので(レビュー 2026-10-17 GUI #5)、
    開始角が 0〜180°、2 行目以降が強度だけ(全部数値で列数が揃う)、数値の総数が点数を超えない
    ことも確かめる。`complete=True` なら head はファイル全文とみなし、総数が点数と合うことも要る。
    曖昧なときは False(xy として読む)。
    """
    lines = _body_lines(head)
    if len(lines) < 2:
        return False
    first = lines[0].replace(",", " ").split()
    if len(first) != 3 or not all(_is_num(t) for t in first):
        return False
    try:
        n = int(float(first[0]))
        start = float(first[1])
        step = float(first[2])
    except ValueError:
        return False
    if n < 10 or step <= 0.0 or float(first[0]) != n:
        return False
    # FullProf の「開始角 刻み 終了角」(3 番目が終了角で大きい)と区別する: RIETAN は 3 番目が刻み
    if step >= start and start > 0:
        return False
    if not 0.0 <= start <= 180.0:
        return False
    second = lines[1].replace(",", " ").split()
    # 2 行目が「2θ 強度」の 2 列なら GENERAL 類似(点数の行が無い)なので対象外。
    if len(second) == 2 and abs(float(second[0]) - start) <= 1e-9:
        return False
    counts: list[int] = []
    total = 0
    for line in lines[1:]:
        toks = line.replace(",", " ").split()
        if not toks or not all(_is_num(t) for t in toks):
            return False
        counts.append(len(toks))
        total += len(toks)
    # 列数は最後の行以外で揃い、最後の行はそれ以下
    if any(c != counts[0] for c in counts[:-1]) or counts[-1] > counts[0]:
        return False
    if total > n + counts[0]:
        return False
    return not complete or total >= n


def read_rietan_block(path: str | Path) -> PowderData:
    """RIETAN-FP の「点数 開始角 刻み」形式を読む。"""
    p = Path(path)
    text = decode_text(p.read_bytes())
    lines = _body_lines(text)
    if len(lines) < 2:
        raise ValueError(f"{p.name}: 数値データが読めませんでした")
    first = lines[0].replace(",", " ").split()
    n = int(float(first[0]))
    start = float(first[1])
    step = float(first[2])
    vals: list[float] = []
    for line in lines[1:]:
        for tok in line.replace(",", " ").split():
            if not _is_num(tok):
                break
            vals.append(float(tok.replace("D", "E").replace("d", "e")))
        if len(vals) >= n:
            break
    if len(vals) < n:
        raise ValueError(f"{p.name}: 強度が {n} 点あるはずが {len(vals)} 点しかありません")
    x = (start + step * np.arange(n, dtype=np.float64)).tolist()
    meta: dict[str, object] = {"format": "rietan", "n_points": n, "step": step}
    if lines and text.lstrip().startswith(("*", "#")):
        meta["comment"] = text.lstrip().splitlines()[0].lstrip("*# ").strip()
    return _powder(x, vals[:n], p.stem, meta)
