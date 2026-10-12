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


def is_rietan_block(head: str) -> bool:
    """先頭部分(数千文字)から「点数 開始角 刻み + 強度の羅列」形式かを判定する。"""
    lines = _body_lines(head)
    if len(lines) < 2:
        return False
    first = lines[0].replace(",", " ").split()
    if len(first) != 3 or not all(_is_num(t) for t in first):
        return False
    try:
        n = int(float(first[0]))
        step = float(first[2])
    except ValueError:
        return False
    if n < 10 or step <= 0.0 or float(first[0]) != n:
        return False
    # FullProf の「開始角 刻み 終了角」(3 番目が終了角で大きい)と区別する: RIETAN は 3 番目が刻み
    if step >= float(first[1]) and float(first[1]) > 0:
        return False
    second = lines[1].replace(",", " ").split()
    # 2 行目が「2θ 強度」の 2 列なら GENERAL 類似(点数の行が無い)なので対象外。
    # 強度だけの行(1 個以上、全部数値)なら本形式
    return (
        len(second) >= 1
        and all(_is_num(t) for t in second)
        and (len(second) != 2 or abs(float(second[0]) - float(first[1])) > 1e-9)
    )


def read_rietan_block(path: str | Path) -> PowderData:
    """RIETAN-FP の「点数 開始角 刻み」形式を読む。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
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
