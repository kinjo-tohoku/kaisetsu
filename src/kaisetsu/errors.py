"""計算を安全に止めるための例外。"""

from __future__ import annotations


class PatternSizeError(ValueError):
    """パターン計算が大きすぎる(hkl の候補やピーク窓の点数が上限を超えた)。

    精密化の途中で格子定数やピーク幅が発散すると、hkl の候補は格子定数の 3 乗で、
    ピーク窓は幅に比例して増え、メモリを使い尽くして落ちる(opXRD 416 の Le Bail で
    21 GB まで膨らんだ)。その前にこの例外で止める。最小二乗の残差評価
    (`kaisetsu.refine.lsq`)はこれを「その試行点は不可」として扱い、精密化自体は続ける。
    """
