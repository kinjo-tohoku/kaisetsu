"""回折データのリーダー。拡張子と中身から形式を判定する。"""

from __future__ import annotations

from pathlib import Path

from kaisetsu.io.readers.gsas import is_gsas_file, read_gsas
from kaisetsu.io.readers.xy import read_xy
from kaisetsu.pattern.histogram import PowderData


def read_powder_data(path: str | Path, **kwargs: object) -> PowderData:
    """粉末回折データを読む。

    対応形式: GSAS 標準形式(.gsa/.gsas/.raw/.xra/.cwn 等、BANK 行を持つもの)、
    ASCII 2〜3 列(.xy/.xye/.dat/.txt: 2θ, 強度[, σ])。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {p}")
    if is_gsas_file(p):
        return read_gsas(p, **kwargs)  # type: ignore[arg-type]
    return read_xy(p)
