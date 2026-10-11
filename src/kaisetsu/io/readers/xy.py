"""ASCII 2〜3 列(2θ, 強度[, σ])のリーダー。

'#', '!', '/', ';' で始まる行とヘッダ文字列は読み飛ばす。区切りは空白・カンマ・タブ。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from kaisetsu.pattern.histogram import PowderData

_SPLIT = re.compile(r"[\s,;]+")


def read_xy(path: str | Path) -> PowderData:
    p = Path(path)
    xs: list[float] = []
    ys: list[float] = []
    es: list[float] = []
    with p.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
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
    if len(xs) < 2:
        raise ValueError(f"{p.name}: 数値データが読めませんでした")
    x = np.array(xs)
    y = np.array(ys)
    e = np.array(es)
    if np.all(np.isfinite(e)) and np.any(e > 0):
        with np.errstate(divide="ignore"):
            w = np.where(e > 0, 1.0 / e**2, 0.0)
    else:
        with np.errstate(divide="ignore"):
            w = np.where(y > 0, 1.0 / np.maximum(y, 1e-300), 0.0)
    order = np.argsort(x, kind="stable")
    return PowderData(
        x[order], y[order], w[order], name=p.stem, metadata={"format": "xy", "path": str(p)}
    )
