"""GSAS 標準粉末データ形式(BANK 行で始まるブロック)のリーダー。

対応:
    - BANK n NCHAN NREC CONST START STEP 0 0 [STD|ESD|FXYE|FXY]
      START, STEP は centidegree。
    - STD: 8 文字固定幅 (I2, F6.0) × 10 / 行。I2 は計数回数 n、強度 y、分散 y / n。
    - ESD: 16 文字 (F8.0 強度, F8.0 σ) × 5 / 行。
    - FXYE: 自由形式 (2θ(centideg), 強度, σ)。FXY: (2θ(centideg), 強度)。

GSAS-II の G2pwd_fxye.py と同じ解釈(強度 0 の点は重み 0)。

出典: GSAS Technical Manual (Larson & Von Dreele, 2004) Section "Powder data file formats"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kaisetsu.pattern.histogram import PowderData


def is_gsas_file(path: Path) -> bool:
    try:
        with path.open(encoding="latin-1") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                if line.startswith("BANK"):
                    return True
    except OSError:
        return False
    return False


def _sfloat(s: str) -> float:
    s = s.strip()
    return float(s) if s else 0.0


def _sint(s: str) -> int:
    s = s.strip()
    return int(s) if s else 0


def read_gsas(path: str | Path, bank: int = 1) -> PowderData:
    """GSAS 形式を読む。複数 BANK があれば bank 番号(1 始まり)で選ぶ。"""
    p = Path(path)
    lines = p.read_text(encoding="latin-1").splitlines()
    title = lines[0].strip() if lines else ""
    bank_lines = [i for i, line in enumerate(lines) if line.startswith("BANK")]
    if not bank_lines:
        raise ValueError(f"{p.name}: BANK 行がありません(GSAS 形式ではありません)")
    if not 1 <= bank <= len(bank_lines):
        raise ValueError(f"{p.name}: BANK {bank} はありません(BANK 数 {len(bank_lines)})")
    start_idx = bank_lines[bank - 1]
    end_idx = bank_lines[bank] if bank < len(bank_lines) else len(lines)
    header = lines[start_idx].split()
    body = [line for line in lines[start_idx + 1 : end_idx] if line and not line.startswith("#")]

    nchan = int(header[2])
    binning = header[4].upper() if len(header) > 4 else "CONST"
    if binning != "CONST":
        raise ValueError(f"{p.name}: ビニング {binning} は未対応です(CONST のみ)")
    start = float(header[5]) / 100.0
    step = float(header[6]) / 100.0
    fmt = header[9].upper() if len(header) > 9 else "STD"

    x: list[float] = []
    y: list[float] = []
    w: list[float] = []
    if fmt == "STD":
        j = 0
        for line in body:
            line = line.rstrip("\n")
            for i in range(0, 80, 8):
                field = line[i : i + 8]
                if len(field.strip()) == 0:
                    break
                n = max(_sint(field[:2]), 1)
                yi = max(_sfloat(field[2:8]), 0.0)
                x.append(start + step * j)
                y.append(yi)
                w.append(n / yi if yi > 0 else 0.0)
                j += 1
    elif fmt == "ESD":
        j = 0
        for line in body:
            for i in range(0, 80, 16):
                field = line[i : i + 16]
                if len(field.strip()) == 0:
                    break
                yi = _sfloat(field[:8])
                ei = _sfloat(field[8:16])
                x.append(start + step * j)
                if yi > 0 and ei > 0:
                    y.append(yi)
                    w.append(1.0 / ei**2)
                else:
                    y.append(0.0)
                    w.append(0.0)
                j += 1
    elif fmt in ("FXYE", "FXY"):
        for line in body:
            parts = line.split()
            if len(parts) < 2:
                continue
            x.append(float(parts[0]) / 100.0)
            yi = float(parts[1])
            y.append(yi)
            if fmt == "FXYE" and len(parts) >= 3:
                ei = float(parts[2])
                w.append(1.0 / ei**2 if ei > 0 else 0.0)
            else:
                w.append(1.0 / yi if yi > 0 else 0.0)
    else:
        raise ValueError(f"{p.name}: データ形式 {fmt} は未対応です")

    if len(x) > nchan:
        x, y, w = x[:nchan], y[:nchan], w[:nchan]
    return PowderData(
        np.array(x),
        np.array(y),
        np.array(w),
        name=p.stem,
        metadata={"format": "GSAS", "subformat": fmt, "title": title, "bank": bank, "path": str(p)},
    )
