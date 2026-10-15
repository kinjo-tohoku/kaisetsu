"""pdCIF(粉末回折 CIF)の測定データを読む。

IUCr の Acta Cryst. E や COD に付く GSAS2CIF / GSAS-II の出力には、構造だけでなく測定した粉末
パターン(``_pd_meas_counts_total`` など)が入っていることがある(例: COD 2240592 の Ba(ClO4)2)。
公開データで練習・検証するときにそのまま読めるようにする。

- 強度: ``_pd_meas_counts_total`` / ``_pd_meas_intensity_total`` /
  ``_pd_proc_intensity_total`` の順に探す
- 2θ: 同じ loop の ``_pd_meas_2theta_scan`` / ``_pd_proc_2theta_corrected`` /
  ``_pd_meas_2theta_corrected``。無ければ ``_pd_meas_2theta_range_min`` と
  ``_pd_meas_2theta_range_inc``(または ``_pd_proc_`` 版)から等間隔
- 重み: ``_pd_proc_ls_weight`` があればそれ(0 の点は精密化から除外された点なので落とす)。
  無ければ Poisson
- 欠損(``.``、``?``)の点は落とす
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

from kaisetsu.pattern.histogram import PowderData

INTENSITY_TAGS = (
    "_pd_meas_counts_total",
    "_pd_meas_intensity_total",
    "_pd_proc_intensity_total",
    "_pd_proc_intensity_net",
)
TWO_THETA_TAGS = (
    "_pd_meas_2theta_scan",
    "_pd_proc_2theta_corrected",
    "_pd_meas_2theta_corrected",
)


def _column(block: gemmi.cif.Block, tag: str) -> np.ndarray | None:
    vals = list(block.find_loop(tag))
    if not vals:
        return None
    return np.array(
        [float(v.split("(")[0]) if v not in (".", "?") else np.nan for v in vals],
        dtype=np.float64,
    )


def _number(block: gemmi.cif.Block, tag: str) -> float | None:
    v = block.find_value(tag)
    if v is None or v in (".", "?"):
        return None
    try:
        return float(v.split("(")[0])
    except ValueError:
        return None


def read_pdcif(path: str | Path) -> PowderData:
    """pdCIF の粉末データを読む。粉末データの loop が無ければ ValueError。"""
    p = Path(path)
    doc = gemmi.cif.read(str(p))
    for block in doc:
        y: np.ndarray | None = None
        used = ""
        for tag in INTENSITY_TAGS:
            y = _column(block, tag)
            if y is not None:
                used = tag
                break
        if y is None:
            continue
        n = len(y)
        x: np.ndarray | None = None
        for t in TWO_THETA_TAGS:
            c = _column(block, t)
            if c is not None and len(c) == n:
                x = c
                break
        if x is None:
            for pre in ("_pd_meas", "_pd_proc"):
                lo = _number(block, f"{pre}_2theta_range_min")
                inc = _number(block, f"{pre}_2theta_range_inc")
                if lo is not None and inc is not None:
                    x = lo + inc * np.arange(n, dtype=np.float64)
                    break
        if x is None:
            raise ValueError(
                f"pdCIF {p.name}: 2θ の情報(_pd_meas_2theta_scan か "
                "_pd_meas_2theta_range_min/inc)がありません"
            )
        w = _column(block, "_pd_proc_ls_weight")
        keep = np.isfinite(x) & np.isfinite(y)
        if w is not None and len(w) == n:
            keep &= np.isfinite(w) & (w > 0)
        else:
            w = None
        meta: dict[str, object] = {
            "format": "pdCIF",
            "block": block.name,
            "intensity_tag": used,
            "n_dropped": int(np.count_nonzero(~keep)),
        }
        lam = _number(block, "_diffrn_radiation_wavelength")
        if lam is not None:
            meta["wavelength"] = lam
        rt = block.find_value("_diffrn_radiation_type")
        if rt is not None:
            meta["radiation_type"] = gemmi.cif.as_string(rt)
        x, y = x[keep], y[keep]
        order = np.argsort(x, kind="stable")
        weight = w[keep][order] if w is not None else None
        return PowderData(x[order], y[order], weight, p.name, meta)
    raise ValueError(
        f"{p.name}: 粉末データの loop(_pd_meas_counts_total など)がありません。"
        "構造だけの CIF は「データ → CIF から相を追加」で読んでください"
    )


__all__ = ["read_pdcif"]
