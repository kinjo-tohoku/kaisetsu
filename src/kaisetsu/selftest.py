"""配布物の起動確認(``kaisetsu-gui --smoke-test``)で使う簡単な自己診断。

サンプルデータ(環境変数 ``KAISETSU_SAMPLE_DIR``、無ければリポジトリの data/public)の蛍石燐灰石を
読み、パターンを 1 回計算して R 因子が有限であることを確かめる。gemmi・numba・scipy が配布物に
正しく入っているかをまとめて確認できる。GUI には依存しない。
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path


def sample_dir() -> Path | None:
    """サンプルデータのフォルダ(fap/FAP.XRA があるところ)。無ければ None。"""
    candidates: list[Path] = []
    env = os.environ.get("KAISETSU_SAMPLE_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "public")
    for c in candidates:
        if (c / "fap" / "FAP.XRA").is_file():
            return c
    return None


def quick_check() -> str:
    """数値計算の経路をひととおり通し、結果を文字列で返す。失敗すれば例外を投げる。"""
    import numpy as np

    from kaisetsu.api import (
        CWInstrument,
        Project,
        Radiation,
        SampleGeometry,
        Structure,
        read_powder_data,
    )
    from kaisetsu.pattern.instrument import GeometryKind

    t0 = time.perf_counter()
    lines = [f"numpy {np.__version__}"]
    d = sample_dir()
    if d is None:
        lines.append("sample data: not found (pattern check skipped)")
        return "\n".join(lines) + "\n"
    proj = Project("selftest")
    proj.add_phase(Structure.from_cif(str(d / "fap" / "fluorapatite_COD9001232.cif")), name="FAP")
    inst = CWInstrument(
        Radiation.xray(1.5405929, 1.5444274, 0.5),
        u=0.004, v=-0.003, w=0.004, x=0.02, y=0.02, s_l=0.02, h_l=0.02, polarization=0.7,
    )  # fmt: skip
    proj.add_histogram(
        read_powder_data(d / "fap" / "FAP.XRA"),
        inst,
        SampleGeometry(GeometryKind.BRAGG_BRENTANO),
        n_background=6,
    )
    rf = proj.rfactors()["total"]
    if not math.isfinite(rf.rwp):
        raise RuntimeError("Rwp が有限でない")
    lines.append(f"sample data: {d}")
    lines.append(f"FAP pattern: Rwp {100 * rf.rwp:.1f}% ({time.perf_counter() - t0:.1f} s)")
    return "\n".join(lines) + "\n"


__all__ = ["quick_check", "sample_dir"]
