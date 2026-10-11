"""GSAS-II の Sabine 粉末消衰補正(GSASIIstrMath.GetPwdrExt)の参照値を作る。

使い方(kaisetsu の環境から):
    uv run python validation/gsas2/extinction_reference.py

内部で gsas2main の Python を subprocess で呼び、X 線・中性子それぞれ 6 ケースの補正値を
validation/reference/sabine_extinction.json に書く。
テストは tests/validation/test_gsas2_extinction.py。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "validation" / "reference" / "sabine_extinction.json"

#: (|F|² [(10⁻¹² cm)² 相当の GSAS-II 単位], 2θ [deg], λ [Å], V [Å³], E)
CASES: list[tuple[float, float, float, float, float]] = [
    (100.0, 60.0, 1.54, 43.8, 50.0),
    (5000.0, 30.0, 1.54, 43.8, 50.0),
    (20000.0, 120.0, 1.54, 43.8, 100.0),
    (50.0, 90.0, 2.5, 160.0, 1.0),
    (500.0, 90.0, 3.0, 160.0, 5.0),
    (900.0, 150.0, 4.0, 100.0, 20.0),
]


def _inner() -> None:
    """gsas2main の Python で実行される部分。"""
    sys.path.insert(0, r"C:\Users\KatsukiKinjo\gsas2main\GSAS-II")
    os.environ["GSASII_QUIET"] = "1"
    from GSASII import GSASIIstrMath as G2stMth  # type: ignore[import-not-found]

    rows = []
    for hist_type, is_x in (("PXC", True), ("PNT", False)):
        for f2, tt, lam, vol, ext in CASES:
            refl = [0.0] * 20
            refl[5] = tt
            refl[9] = f2
            refl[14] = lam
            calc = {"h:histType": hist_type}
            parm = {"h:2-theta": tt, "h:Lam": lam, "p:Vol": vol, "p:h:Extinction": ext}
            v = float(G2stMth.GetPwdrExt(refl, 0, "p:", "p:h:", "h:", calc, parm))
            rows.append(
                {"xray": is_x, "f2": f2, "tt": tt, "lam": lam, "vol": vol, "ext": ext, "e": v}
            )
    print(json.dumps(rows))


def main() -> int:
    if "--inner" in sys.argv:
        _inner()
        return 0
    sys.path.insert(0, str(Path(__file__).parent))
    from run_gsas2 import gsas2_python

    proc = subprocess.run(
        [str(gsas2_python()), "-I", str(Path(__file__)), "--inner"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(proc.stdout.strip().splitlines()[-1])
    REF.write_text(
        json.dumps({"source": "GSAS-II GSASIIstrMath.GetPwdrExt", "cases": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"書き出し: {REF}({len(rows)} ケース)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
