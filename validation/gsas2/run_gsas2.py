"""GSAS-II を別プロセスで実行して参照値 JSON を validation/reference/ に作る。

使い方(kaisetsu の環境から):
    uv run python validation/gsas2/run_gsas2.py            # 全ケース
    uv run python validation/gsas2/run_gsas2.py pbso4_xray # 指定ケース

GSAS-II の Python は環境変数 KAISETSU_GSAS2_PYTHON で上書きできる。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "public"
REF_DIR = ROOT / "validation" / "reference"
EXPORT = Path(__file__).with_name("export_reference.py")

DEFAULT_G2_PYTHON = r"C:\Users\KatsukiKinjo\gsas2main\python.exe"

CASES: dict[str, dict[str, object]] = {
    "pbso4_xray": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": DATA / "pbso4" / "PBSO4.XRA",
        "instprm": DATA / "pbso4" / "INST_XRY.PRM",
        "phase_name": "PbSO4",
    },
    "pbso4_neutron": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": DATA / "pbso4" / "PBSO4.CWN",
        "instprm": DATA / "pbso4" / "inst_d1a.prm",
        "phase_name": "PbSO4",
    },
    "diamond_xray_sim": {
        "cif": DATA / "cif" / "diamond.cif",
        "simulate": (10.0, 150.0, 0.02),
        "instprm": DATA / "pbso4" / "INST_XRY.PRM",
        "phase_name": "diamond",
    },
}


def gsas2_python() -> Path:
    p = Path(os.environ.get("KAISETSU_GSAS2_PYTHON", DEFAULT_G2_PYTHON))
    if not p.exists():
        raise SystemExit(
            f"GSAS-II の Python が見つかりません: {p}\n"
            "環境変数 KAISETSU_GSAS2_PYTHON に gsas2main/python.exe のパスを設定してください"
        )
    return p


def run_case(name: str) -> Path:
    case = CASES[name]
    REF_DIR.mkdir(parents=True, exist_ok=True)
    out = REF_DIR / f"{name}.json"
    py = gsas2_python()
    cmd = [
        str(py),
        str(EXPORT),
        "--cif",
        str(case["cif"]),
        "--instprm",
        str(case["instprm"]),
        "--out",
        str(out),
        "--phase-name",
        str(case["phase_name"]),
    ]
    if "data" in case:
        cmd += ["--data", str(case["data"])]
    else:
        cmd += ["--simulate", *(str(v) for v in case["simulate"])]  # type: ignore[union-attr]
    env = dict(os.environ)
    # conda 環境の DLL を見つけられるように
    env["PATH"] = os.pathsep.join(
        [
            str(py.parent),
            str(py.parent / "Library" / "bin"),
            str(py.parent / "Scripts"),
            env.get("PATH", ""),
        ]
    )
    env.setdefault("KAISETSU_GSAS2_DIR", str(py.parent / "GSAS-II"))
    print(f"[{name}] running GSAS-II ...", flush=True)
    res = subprocess.run(cmd, env=env, cwd=str(REF_DIR), capture_output=True, text=True)
    log = REF_DIR / f"{name}.log"
    log.write_text(res.stdout + "\n--- stderr ---\n" + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        print(res.stdout[-3000:])
        print(res.stderr[-3000:])
        raise SystemExit(f"[{name}] GSAS-II の実行に失敗しました(ログ: {log})")
    print(f"[{name}] -> {out}")
    return out


def main(argv: list[str]) -> int:
    names = argv[1:] or list(CASES)
    for n in names:
        if n not in CASES:
            raise SystemExit(f"未知のケース: {n}。選択肢: {', '.join(CASES)}")
        run_case(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
