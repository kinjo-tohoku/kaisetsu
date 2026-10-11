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

# 精密化の段階(kaisetsu 側の検証テストと同じ順序)
_FIT_STAGES = [
    "--refine", "bkg:6,scale",
    "--refine", "cell,zero",
    "--refine", "w",
    "--refine", "u,v,x,y",
    "--refine", "xyz",
    "--refine", "uiso",
]  # fmt: skip

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
    # 非対称・ローレンツ幅・試料変位・選択配向・偏光を効かせたケース(Phase 2 の検証用)
    "pbso4_xray_asym": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": DATA / "pbso4" / "PBSO4.XRA",
        "instprm": DATA / "pbso4" / "INST_XRY.PRM",
        "phase_name": "PbSO4",
        "extra": [
            "--set-inst",
            "SH/L=0.05",
            "--set-inst",
            "X=1.0",
            "--set-inst",
            "Y=3.0",
            "--set-inst",
            "Zero=0.02",
            "--set-inst",
            "Polariz.=0.5",
            "--set-sample",
            "Shift=100",
            "--md",
            "0.8",
            "0",
            "0",
            "1",
            "--size",
            "0.3",
            "--mustrain",
            "2000",
        ],
    },
    # Phase 3: GSAS-II に実際に精密化させる(段階的にフラグを増やす)
    "pbso4_xray_fit": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": DATA / "pbso4" / "PBSO4.XRA",
        "instprm": DATA / "pbso4" / "INST_XRY.PRM",
        "phase_name": "PbSO4",
        "extra": [*_FIT_STAGES, "--refine", "uiso", "--refine", "uiso", "--cycles", "30"],
    },
    "pbso4_neutron_fit": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": DATA / "pbso4" / "PBSO4.CWN",
        "instprm": DATA / "pbso4" / "inst_d1a.prm",
        "phase_name": "PbSO4",
        "extra": [*_FIT_STAGES, "--refine", "uiso", "--refine", "uiso", "--cycles", "30"],
    },
    "pbso4_combined_fit": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": [DATA / "pbso4" / "PBSO4.XRA", DATA / "pbso4" / "PBSO4.CWN"],
        "instprm": [DATA / "pbso4" / "INST_XRY.PRM", DATA / "pbso4" / "inst_d1a.prm"],
        "phase_name": "PbSO4",
        "extra": [*_FIT_STAGES, "--refine", "uiso", "--refine", "uiso", "--cycles", "30"],
    },
    # 中性子: 円筒試料の吸収と Debye-Scherrer の試料変位
    "pbso4_neutron_abs": {
        "cif": DATA / "pbso4" / "PbSO4-Wyckoff.cif",
        "data": DATA / "pbso4" / "PBSO4.CWN",
        "instprm": DATA / "pbso4" / "inst_d1a.prm",
        "phase_name": "PbSO4",
        "extra": [
            "--set-inst",
            "X=2.0",
            "--set-inst",
            "Y=4.0",
            "--set-inst",
            "SH/L=0.03",
            "--set-sample",
            "Absorption=0.5",
            "--set-sample",
            "DisplaceX=100",
            "--set-sample",
            "DisplaceY=-50",
        ],
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
        "--out",
        str(out),
        "--phase-name",
        str(case["phase_name"]),
    ]
    instprms = case["instprm"] if isinstance(case["instprm"], list) else [case["instprm"]]
    if "data" in case:
        datas = case["data"] if isinstance(case["data"], list) else [case["data"]]
        for d, ip in zip(datas, instprms, strict=True):
            cmd += ["--data", str(d), "--instprm", str(ip)]
    else:
        cmd += ["--instprm", str(instprms[0])]
        cmd += ["--simulate", *(str(v) for v in case["simulate"])]  # type: ignore[union-attr]
    cmd += [str(v) for v in case.get("extra", [])]  # type: ignore[union-attr]
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
