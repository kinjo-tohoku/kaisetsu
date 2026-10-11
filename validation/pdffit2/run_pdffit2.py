"""diffpy.pdffit2 で PDF の参照値を作る(検証専用。kaisetsu の依存にはしない)。

    uv run python validation/pdffit2/run_pdffit2.py

pdffit2 の入った Python を環境変数 KAISETSU_PDFFIT2_PYTHON で指定する(既定はこのリポジトリの
scratch 用 venv)。出力は validation/reference/pdffit2_ni.json。

内容:
    calc: 固定パラメータ(a = 3.524, U_iso = 0.005, δ2 = 2.0, Q_damp = 0.06, Q_max = 40)
          での Ni の G(r)(r = 0.01〜20 Å、0.01 刻み)
    fit_xray / fit_neutron: data/public/pdf/Ni-*.gr を a, U_iso, δ2, Q_damp, スケールで精密化した結果
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "reference" / "pdffit2_ni.json"
DEFAULT_PY = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Temp/claude/C--Users-KatsukiKinjo-Documents-RietveldProgram"
    / "0ecb529a-245e-497f-8b31-0718b6725688/scratchpad/pdffit2env/.venv/Scripts/python.exe"
)

WORKER = r'''
import json, sys
import numpy as np
from diffpy.pdffit2 import PdfFit, redirect_stdout
redirect_stdout(open(sys.argv[3], "w"))
cif = sys.argv[1]; data_dir = sys.argv[2]
out = {}
# pdffit2 の C++ 側は CIF を読めないので diffpy.structure で PDFfit 形式に変換する
from diffpy.structure import loadStructure
import os, tempfile
_txt = open(cif, encoding="utf-8").read()
if not _txt.lstrip().startswith("data_"):
    _txt = "data_kaisetsu" + chr(10) + _txt  # diffpy.structure の CIF パーサは data_ 見出しを要求する
_cif2 = os.path.join(tempfile.gettempdir(), "kaisetsu_ni.cif")
open(_cif2, "w", encoding="utf-8").write(_txt)
_st = loadStructure(_cif2)
_stru = os.path.join(tempfile.gettempdir(), "kaisetsu_ni.stru")
_st.write(_stru, "pdffit")
cif = _stru

def setup(P, radiation, qmax, qdamp, uiso, delta2, a=None):
    P.read_struct(cif)
    if a is not None:
        P.setvar(P.lat(1), a); P.setvar(P.lat(2), a); P.setvar(P.lat(3), a)
    n = P.num_atoms()
    for i in range(1, n + 1):
        for u in (P.u11, P.u22, P.u33):
            P.setvar(u(i), uiso)
    P.setvar(P.delta2(), delta2)
    return n

# 1) 固定パラメータでの計算
P = PdfFit()
setup(P, "X", 40.0, 0.06, 0.005, 2.0, a=3.524)
P.alloc("X", 40.0, 0.06, 0.01, 20.0, 2000)
P.calc()
out["calc"] = {
    "a": 3.524, "uiso": 0.005, "delta2": 2.0, "qdamp": 0.06, "qmax": 40.0,
    "r": list(P.getR()), "g": list(P.getpdf_fit()),
}
# 中性子でも(単元素なので規格化は同じはず)
P = PdfFit()
setup(P, "N", 27.0, 0.02, 0.005, 0.0, a=3.524)
P.alloc("N", 27.0, 0.02, 0.01, 20.0, 2000)
P.calc()
out["calc_neutron"] = {"a": 3.524, "uiso": 0.005, "delta2": 0.0, "qdamp": 0.02, "qmax": 27.0,
                       "r": list(P.getR()), "g": list(P.getpdf_fit())}

# 2) 実データの精密化
def fit(fname, radiation, qmax, qdamp0, rmin, rmax):
    P = PdfFit()
    n = setup(P, radiation, qmax, qdamp0, 0.005, 2.0)
    P.read_data(fname, radiation, qmax, qdamp0)
    P.pdfrange(1, rmin, rmax)
    P.constrain(P.lat(1), 1); P.constrain(P.lat(2), 1); P.constrain(P.lat(3), 1)
    P.setpar(1, P.lat(1))
    P.constrain(P.dscale(), 2); P.setpar(2, 1.0)
    P.constrain(P.qdamp(), 3); P.setpar(3, qdamp0)
    for i in range(1, n + 1):
        for u in (P.u11, P.u22, P.u33):
            P.constrain(u(i), 4)
    P.setpar(4, 0.005)
    P.constrain(P.delta2(), 5); P.setpar(5, 2.0)
    P.refine()
    res = {"a": P.getpar(1), "dscale": P.getpar(2), "qdamp": P.getpar(3), "uiso": P.getpar(4),
           "delta2": P.getpar(5), "rw": P.getrw(), "rmin": rmin, "rmax": rmax, "qmax": qmax}
    esd = {}
    for k, name in zip(range(1, 6), ("a", "dscale", "qdamp", "uiso", "delta2")):
        try:
            esd[name] = P.getpar_esd(k) if hasattr(P, "getpar_esd") else None
        except Exception:
            esd[name] = None
    res["esd"] = esd
    res["r"] = list(P.getR()); res["g_obs"] = list(P.getpdf_obs()); res["g_fit"] = list(P.getpdf_fit())
    return res

out["fit_xray"] = fit(data_dir + "/Ni-xray.gr", "X", 40.0, 0.06, 1.5, 20.0)
out["fit_neutron"] = fit(data_dir + "/Ni-neutron.gr", "N", 27.0, 0.02, 1.5, 20.0)
json.dump(out, open(sys.argv[4], "w"))
'''


def main() -> None:
    py = Path(os.environ.get("KAISETSU_PDFFIT2_PYTHON", str(DEFAULT_PY)))
    if not py.exists():
        sys.exit(f"pdffit2 の Python が見つかりません: {py}(KAISETSU_PDFFIT2_PYTHON で指定)")
    with tempfile.TemporaryDirectory() as td:
        worker = Path(td) / "worker.py"
        worker.write_text(WORKER, encoding="utf-8")
        log = Path(td) / "pdffit2.log"
        result = Path(td) / "out.json"
        subprocess.run(
            [
                str(py),
                str(worker),
                str(ROOT / "data/public/pdf/Ni.cif"),
                str(ROOT / "data/public/pdf"),
                str(log),
                str(result),
            ],
            check=True,
        )
        data = json.loads(result.read_text(encoding="utf-8"))
    data["pdffit2_version"] = subprocess.run(
        [str(py), "-c", "import diffpy.pdffit2 as m; print(m.__version__)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data), encoding="utf-8")
    fx, fn = data["fit_xray"], data["fit_neutron"]
    print(f"書き出し: {OUT}")
    print(f"X 線: a = {fx['a']:.5f}  Uiso = {fx['uiso']:.5f}  δ2 = {fx['delta2']:.3f}  "
          f"Qdamp = {fx['qdamp']:.4f}  scale = {fx['dscale']:.4f}  Rw = {fx['rw']:.4f}")  # fmt: skip
    print(f"中性子: a = {fn['a']:.5f}  Uiso = {fn['uiso']:.5f}  δ2 = {fn['delta2']:.3f}  "
          f"Qdamp = {fn['qdamp']:.4f}  scale = {fn['dscale']:.4f}  Rw = {fn['rw']:.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
