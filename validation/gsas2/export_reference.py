"""GSAS-II で参照値を JSON に書き出す。

**gsas2main 同梱の Python で実行すること**(kaisetsu の環境ではない)。
通常は validation/gsas2/run_gsas2.py 経由で呼ばれる。

使い方:
    python export_reference.py --cif PHASE.cif --instprm INST.prm
        (--data DATA.xra | --simulate TMIN TMAX TSTEP)
        --out ref.json [--phase-name NAME]

書き出す内容:
    - 反射リスト(h, k, l, 多重度, d, 2θ位置, |Fobs|^2, |Fcalc|^2, 位相 ...)
    - 反射ごとの原子散乱因子 f0(X線のとき)
    - 相の情報(格子、空間群、原子リスト、散乱因子係数、f', f''、中性子散乱長)
    - 装置パラメータ、試料パラメータ、バックグラウンド、相-ヒストグラムパラメータ
    - 観測・計算パターン(x, yobs, ycalc, background, weight)と R 因子
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HC_KEV_ANGSTROM = 12.398419843320026


def to_jsonable(obj, depth=0):
    if depth > 12:
        return repr(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, depth + 1) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--instprm", required=True)
    ap.add_argument("--data")
    ap.add_argument("--simulate", nargs=3, type=float, metavar=("TMIN", "TMAX", "TSTEP"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase-name", default="phase")
    args = ap.parse_args()
    if not args.data and not args.simulate:
        ap.error("--data か --simulate のどちらかが必要です")

    g2dir = os.environ.get("KAISETSU_GSAS2_DIR", r"C:\Users\KatsukiKinjo\gsas2main\GSAS-II")
    sys.path.insert(0, g2dir)
    from GSASII import GSASIIscriptable as G2sc

    try:
        from GSASII import GSASIIElem as G2el
    except Exception:
        G2el = None
    try:
        from GSASII import atmdata
    except Exception:
        atmdata = None

    workdir = tempfile.mkdtemp(prefix="kaisetsu_g2_")
    gpx = G2sc.G2Project(newgpx=os.path.join(workdir, "ref.gpx"))
    if args.data:
        hist = gpx.add_powder_histogram(os.path.abspath(args.data), os.path.abspath(args.instprm))
    else:
        tmin, tmax, tstep = args.simulate
        hist = gpx.add_simulated_powder_histogram(
            "sim", os.path.abspath(args.instprm), tmin, tmax, Tstep=tstep
        )
    phase = gpx.add_phase(os.path.abspath(args.cif), phasename=args.phase_name, histograms=[hist])
    # 0 サイクル: 計算パターンと反射リストを得るだけ。何か 1 つ精密化フラグが必要。
    hist.set_refinements({"Sample Parameters": ["Scale"]})
    gpx.set_Controls("cycles", 0)
    gpx.refine()

    out: dict = {"gsas2_dir": g2dir, "case": vars(args)}
    try:
        from GSASII import GSASIIpath

        out["gsas2_version"] = to_jsonable(GSASIIpath.GetVersionNumber())
    except Exception as exc:
        out["gsas2_version"] = repr(exc)

    # ---- 反射リスト --------------------------------------------------
    refl_all = hist.reflections()
    key = phase.name if phase.name in refl_all else next(iter(refl_all))
    refl = refl_all[key]
    reflist = np.asarray(refl["RefList"], dtype=float)
    ncol = reflist.shape[1]
    base_cols = [
        "h",
        "k",
        "l",
        "mult",
        "d",
        "pos",
        "sig",
        "gam",
        "Fobs2",
        "Fcalc2",
        "phase",
        "Icorr",
        "Prfo",
        "Trans",
        "ExtP",
    ]
    cols = base_cols[:ncol] + [f"col{i}" for i in range(len(base_cols), ncol)]
    out["reflections"] = {
        "columns": cols,
        "data": reflist.tolist(),
        "type": refl.get("Type"),
        "FF": to_jsonable(refl.get("FF", {})),
    }

    # ---- 相の情報 ----------------------------------------------------
    general = phase.data["General"]
    sg = general["SGData"]
    cx, ct, cs, cia = general["AtomPtrs"]
    atoms = []
    for row in phase.data["Atoms"]:
        atoms.append(
            {
                "label": row[ct - 1],
                "type": row[ct],
                "xyz": [float(v) for v in row[cx : cx + 3]],
                "frac": float(row[cx + 3]),
                "site_sym": row[cs],
                "mult": int(row[cs + 1]),
                "iso_or_aniso": row[cia],
                "Uiso": float(row[cia + 1]),
                "Uij": [float(v) for v in row[cia + 2 : cia + 8]],
            }
        )
    out["phase"] = {
        "name": phase.name,
        "cell": [float(v) for v in general["Cell"][1:8]],
        "spacegroup": sg.get("SpGrp"),
        "laue": sg.get("SGLaue"),
        "centrosymmetric": bool(sg.get("SGInv")),
        "lattice": sg.get("SGLatt"),
        "system": sg.get("SGSys"),
        "n_symops": len(sg.get("SGOps", [])),
        "n_centering": len(sg.get("SGCen", [])),
        "symops": [[to_jsonable(m), to_jsonable(t)] for m, t in sg.get("SGOps", [])],
        "centering": to_jsonable(sg.get("SGCen", [])),
        "atom_types": list(general.get("AtomTypes", [])),
        "no_atoms": to_jsonable(general.get("NoAtoms", {})),
        "type": general.get("Type"),
        "atoms": atoms,
        "FFtables": to_jsonable(general.get("FFtables", {})),
        "BLtables": to_jsonable(general.get("BLtables", {})),
    }

    # ---- 装置・試料・背景 ----------------------------------------------
    inst = hist.getHistEntryValue(["Instrument Parameters"])[0]
    out["instrument"] = {
        k: (v[1] if isinstance(v, list) and len(v) == 3 else to_jsonable(v))
        for k, v in inst.items()
    }
    out["sample"] = to_jsonable(hist.data.get("Sample Parameters", {}))
    out["background"] = to_jsonable(hist.data.get("Background", []))
    out["limits"] = to_jsonable(hist.data.get("Limits", []))
    out["phase_hist"] = to_jsonable(phase.data["Histograms"].get(hist.name, {}))

    # ---- f', f'' と散乱因子係数 ---------------------------------------
    lam = None
    for k in ("Lam1", "Lam"):
        if k in out["instrument"]:
            lam = float(out["instrument"][k])
            break
    anomalous = {}
    ff_coeff = {}
    blens = {}
    for el in general.get("AtomTypes", []):
        if G2el is not None and lam:
            try:
                fp, fpp, mu = G2el.FPcalc(G2el.GetXsectionCoeff(el), HC_KEV_ANGSTROM / lam)
                anomalous[el] = {"fp": float(fp), "fpp": float(fpp), "mu": float(mu)}
            except Exception as exc:
                anomalous[el] = repr(exc)
            try:
                ff_coeff[el] = to_jsonable(G2el.GetFormFactorCoeff(el))
            except Exception as exc:
                ff_coeff[el] = repr(exc)
        if atmdata is not None:
            try:
                blens[el] = to_jsonable(atmdata.AtmBlens.get(el + "_", {}))
            except Exception as exc:
                blens[el] = repr(exc)
    out["anomalous"] = anomalous
    out["form_factor_coeff"] = ff_coeff
    out["neutron_blens"] = blens
    out["wavelength"] = lam

    # ---- パターン -----------------------------------------------------
    pattern = {}
    for name in ("x", "yobs", "ycalc", "background", "yweight"):
        try:
            pattern[name] = [float(v) for v in hist.getdata(name)]
        except Exception as exc:
            pattern[name] = repr(exc)
    out["pattern"] = pattern
    try:
        out["rvals"] = to_jsonable(gpx.data["Covariance"]["data"].get("Rvals", {}))
    except Exception as exc:
        out["rvals"] = repr(exc)
    try:
        out["hist_residuals"] = to_jsonable(hist.data.get("data", [{}])[0])
    except Exception as exc:
        out["hist_residuals"] = repr(exc)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(reflist)} reflections, {ncol} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
