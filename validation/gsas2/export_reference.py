"""GSAS-II で参照値を JSON に書き出す。

**gsas2main 同梱の Python で実行すること**(kaisetsu の環境ではない)。
通常は validation/gsas2/run_gsas2.py 経由で呼ばれる。

使い方:
    python export_reference.py --cif PHASE.cif
        (--data DATA --instprm INST [--data DATA2 --instprm INST2 ...]
         | --simulate TMIN TMAX TSTEP --instprm INST)
        --out ref.json [--phase-name NAME]
        [--set-inst KEY=VAL ...] [--set-sample KEY=VAL ...] [--md R H K L] [--size S] [--mustrain M]
        [--refine STAGE ...] [--cycles N]

--refine を与えなければ 0 サイクル(計算のみ)。与えた場合は各 STAGE を順に実行する。
STAGE はカンマ区切りのトークン: bkg:N, scale, cell, zero, u, v, w, x, y, shl, shift, dispx, dispy,
absorption, xyz, uiso, size, strain, md。フラグは累積する。

書き出す内容:
    - ヒストグラムごと: 反射リスト、反射ごとの f0、装置・試料・背景・相-ヒストグラムパラメータ、
      観測・計算パターン、R 因子
    - 相の情報(格子、空間群、原子リスト、散乱因子係数、f', f''、中性子散乱長)
    - 精密化した場合は変数リスト・値・esd・Rvals・共分散
後方互換のため、最初のヒストグラムの内容はトップレベルにも置く。
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


_INST_TOKENS = {"zero": "Zero", "u": "U", "v": "V", "w": "W", "x": "X", "y": "Y", "shl": "SH/L"}
_SAMPLE_TOKENS = {
    "scale": "Scale",
    "shift": "Shift",
    "dispx": "DisplaceX",
    "dispy": "DisplaceY",
    "absorption": "Absorption",
    "transparency": "Transparency",
}


def apply_stage(gpx, hists, phase, stage: str, atom_flags: set) -> None:
    """1 段階分の精密化フラグを立てる(累積)。"""
    for tok in [t.strip() for t in stage.split(",") if t.strip()]:
        if tok.startswith("bkg"):
            n = int(tok.split(":")[1]) if ":" in tok else 6
            for h in hists:
                h.set_refinements({"Background": {"no. coeffs": n, "refine": True}})
        elif tok in _INST_TOKENS:
            for h in hists:
                h.set_refinements({"Instrument Parameters": [_INST_TOKENS[tok]]})
        elif tok in _SAMPLE_TOKENS:
            for h in hists:
                h.set_refinements({"Sample Parameters": [_SAMPLE_TOKENS[tok]]})
        elif tok == "cell":
            phase.set_refinements({"Cell": True})
        elif tok == "xyz":
            atom_flags.add("X")
            phase.set_refinements({"Atoms": {"all": "".join(sorted(atom_flags))}})
        elif tok == "uiso":
            atom_flags.add("U")
            phase.set_refinements({"Atoms": {"all": "".join(sorted(atom_flags))}})
        elif tok == "occ":
            atom_flags.add("F")
            phase.set_refinements({"Atoms": {"all": "".join(sorted(atom_flags))}})
        elif tok == "size":
            phase.set_HAP_refinements(
                {"Size": {"type": "isotropic", "refine": True}}, histograms=hists
            )
        elif tok == "strain":
            phase.set_HAP_refinements(
                {"Mustrain": {"type": "isotropic", "refine": True}}, histograms=hists
            )
        elif tok == "md":
            phase.set_HAP_refinements({"Pref.Ori.": True}, histograms=hists)
        elif tok == "phasescale":
            phase.set_HAP_refinements({"Scale": True}, histograms=hists)
        else:
            raise SystemExit(f"未知の精密化トークン: {tok!r}")


def export_histogram(hist, phase, general, G2el, atmdata) -> dict:
    out: dict = {"name": hist.name}
    refl_all = hist.reflections()
    key = phase.name if phase.name in refl_all else next(iter(refl_all))
    refl = refl_all[key]
    reflist = np.asarray(refl["RefList"], dtype=float)
    ncol = reflist.shape[1]
    base_cols = [
        "h", "k", "l", "mult", "d", "pos", "sig", "gam",
        "Fobs2", "Fcalc2", "phase", "Icorr", "Prfo", "Trans", "ExtP",
    ]  # fmt: skip
    cols = base_cols[:ncol] + [f"col{i}" for i in range(len(base_cols), ncol)]
    out["reflections"] = {
        "columns": cols,
        "data": reflist.tolist(),
        "type": refl.get("Type"),
        "FF": to_jsonable(refl.get("FF", {})),
    }
    inst = hist.getHistEntryValue(["Instrument Parameters"])[0]
    out["instrument"] = {
        k: (v[1] if isinstance(v, list) and len(v) == 3 else to_jsonable(v))
        for k, v in inst.items()
    }
    out["sample"] = to_jsonable(hist.data.get("Sample Parameters", {}))
    out["background"] = to_jsonable(hist.data.get("Background", []))
    out["limits"] = to_jsonable(hist.data.get("Limits", []))
    out["phase_hist"] = to_jsonable(phase.data["Histograms"].get(hist.name, {}))

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

    pattern = {}
    for name in ("x", "yobs", "ycalc", "background", "yweight"):
        try:
            pattern[name] = [float(v) for v in hist.getdata(name)]
        except Exception as exc:
            pattern[name] = repr(exc)
    out["pattern"] = pattern
    try:
        out["hist_residuals"] = to_jsonable(hist.data.get("data", [{}])[0])
    except Exception as exc:
        out["hist_residuals"] = repr(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--instprm", action="append", default=[])
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--simulate", nargs=3, type=float, metavar=("TMIN", "TMAX", "TSTEP"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase-name", default="phase")
    ap.add_argument("--set-inst", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--set-sample", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--md", nargs=4, type=float, metavar=("R", "H", "K", "L"))
    ap.add_argument("--size", type=float)
    ap.add_argument("--mustrain", type=float)
    ap.add_argument("--refine", action="append", default=[], metavar="STAGE")
    ap.add_argument("--cycles", type=int, default=10)
    args = ap.parse_args()
    if not args.data and not args.simulate:
        ap.error("--data か --simulate のどちらかが必要です")
    if args.data and len(args.data) != len(args.instprm):
        ap.error("--data と --instprm は同じ数だけ指定してください")
    if args.simulate and len(args.instprm) != 1:
        ap.error("--simulate には --instprm を 1 つ指定してください")

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
    hists = []
    if args.data:
        for d, ip in zip(args.data, args.instprm, strict=True):
            hists.append(gpx.add_powder_histogram(os.path.abspath(d), os.path.abspath(ip)))
    else:
        tmin, tmax, tstep = args.simulate
        hists.append(
            gpx.add_simulated_powder_histogram(
                "sim", os.path.abspath(args.instprm[0]), tmin, tmax, Tstep=tstep
            )
        )
    phase = gpx.add_phase(os.path.abspath(args.cif), phasename=args.phase_name, histograms=hists)

    # ---- パラメータの上書き(全ヒストグラムに同じ値) --------------------
    for hist in hists:
        inst_params = hist.data["Instrument Parameters"][0]
        for item in args.set_inst:
            key, val = item.split("=", 1)
            if key not in inst_params:
                raise SystemExit(f"装置パラメータ {key!r} がありません: {list(inst_params)}")
            inst_params[key][1] = float(val)
        sample_params = hist.data["Sample Parameters"]
        for item in args.set_sample:
            key, val = item.split("=", 1)
            if key not in sample_params:
                raise SystemExit(f"試料パラメータ {key!r} がありません: {list(sample_params)}")
            if isinstance(sample_params[key], list):
                sample_params[key][0] = float(val)
            else:
                sample_params[key] = float(val)
        hap = phase.data["Histograms"][hist.name]
        if args.md:
            hap["Pref.Ori."][1] = float(args.md[0])
            hap["Pref.Ori."][3] = [int(v) for v in args.md[1:]]
        if args.size is not None:
            hap["Size"][1][0] = float(args.size)
        if args.mustrain is not None:
            hap["Mustrain"][1][0] = float(args.mustrain)

    stages_log = []
    if args.refine:
        atom_flags: set = set()
        for stage in args.refine:
            apply_stage(gpx, hists, phase, stage, atom_flags)
            gpx.set_Controls("cycles", args.cycles)
            gpx.refine()
            rv = gpx.data["Covariance"]["data"].get("Rvals", {})
            stages_log.append({"stage": stage, "Rvals": to_jsonable(rv)})
            print(f"stage {stage!r}: Rwp = {rv.get('Rwp')}  GOF = {rv.get('GOF')}", flush=True)
    else:
        # 0 サイクル: 計算パターンと反射リストを得るだけ。何か 1 つ精密化フラグが必要。
        hists[0].set_refinements({"Sample Parameters": ["Scale"]})
        gpx.set_Controls("cycles", 0)
        gpx.refine()

    out: dict = {"gsas2_dir": g2dir, "case": vars(args), "stages": stages_log}
    try:
        from GSASII import GSASIIpath

        out["gsas2_version"] = to_jsonable(GSASIIpath.GetVersionNumber())
    except Exception as exc:
        out["gsas2_version"] = repr(exc)

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
                "refine": row[ct + 1],
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
    }

    # ---- ヒストグラムごと ----------------------------------------------
    hist_out = [export_histogram(h, phase, general, G2el, atmdata) for h in hists]
    out["histograms"] = hist_out
    out.update({k: v for k, v in hist_out[0].items() if k != "name"})

    # ---- 共分散・R 因子 ------------------------------------------------
    cov = gpx.data.get("Covariance", {}).get("data", {})
    vary = list(cov.get("varyList", []))
    sig = to_jsonable(cov.get("sig", []))
    parm = cov.get("parmDict", {})
    out["refinement"] = {
        "varyList": vary,
        "values": {v: to_jsonable(parm.get(v)) for v in vary},
        "sig": dict(zip(vary, sig, strict=True)) if len(sig) == len(vary) else sig,
        "Rvals": to_jsonable(cov.get("Rvals", {})),
        "covMatrix": to_jsonable(cov.get("covMatrix", [])),
        "newCellDict": to_jsonable(cov.get("newCellDict", {})),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(hists)} histogram(s), {len(vary)} varied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
