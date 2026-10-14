"""プロジェクトファイル(JSON)の保存と読み込み。

スキーマ(schema_version 2。1 との違いは pdf_datasets の追加):
    {
      "schema_version": 1, "name": ..., "program": "kaisetsu", "version": ...,
      "phases": [{name, le_bail, structure: {...}, f2_tables: {hist_index: {"h k l": F2}}}],
      "histograms": [{name, data: {x, y, weight, name, metadata}, instrument, sample, background,
                      contributions: {phase: {...}}, x_min, x_max, excluded, weight_factor,
                      kalpha2_mode, d_min, d_min_margin_deg}],
      "pdf_datasets": [{name, data: {r, g, dg, name, metadata}, radiation, qmax, qmin, qdamp,
                        qbroad, scale, r_min, r_max, weight_factor, dr_fine,
                        phases: {phase: {scale, delta1, delta2, sratio, rcut, spdiameter,
                                         stepcut}}}],
      "params": {name: {value, vary, min, max, step, esd, tie}},
      "log": [...]
    }
データ配列も JSON に含める(数万点なら数百 KB)。`save_project(..., external_arrays=True)` なら
配列を同名の .npz に分けて保存し、JSON には {"$npz": キー} を置く(大きなデータ向け)。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from kaisetsu import __version__
from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.crystal.structure import Structure
from kaisetsu.pattern.calculator import PhaseContribution
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.pdf.data import PDFData
from kaisetsu.profile.background import (
    BackgroundModel,
    ChebyshevBackground,
    GaussianHump,
    InterpolatedBackground,
)
from kaisetsu.refine.project import HistogramEntry, Project
from kaisetsu.scattering.radiation import Radiation, RadiationType

SCHEMA_VERSION = 2


def _f(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else float(v)


def structure_to_dict(st: Structure) -> dict[str, Any]:
    return {
        "name": st.name,
        "lattice": [float(v) for v in st.lattice.parameters],
        "spacegroup": {
            "symbol": st.spacegroup.symbol,
            "number": st.spacegroup.number,
            "symops": [op.triplet for op in st.spacegroup.operations],
        },
        "sites": [
            {
                "label": s.label,
                "species": s.species,
                "position": [float(v) for v in s.position],
                "occupancy": float(s.occupancy),
                "u_iso": float(s.u_iso),
                "u_aniso": None if s.u_aniso is None else [float(v) for v in s.u_aniso],
            }
            for s in st.sites
        ],
    }


def structure_from_dict(d: dict[str, Any]) -> Structure:
    lat = Lattice.from_parameters(d["lattice"])
    sgd = d["spacegroup"]
    sg = (
        SpaceGroup.from_symops(sgd["symops"])
        if sgd.get("symops")
        else SpaceGroup.from_symbol(sgd["symbol"])
    )
    sites = [
        AtomSite(
            s["label"],
            s["species"],
            np.asarray(s["position"], dtype=float),
            s.get("occupancy", 1.0),
            s.get("u_iso", 0.0),
            None if s.get("u_aniso") is None else np.asarray(s["u_aniso"], dtype=float),
        )
        for s in d["sites"]
    ]
    return Structure(lat, sg, sites, name=d.get("name", ""))


def _instrument_to_dict(inst: CWInstrument | TOFInstrument) -> dict[str, Any]:
    if isinstance(inst, TOFInstrument):
        d = asdict(inst)
        d["kind"] = "tof"
        return d
    d = asdict(inst)
    d["radiation"] = {
        "kind": inst.radiation.kind.value,
        "wavelength": inst.radiation.wavelength,
        "wavelength2": inst.radiation.wavelength2,
        "ratio2": inst.radiation.ratio2,
        "wavelength3": inst.radiation.wavelength3,
        "ratio3": inst.radiation.ratio3,
    }
    return d


def _instrument_from_dict(d: dict[str, Any]) -> CWInstrument | TOFInstrument:
    if d.get("kind") == "tof":
        # 無くなったフィールド(v0.1.1 までの zr_scale0, zr_scale1 など)は読み飛ばす
        known = {f.name for f in fields(TOFInstrument)}
        kw = {k: v for k, v in d.items() if k in known}
        return TOFInstrument(**kw)
    r = d["radiation"]
    rad = Radiation(
        RadiationType(r["kind"]),
        r["wavelength"],
        r.get("wavelength2"),
        r.get("ratio2", 0.0),
        r.get("wavelength3"),
        r.get("ratio3", 0.0),
    )
    kw = {k: v for k, v in d.items() if k != "radiation"}
    return CWInstrument(radiation=rad, **kw)


def _sample_to_dict(s: SampleGeometry) -> dict[str, Any]:
    d = asdict(s)
    d["kind"] = s.kind.value
    return d


def _sample_from_dict(d: dict[str, Any]) -> SampleGeometry:
    kw = dict(d)
    kw["kind"] = GeometryKind(kw.get("kind", "bragg-brentano"))
    if "mu_eff_inv_mm" in kw:  # 旧形式(有効線吸収係数)→ 侵入深さ 1/μ_eff [mm]
        mu = float(kw.pop("mu_eff_inv_mm"))
        kw["transparency_mm"] = 1.0 / mu if mu > 0 else 0.0
    smp = SampleGeometry(**kw)
    if smp.roughness_a == 0.0 and smp.roughness_b == 0.0:  # 旧形式の「補正なし」
        smp.roughness_a = smp.roughness_b = 1.0
    return smp


def _background_to_dict(bk: BackgroundModel) -> dict[str, Any]:
    d: dict[str, Any] = {
        "kind": getattr(bk, "kind", "chebyshev"),
        "coefficients": bk.coefficients.tolist(),
        "x_min": bk.x_min,
        "x_max": bk.x_max,
    }
    if isinstance(bk, InterpolatedBackground):
        d["points_x"] = bk.points_x.tolist()
    return d


def _background_from_dict(d: dict[str, Any]) -> BackgroundModel:
    coeffs = np.asarray(d["coefficients"], dtype=float)
    if d.get("kind") == "points":
        return InterpolatedBackground(
            np.asarray(d["points_x"], dtype=float), coeffs, d["x_min"], d["x_max"]
        )
    return ChebyshevBackground(coeffs, d["x_min"], d["x_max"])


def _contribution_to_dict(pc: PhaseContribution) -> dict[str, Any]:
    return {
        "scale": pc.scale,
        "march_dollase_r": pc.march_dollase_r,
        "march_dollase_axis": list(pc.march_dollase_axis),
        "size_um": pc.size_um,
        "strain_e6": pc.strain_e6,
        "size_lorentz_fraction": pc.size_lorentz_fraction,
        "strain_lorentz_fraction": pc.strain_lorentz_fraction,
        "stephens": None if pc.stephens is None else [float(v) for v in pc.stephens],
        "size_axis": list(pc.size_axis),
        "size_axial_um": pc.size_axial_um,
        "sh_order": int(pc.sh_order),
        "extinction": float(pc.extinction),
        "sh_coefficients": (
            None if pc.sh_coefficients is None else [float(v) for v in pc.sh_coefficients]
        ),
    }


def project_to_dict(proj: Project) -> dict[str, Any]:
    phases = []
    for ph in proj.phases.values():
        ph.sync()
        phases.append(
            {
                "name": ph.name,
                "le_bail": ph.le_bail,
                "f2_update": ph.f2_update,
                "pawley": ph.pawley,
                "brindley_radius_um": ph.brindley_radius_um,
                "structure": structure_to_dict(ph.structure),
                "f2_tables": {
                    str(i): {" ".join(str(v) for v in k): val for k, val in tbl.items()}
                    for i, tbl in ph.f2_tables.items()
                },
            }
        )
    hists = []
    for h in proj.histograms:
        assert h.data.weight is not None
        hists.append(
            {
                "name": h.name,
                "data": {
                    "name": h.data.name,
                    "x": h.data.x.tolist(),
                    "y": h.data.y.tolist(),
                    "weight": h.data.weight.tolist(),
                    "metadata": {
                        k: v
                        for k, v in h.data.metadata.items()
                        if isinstance(v, str | int | float | bool)
                    },
                },
                "instrument": _instrument_to_dict(h.instrument),
                "sample": _sample_to_dict(h.sample),
                "background": _background_to_dict(h.background),
                "humps": [
                    {"center": hm.center, "fwhm": hm.fwhm, "height": hm.height} for hm in h.humps
                ],
                "contributions": {
                    n: _contribution_to_dict(pc) for n, pc in h.contributions.items()
                },
                "x_min": h.x_min,
                "x_max": h.x_max,
                "excluded": [list(e) for e in h.excluded],
                "weight_factor": h.weight_factor,
                "kalpha2_mode": h.kalpha2_mode,
                "fcj_method": h.fcj_method,
                "d_min": h.d_min,
                "d_min_margin_deg": h.d_min_margin_deg,
            }
        )
    pdfs = []
    for g in proj.pdf_datasets:
        pdfs.append(
            {
                "name": g.name,
                "data": {
                    "name": g.data.name,
                    "r": g.data.r.tolist(),
                    "g": g.data.g.tolist(),
                    "dg": None if g.data.dg is None else g.data.dg.tolist(),
                    "metadata": {
                        k: v
                        for k, v in g.data.metadata.items()
                        if isinstance(v, str | int | float | bool)
                    },
                },
                "radiation": g.radiation,
                "qmax": g.qmax,
                "qmin": g.qmin,
                "qdamp": g.qdamp,
                "qbroad": g.qbroad,
                "scale": g.scale,
                "r_min": g.r_min,
                "r_max": g.r_max,
                "weight_factor": g.weight_factor,
                "dr_fine": g.dr_fine,
                "phases": {
                    n: {
                        "scale": pp.scale,
                        "delta1": pp.delta1,
                        "delta2": pp.delta2,
                        "sratio": pp.sratio,
                        "rcut": pp.rcut,
                        "spdiameter": pp.spdiameter,
                        "stepcut": pp.stepcut,
                    }
                    for n, pp in g.phases.items()
                },
            }
        )
    params = {}
    for p in proj.params:
        params[p.name] = {
            "value": p.value,
            "vary": p.vary,
            "min": _f(p.min),
            "max": _f(p.max),
            "step": p.step,
            "esd": p.esd,
            "initial": p.initial,
            "tie": None
            if p.tie is None
            else {
                "masters": list(p.tie.masters),
                "coeffs": list(p.tie.coeffs),
                "offset": p.tie.offset,
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "program": "kaisetsu",
        "version": __version__,
        "name": proj.name,
        "phases": phases,
        "histograms": hists,
        "pdf_datasets": pdfs,
        "params": params,
        "penalize_negative_widths": bool(proj.penalize_negative_widths),
        "log": list(proj.log),
    }


def project_from_dict(d: dict[str, Any]) -> Project:
    if d.get("schema_version", 1) > SCHEMA_VERSION:
        raise ValueError(
            f"このプロジェクトファイルは新しい形式(schema {d['schema_version']})です。"
            "kaisetsu を更新してください"
        )
    proj = Project(d.get("name", ""))
    for phd in d["phases"]:
        entry = proj.add_phase(
            structure_from_dict(phd["structure"]),
            name=phd["name"],
            le_bail=phd.get("le_bail", False),
        )
        entry.f2_update = bool(phd.get("f2_update", True))
        entry.pawley = bool(phd.get("pawley", False))
        entry.brindley_radius_um = float(phd.get("brindley_radius_um", 0.0))
        for i, tbl in phd.get("f2_tables", {}).items():
            entry.f2_tables[int(i)] = {
                tuple(int(v) for v in k.split()): float(val)  # type: ignore[misc]
                for k, val in tbl.items()
            }
    for hd in d["histograms"]:
        dd = hd["data"]
        data = PowderData(
            np.asarray(dd["x"], dtype=float),
            np.asarray(dd["y"], dtype=float),
            np.asarray(dd["weight"], dtype=float),
            name=dd.get("name", ""),
            metadata=dict(dd.get("metadata", {})),
        )
        bk = hd["background"]
        h: HistogramEntry = proj.add_histogram(
            data,
            _instrument_from_dict(hd["instrument"]),
            _sample_from_dict(hd["sample"]),
            _background_from_dict(bk),
            name=hd["name"],
            phases=list(hd["contributions"]),
            x_min=hd["x_min"],
            x_max=hd["x_max"],
            excluded=[(float(a), float(b)) for a, b in hd.get("excluded", [])],
            kalpha2_mode=hd.get("kalpha2_mode", "exact"),
            weight_factor=hd.get("weight_factor", 1.0),
        )
        h.d_min = hd.get("d_min")
        h.d_min_margin_deg = hd.get("d_min_margin_deg", 2.0)
        for hm in hd.get("humps", []):
            h.humps.append(
                GaussianHump(float(hm["center"]), float(hm["fwhm"]), float(hm["height"]))
            )
        h.fcj_method = hd.get("fcj_method", "accurate")
        for n, cd in hd["contributions"].items():
            pc = h.contributions[n]
            pc.scale = cd["scale"]
            pc.march_dollase_r = cd["march_dollase_r"]
            pc.march_dollase_axis = tuple(int(v) for v in cd["march_dollase_axis"])  # type: ignore[assignment]
            pc.size_um = cd.get("size_um")
            pc.strain_e6 = cd.get("strain_e6", 0.0)
            pc.size_lorentz_fraction = cd.get("size_lorentz_fraction", 1.0)
            pc.strain_lorentz_fraction = cd.get("strain_lorentz_fraction", 1.0)
            if cd.get("stephens") is not None:
                pc.stephens = np.asarray(cd["stephens"], dtype=float)
            if cd.get("size_axis") is not None:
                pc.size_axis = tuple(int(v) for v in cd["size_axis"])  # type: ignore[assignment]
            pc.size_axial_um = cd.get("size_axial_um")
            pc.sh_order = int(cd.get("sh_order", 0))
            pc.extinction = float(cd.get("extinction", 0.0))
            if cd.get("sh_coefficients") is not None:
                pc.sh_coefficients = np.asarray(cd["sh_coefficients"], dtype=float)
    for gd in d.get("pdf_datasets", []):
        dd = gd["data"]
        pdata = PDFData(
            np.asarray(dd["r"], dtype=float),
            np.asarray(dd["g"], dtype=float),
            None if dd.get("dg") is None else np.asarray(dd["dg"], dtype=float),
            name=dd.get("name", ""),
            metadata=dict(dd.get("metadata", {})),
        )
        g = proj.add_pdf(
            pdata,
            gd.get("radiation", "xray"),
            qmax=gd.get("qmax", 0.0),
            qmin=gd.get("qmin", 0.0),
            qdamp=gd.get("qdamp", 0.0),
            qbroad=gd.get("qbroad", 0.0),
            name=gd.get("name"),
            phases=list(gd.get("phases", {})),
            r_min=gd.get("r_min"),
            r_max=gd.get("r_max"),
            weight_factor=gd.get("weight_factor", 1.0),
        )
        g.scale = gd.get("scale", 1.0)
        g.dr_fine = gd.get("dr_fine", 0.01)
        for n, pd in gd.get("phases", {}).items():
            pp = g.phases[n]
            for key in ("scale", "delta1", "delta2", "sratio", "rcut", "spdiameter", "stepcut"):
                if key in pd:
                    setattr(pp, key, float(pd[key]))
    proj.build_parameters()
    for name, pd in _migrate_param_names(dict(d.get("params", {}))).items():
        if name not in proj.params:
            continue
        p = proj.params[name]
        if pd.get("value") is not None:
            p.value = float(pd["value"])
        if "min" in pd:
            p.min = -math.inf if pd["min"] is None else float(pd["min"])
        if "max" in pd:
            p.max = math.inf if pd["max"] is None else float(pd["max"])
        p.step = float(pd.get("step", p.step))
        p.esd = pd.get("esd")
        init = pd.get("initial")
        p.initial = float(init) if init is not None else p.value
        if p.tie is None and not p.locked:
            p.vary = bool(pd.get("vary", False))
        t = pd.get("tie")
        if t and p.tie is None and all(m in proj.params for m in t["masters"]):
            proj.params.tie(name, t["masters"], t["coeffs"], t.get("offset", 0.0))
    proj.params.evaluate_ties()
    proj.params.apply_all()
    proj.log = list(d.get("log", []))
    proj.penalize_negative_widths = bool(d.get("penalize_negative_widths", False))
    return proj


def _migrate_param_names(params: dict[str, Any]) -> dict[str, Any]:
    """旧バージョンのパラメータ名を今の名前に直す(h*:inst:s_l, h_l → sh_l)。

    値と上下限は装置の辞書(s_l, h_l)と現在の既定から決まるので、vary だけを引き継ぐ。
    """
    out: dict[str, Any] = {}
    for name, pd in params.items():
        if name.endswith((":inst:s_l", ":inst:h_l")):
            new = name.rsplit(":", 1)[0] + ":sh_l"
            if new in params:
                continue
            cur = out.setdefault(new, {"value": None, "vary": False})
            cur["vary"] = bool(cur.get("vary")) or bool(pd.get("vary", False))
            continue
        out[name] = pd
    return out


_ARRAY_KEYS = ("x", "y", "weight", "r", "g", "dg")


def _npz_path(p: Path) -> Path:
    name = p.name
    for suf in (".kaisetsu.json", ".json"):
        if name.endswith(suf):
            return p.with_name(name[: -len(suf)] + ".npz")
    return p.with_suffix(".npz")


def _externalize(d: dict[str, Any]) -> dict[str, Any]:
    """ヒストグラムと PDF の配列を取り出し、JSON 側に参照キーを置く。"""
    arrays: dict[str, Any] = {}
    for prefix, items in (("h", d.get("histograms", [])), ("g", d.get("pdf_datasets", []))):
        for i, item in enumerate(items):
            data = item["data"]
            for key in _ARRAY_KEYS:
                if key in data and isinstance(data[key], list):
                    name = f"{prefix}{i}_{key}"
                    arrays[name] = np.asarray(data[key], dtype=float)
                    data[key] = {"$npz": name}
    return arrays


def _internalize(d: dict[str, Any], npz: Path | None) -> None:
    loaded: Any = None
    for items in (d.get("histograms", []), d.get("pdf_datasets", [])):
        for item in items:
            data = item["data"]
            for key in _ARRAY_KEYS:
                v = data.get(key)
                if isinstance(v, dict) and "$npz" in v:
                    if loaded is None:
                        if npz is None or not npz.exists():
                            raise FileNotFoundError(
                                f"プロジェクトの配列ファイル {npz} が見つかりません"
                            )
                        loaded = np.load(npz)
                    data[key] = loaded[v["$npz"]].tolist()


def save_project(proj: Project, path: str | Path, *, external_arrays: bool = False) -> Path:
    p = Path(path)
    d = project_to_dict(proj)
    if external_arrays:
        arrays = _externalize(d)
        np.savez_compressed(_npz_path(p), **arrays)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_project(path: str | Path) -> Project:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"プロジェクトファイルが見つかりません: {p}")
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    _internalize(d, _npz_path(p))
    return project_from_dict(d)
