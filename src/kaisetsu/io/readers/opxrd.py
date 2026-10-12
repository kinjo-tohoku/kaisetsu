"""opXRD(Open Powder X-ray Diffraction Database、Hollarek ら 2025、CC BY 4.0)の JSON。

1 ファイル 1 パターン: {"two_theta_values": [...], "intensities": [...], "label": "<JSON 文字列>",
"metadata": "<JSON 文字列>"}。label は xrdpattern ライブラリの PowderExperiment を入れ子の
JSON 文字列で書いたもので、phases(各相も JSON 文字列: lattice "(a, b, c, α, β, γ)"、basis =
全原子の JSON 文字列のリスト {x, y, z, occupancy, symbol}、spacegroup(たいてい null))、
xray_info {primary_wavelength, secondary_wavelength}、is_simulated、temp_K が入る。
構造ラベルは空間群なしの全原子セルなので、Rietveld 法に使うときは
`kaisetsu.crystal.symmetry.find_symmetry` で対称性を見つけて非対称単位に戻す。

出典: D. Hollarek et al., "opXRD: Open Experimental Powder X-ray Diffraction Database",
      arXiv:2503.05577 (2025)、Zenodo 10.5281/zenodo.15298026。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.crystal.structure import Structure
from kaisetsu.pattern.histogram import PowderData


@dataclass
class OpxrdPattern:
    """opXRD の 1 パターン(データ + ラベル)。"""

    data: PowderData
    #: ラベルの相(空間群が無ければ P1 の全原子セル)
    structures: list[Structure] = field(default_factory=list)
    wavelength: float | None = None
    wavelength2: float | None = None
    is_simulated: bool = False
    temperature_k: float | None = None
    institution: str = ""
    contributor: str = ""
    original_format: str = ""


def is_opxrd(head: str) -> bool:
    # 先頭 4000 文字しか無いので intensities は見ない(two_theta が数万点あると後ろに来る)
    return head.lstrip().startswith("{") and '"two_theta_values"' in head[:200]


def _loads(v: Any) -> Any:
    """入れ子の JSON 文字列をほどく(文字列でなければそのまま)。"""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ("{", "[") or s in ("null", "true", "false"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return v
    return v


def _float(v: Any) -> float | None:
    if v is None or v == "null" or v == "nan":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _structure_from_phase(ph: dict[str, Any], name: str) -> Structure | None:
    lat = ph.get("lattice")
    if not lat:
        return None
    nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", str(lat))]
    if len(nums) < 6:
        return None
    lattice = Lattice(*nums[:6])
    basis = _loads(ph.get("basis"))
    sites: list[AtomSite] = []
    counts: dict[str, int] = {}
    if isinstance(basis, list):
        for raw in basis:
            s = _loads(raw)
            if not isinstance(s, dict):
                continue
            sym = str(s.get("symbol") or s.get("species") or "")
            sym = re.sub(r"0\+$|0-$", "", sym)  # 'Ba0+' → 'Ba'
            if not sym:
                continue
            el = AtomSite("x", sym, np.zeros(3)).element
            counts[el] = counts.get(el, 0) + 1
            occ = _float(s.get("occupancy"))
            sites.append(
                AtomSite(
                    f"{el}{counts[el]}",
                    sym,
                    np.array([float(s["x"]), float(s["y"]), float(s["z"])]) % 1.0,
                    occupancy=min(max(occ if occ is not None else 1.0, 0.0), 1.0),
                    u_iso=0.01,
                )
            )
    if not sites:
        return None
    spg = ph.get("spacegroup")
    # basis は常に全原子セル(pymatgen Structure 由来)なので、空間群番号があっても P1 で持ち、
    # 対称性は find_symmetry で求める
    _ = spg
    return Structure(lattice, SpaceGroup.from_number(1), sites, name=name)


def read_opxrd(path: str | Path) -> OpxrdPattern:
    """opXRD の JSON を読む。"""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    x = np.asarray(raw["two_theta_values"], dtype=np.float64)
    y = np.asarray(raw["intensities"], dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    label = _loads(raw.get("label")) or {}
    meta = _loads(raw.get("metadata")) or {}
    xi = _loads(label.get("xray_info")) or {}
    lam1 = _float(xi.get("primary_wavelength"))
    lam2 = _float(xi.get("secondary_wavelength"))
    structures: list[Structure] = []
    for k, ph_raw in enumerate(label.get("phases") or []):
        ph = _loads(ph_raw)
        if isinstance(ph, dict):
            st = _structure_from_phase(ph, name=f"{p.stem}_phase{k + 1}" if k else p.stem)
            if st is not None:
                structures.append(st)
    md: dict[str, object] = {
        "format": "opXRD",
        "institution": meta.get("institution") or "",
        "contributor": meta.get("contributor_name") or "",
        "original_format": meta.get("original_file_format") or "",
        "measurement_date": meta.get("measurement_date") or "",
    }
    if lam1:
        md["wavelength"] = lam1
    if lam2:
        md["wavelength2"] = lam2
    data = PowderData(x, y, name=p.stem, metadata=md)
    sim = label.get("is_simulated")
    return OpxrdPattern(
        data=data,
        structures=structures,
        wavelength=lam1,
        wavelength2=lam2,
        is_simulated=bool(sim) if isinstance(sim, bool) else str(sim).lower() == "true",
        temperature_k=_float(label.get("temp_K")),
        institution=str(md["institution"]),
        contributor=str(md["contributor"]),
        original_format=str(md["original_format"]),
    )


__all__ = ["OpxrdPattern", "is_opxrd", "read_opxrd"]
