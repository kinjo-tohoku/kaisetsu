"""P1(全原子を書き下した単位胞)の構造から対称性を見つけ、標準設定の非対称単位に戻す。

opXRD のラベル、DFT の出力、VESTA の P1 書き出しなどは空間群が無く全原子が並んでいる。
そのまま Rietveld 法に使うと原子座標・変位パラメータの数が多すぎて精密化が成り立たないので、
spglib で空間群を求め、標準設定(ITA の標準。単斜晶は b 軸ユニーク・セル選択 1、三方晶は
六方軸)の単位胞に変換し、対称等価な原子を 1 つにまとめる。

出典: A. Togo, K. Shinohara & I. Tanaka, "Spglib: a software library for crystal symmetry search",
      Sci. Technol. Adv. Mater. Meth. 4, 2384822 (2024)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gemmi
import numpy as np
import spglib
import spglib.error

from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.crystal.structure import Structure

spglib.error.OLD_ERROR_HANDLING = False  # 失敗は None でなく SpglibError で受ける


def _u_equiv(site: AtomSite, structure: Structure) -> float:
    """等価等方変位パラメータ U_eq = (1/3) Σ_ij U^ij a*_i a*_j (a_i · a_j)。

    等方性サイトは U_iso をそのまま返す。
    出典: Fischer & Tillmanns, Acta Cryst. C44, 775-776 (1988)。
    """
    if site.u_aniso is None:
        return float(site.u_iso)
    rec = structure.lattice.reciprocal
    a_star = np.array([rec.a, rec.b, rec.c])
    g = structure.lattice.metric_tensor
    return float(np.sum(site.u_matrix() * np.outer(a_star, a_star) * g) / 3.0)


@dataclass(frozen=True)
class SymmetryResult:
    """`find_symmetry` の結果。"""

    structure: Structure
    number: int
    hm: str
    hall: str
    n_atoms_in_cell: int
    n_sites: int

    def summary(self) -> str:
        return (
            f"空間群 {self.hm}(No. {self.number})、単位胞の {self.n_atoms_in_cell} 原子 → "
            f"独立サイト {self.n_sites} 個"
        )


def _lattice_from_matrix(m: np.ndarray) -> Lattice:
    a, b, c = (float(np.linalg.norm(v)) for v in m)

    def ang(u: np.ndarray, v: np.ndarray) -> float:
        return math.degrees(
            math.acos(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
        )

    return Lattice(a, b, c, ang(m[1], m[2]), ang(m[0], m[2]), ang(m[0], m[1]))


def find_symmetry(
    structure: Structure,
    symprec: float = 1e-3,
    angle_tolerance: float = -1.0,
    name: str | None = None,
) -> SymmetryResult:
    """構造の対称性を見つけ、標準設定の格子と非対称単位で表した Structure を返す。

    元の構造に空間群があればまず全原子に展開してから調べる(P1 でなくてもよい)。
    散乱種・占有率・U_iso が違う原子は別の種類として扱う(同じ元素でも混ざらない)。
    異方性 U_ij は標準設定への変換で向きが変わるので持ち越さず、等価等方 U_eq
    (= (1/3) Σ_ij U^ij a*_i a*_j a_i·a_j)に落とす。

    Args:
        structure: 入力構造(たいてい P1 の全原子)
        symprec: 座標の許容誤差 [Å 相当の分率]。精密化済みの構造なら 1e-3、粗い構造なら 1e-2
        angle_tolerance: 角度の許容誤差 [deg]。負なら spglib の既定
        name: 返す構造の名前(None なら元の名前)
    """
    exp = structure.expand()
    lat = structure.lattice.orthogonalization_matrix.T  # 行が a, b, c
    kinds: list[tuple[str, float, float]] = []
    types: list[int] = []
    for k in range(len(exp)):
        site = structure.sites[int(exp.site_index[k])]
        key = (site.species, round(site.occupancy, 6), round(_u_equiv(site, structure), 6))
        if key not in kinds:
            kinds.append(key)
        types.append(kinds.index(key) + 1)
    cell = (
        np.asarray(lat, dtype=float).tolist(),
        np.asarray(exp.positions, dtype=float).tolist(),
        types,
    )
    try:
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    except spglib.SpglibError:
        ds = None
    if ds is None:
        raise ValueError("対称性を見つけられませんでした(spglib)。symprec を大きくしてみてください")
    std_lat = np.asarray(ds.std_lattice, dtype=float)
    std_pos = np.asarray(ds.std_positions, dtype=float)
    std_types = [int(t) for t in ds.std_types]
    std = (std_lat.tolist(), std_pos.tolist(), std_types)
    try:
        ds2 = spglib.get_symmetry_dataset(std, symprec=symprec, angle_tolerance=angle_tolerance)
    except spglib.SpglibError:
        ds2 = None
    if ds2 is None:
        raise ValueError("標準設定での対称性の判定に失敗しました(spglib)")
    ops = gemmi.symops_from_hall(ds2.hall)
    sg = SpaceGroup(ops, gemmi.find_spacegroup_by_ops(ops))
    lattice = _lattice_from_matrix(std_lat)
    reps = sorted(set(int(v) for v in ds2.equivalent_atoms))
    sites: list[AtomSite] = []
    counts: dict[str, int] = {}
    for i in reps:
        species, occ, uiso = kinds[std_types[i] - 1]
        el = AtomSite("x", species, np.zeros(3)).element
        counts[el] = counts.get(el, 0) + 1
        sites.append(
            AtomSite(f"{el}{counts[el]}", species, std_pos[i] % 1.0, occupancy=occ, u_iso=uiso)
        )
    out = Structure(lattice, sg, sites, name=structure.name if name is None else name)
    return SymmetryResult(
        structure=out,
        number=int(ds2.number),
        hm=str(ds2.international),
        hall=str(ds2.hall),
        n_atoms_in_cell=len(exp),
        n_sites=len(sites),
    )


__all__ = ["SymmetryResult", "find_symmetry"]
