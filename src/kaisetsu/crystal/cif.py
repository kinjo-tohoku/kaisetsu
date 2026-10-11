"""CIF の読み書き(gemmi を利用)。

読み込みの方針:
    - 対称操作は CIF に明示されていればそれを優先する(設定・原点の取り違えを防ぐ)。
      無ければ H-M 記号、Hall 記号、国際番号の順に解釈する。
    - ADP は U_iso / U_ij を使う。B_iso しか無い場合は gemmi が U に換算する。
"""

from __future__ import annotations

import math
from pathlib import Path

import gemmi
import numpy as np

from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.crystal.structure import Structure


def read_structure(path: str | Path, block: str | None = None) -> Structure:
    """CIF ファイルから結晶構造を読む。

    Args:
        path: CIF ファイル
        block: 読み込むデータブロック名。None なら最初のブロック。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIF ファイルが見つかりません: {path}")
    doc = gemmi.cif.read(str(path))
    cif_block = doc[0] if block is None else doc.find_block(block)
    if cif_block is None:
        raise ValueError(f"CIF にブロック {block!r} がありません")
    st = gemmi.make_small_structure_from_block(cif_block)

    cell = st.cell
    if cell.a <= 0 or cell.b <= 0 or cell.c <= 0:
        raise ValueError(f"CIF {path.name}: 格子定数が読めません")
    lattice = Lattice(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)

    spacegroup = _spacegroup_from_small_structure(st, path.name)

    sites: list[AtomSite] = []
    for s in st.sites:
        species = s.type_symbol.strip() if s.type_symbol.strip() else s.element.name
        if species.lower() in ("", "x", "?"):
            species = s.element.name
        aniso = None
        if s.aniso.nonzero():
            aniso = np.array(
                [s.aniso.u11, s.aniso.u22, s.aniso.u33, s.aniso.u12, s.aniso.u13, s.aniso.u23]
            )
        occ = float(s.occ) if s.occ > 0 else 1.0
        sites.append(
            AtomSite(
                label=s.label,
                species=species,
                position=np.array([s.fract.x, s.fract.y, s.fract.z]),
                occupancy=min(occ, 1.0),
                u_iso=float(s.u_iso) if math.isfinite(s.u_iso) else 0.0,
                u_aniso=aniso,
            )
        )
    if not sites:
        raise ValueError(f"CIF {path.name}: 原子サイトがありません")

    name = cif_block.name
    for tag in ("_pd_phase_name", "_chemical_name_mineral", "_chemical_name_common"):
        v = cif_block.find_value(tag)
        if v:
            name = gemmi.cif.as_string(v)
            break
    return Structure(lattice, spacegroup, sites, name=name)


def _spacegroup_from_small_structure(st: gemmi.SmallStructure, fname: str) -> SpaceGroup:
    if st.symops:
        return SpaceGroup.from_symops(st.symops)
    sg: gemmi.SpaceGroup | None = getattr(st, "spacegroup", None)
    if sg is not None:
        return SpaceGroup.from_gemmi(sg)
    if st.spacegroup_hm:
        return SpaceGroup.from_symbol(st.spacegroup_hm)
    if st.spacegroup_hall:
        sg = gemmi.find_spacegroup_by_name(st.spacegroup_hall)
        if sg is not None:
            return SpaceGroup.from_gemmi(sg)
    if st.spacegroup_number:
        return SpaceGroup.from_number(int(st.spacegroup_number))
    raise ValueError(f"CIF {fname}: 空間群の情報(対称操作・H-M 記号・番号)がありません")


def write_structure(structure: Structure, path: str | Path) -> None:
    """結晶構造を簡易な CIF に書き出す(対称操作・原子・ADP を含む)。"""
    lat = structure.lattice
    sg = structure.spacegroup
    lines = [
        f"data_{_safe_name(structure.name) or 'phase'}",
        f"_pd_phase_name '{structure.name}'",
        f"_cell_length_a {lat.a:.6f}",
        f"_cell_length_b {lat.b:.6f}",
        f"_cell_length_c {lat.c:.6f}",
        f"_cell_angle_alpha {lat.alpha:.5f}",
        f"_cell_angle_beta {lat.beta:.5f}",
        f"_cell_angle_gamma {lat.gamma:.5f}",
        f"_cell_volume {lat.volume:.4f}",
        f"_space_group_IT_number {sg.number}",
        f"_space_group_name_H-M_alt '{sg.hm}'",
        "loop_",
        "_space_group_symop_id",
        "_space_group_symop_operation_xyz",
    ]
    for i, op in enumerate(sg.operations, start=1):
        lines.append(f"{i} '{op.triplet}'")
    lines += [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_adp_type",
        "_atom_site_U_iso_or_equiv",
        "_atom_site_symmetry_multiplicity",
    ]
    mults = structure.multiplicities()
    for s, m in zip(structure.sites, mults, strict=True):
        x, y, z = s.position
        adp = "Uani" if s.is_anisotropic else "Uiso"
        u = _u_equiv(s) if s.is_anisotropic else s.u_iso
        lines.append(
            f"{s.label} {s.species} {x:.6f} {y:.6f} {z:.6f} {s.occupancy:.4f} {adp} {u:.5f} {m}"
        )
    aniso_sites = [s for s in structure.sites if s.is_anisotropic]
    if aniso_sites:
        lines += [
            "loop_",
            "_atom_site_aniso_label",
            "_atom_site_aniso_U_11",
            "_atom_site_aniso_U_22",
            "_atom_site_aniso_U_33",
            "_atom_site_aniso_U_12",
            "_atom_site_aniso_U_13",
            "_atom_site_aniso_U_23",
        ]
        for s in aniso_sites:
            assert s.u_aniso is not None
            lines.append(s.label + " " + " ".join(f"{v:.5f}" for v in s.u_aniso))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _u_equiv(site: AtomSite) -> float:
    """U_eq の簡易値(直交格子で正確。一般格子では近似)。"""
    assert site.u_aniso is not None
    return float(np.mean(site.u_aniso[:3]))


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
