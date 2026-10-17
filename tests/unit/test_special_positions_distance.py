"""丸めた特殊位置を直交座標の距離で認識すること(レビュー 2026-10-17 物理 #1 の回帰テスト)。

分率座標の成分ごとの比較では、六方晶の操作 (−y, x−y, z) で誤差が 2 倍になり、
3 桁に丸めた (0.333, 0.667, z) が一般位置(多重度 6)として扱われ |F|² が約 9 倍になっていた。
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from kaisetsu.api import AtomSite, Lattice, Project, Radiation, SpaceGroup, Structure
from kaisetsu.crystal.cif import read_structure
from kaisetsu.crystal.lattice import CrystalSystem
from kaisetsu.refine.constraints import adp_constraint, position_constraint
from kaisetsu.reflections.structure_factor import powder_structure_factor_squared

ZNO_CIF = """data_ZnO
_cell_length_a 3.2498
_cell_length_b 3.2498
_cell_length_c 5.2066
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'P 63 m c'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
{mult_tag}
Zn1 Zn {x} {y} 0.0 1 0.008{mult}
O1 O {x} {y} 0.3819 1 0.010{mult}
"""
HKL = [[1, 0, 0], [0, 0, 2], [1, 0, 1], [1, 1, 0], [1, 0, 3]]
HEX = Lattice.hexagonal(3.2498, 5.2066)
P63MC = SpaceGroup.from_symbol("P 63 m c")


def _write_zno(path: Path, x: str, y: str, *, with_mult: bool = True) -> Path:
    text = ZNO_CIF.format(
        x=x,
        y=y,
        mult_tag="_atom_site_symmetry_multiplicity" if with_mult else "",
        mult=" 2" if with_mult else "",
    )
    path.write_text(text.replace("\n\n", "\n"), encoding="utf-8")
    return path


def _exact_zno() -> Structure:
    return Structure(
        HEX,
        P63MC,
        [
            AtomSite("Zn1", "Zn", [1 / 3, 2 / 3, 0.0], u_iso=0.008),
            AtomSite("O1", "O", [1 / 3, 2 / 3, 0.3819], u_iso=0.010),
        ],
    )


@pytest.mark.parametrize(
    ("x", "y"), [("0.33333", "0.66667"), ("0.3333", "0.6667"), ("0.333", "0.667")]
)
def test_rounded_zno_cif_keeps_multiplicity_two(tmp_path: Path, x: str, y: str) -> None:
    """3 桁でも 4 桁でも多重度 2、組成 Zn2O2、単位胞に 4 原子。"""
    st = read_structure(_write_zno(tmp_path / "zno.cif", x, y, with_mult=False))
    assert st.multiplicities().tolist() == [2, 2]
    assert st.composition() == {"Zn": 2.0, "O": 2.0}
    assert len(st.expand()) == 4
    assert np.all(st.min_image_distances() > 1.0)


@pytest.mark.parametrize(("x", "y"), [("0.3333", "0.6667"), ("0.333", "0.667")])
def test_project_snaps_rounded_coordinates_to_exact_special_position(
    tmp_path: Path, x: str, y: str
) -> None:
    """Project に入れると 1/3, 2/3 に乗り直り、|F|² が厳密な構造と一致する。"""
    st = read_structure(_write_zno(tmp_path / "zno.cif", x, y))
    proj = Project()
    proj.add_phase(st, "ZnO")
    st2 = proj.phases["ZnO"].structure
    for s in st2.sites:
        assert s.position[:2] == pytest.approx([1 / 3, 2 / 3], abs=1e-9)
    assert st2.multiplicities().tolist() == [2, 2]
    # 座標 x, y は対称性で固定(z だけ動く)
    for label in ("Zn1", "O1"):
        assert proj.params[f"ZnO:{label}:x"].locked
        assert proj.params[f"ZnO:{label}:y"].locked
        assert not proj.params[f"ZnO:{label}:z"].locked
    rad = Radiation.xray(1.5406)
    f2 = powder_structure_factor_squared(st2, HKL, rad)
    f2_exact = powder_structure_factor_squared(_exact_zno(), HKL, rad)
    assert np.allclose(f2, f2_exact, rtol=1e-9)


def test_two_digit_coordinates_are_not_within_distance_tolerance(tmp_path: Path) -> None:
    """(0.33, 0.67) は 0.01 Å の許容には入らない(像どうしが 0.03 Å 離れる)。

    CIF の多重度を使った乗せ直しは test_cif_values.py で確かめる。
    """
    st = read_structure(_write_zno(tmp_path / "zno.cif", "0.33", "0.67", with_mult=False))
    assert st.multiplicities().tolist() == [6, 6]
    assert np.all(st.min_image_distances() < 0.3)


def test_general_position_is_not_snapped() -> None:
    """特殊位置の近くでない一般位置は多重度 12 のまま、座標も拘束も変わらない。"""
    xyz = [0.30, 0.64, 0.10]  # 2y − x = 0.98: 6c (2y, y, z) から 0.065 Å 離れている
    assert P63MC.multiplicity(xyz, lattice=HEX) == 12
    pc = position_constraint(P63MC, xyz, lattice=HEX)
    assert pc.free == (0, 1, 2)
    assert pc.symmetrized == pytest.approx(xyz)
    # 6c (2y, y, z) に乗っている (0.30, 0.65, z) は特殊位置として扱う
    pc6 = position_constraint(P63MC, [0.30, 0.65, 0.10], lattice=HEX)
    assert P63MC.multiplicity([0.30, 0.65, 0.10], lattice=HEX) == 6
    assert len(pc6.free) == 2


@pytest.mark.parametrize(
    ("symbol", "lattice", "xyz"),
    [
        ("P 63 m c", HEX, [1 / 3, 2 / 3, 0.1]),
        ("P 63/m m c", HEX, [0.0, 0.0, 0.25]),
        ("Fd-3m", Lattice.cubic(8.08), [0.125, 0.125, 0.125]),
        ("Fd-3m", Lattice.cubic(8.08), [0.2624, 0.2624, 0.2624]),
        ("I 4/m c m", Lattice.tetragonal(5.5, 7.9), [0.7741, 0.2741, 0.0]),
        ("R -3 c", Lattice.hexagonal(4.759, 12.991), [0.30624, 0.0, 0.25]),
        ("P 21/c", Lattice.monoclinic(5.0, 6.0, 7.0, 100.0), [0.0, 0.0, 0.0]),
        ("P 21/c", Lattice.monoclinic(5.0, 6.0, 7.0, 100.0), [0.13, 0.27, 0.41]),
    ],
)
def test_distance_criterion_agrees_with_fractional_on_exact_positions(
    symbol: str, lattice: Lattice, xyz: list[float]
) -> None:
    """厳密な座標では、距離判定と従来の分率座標判定でサイト対称群・多重度が一致する。"""
    sg = SpaceGroup.from_symbol(symbol)
    frac_ops = {op.triplet for op in sg.site_symmetry(xyz)}
    dist_ops = {op.triplet for op in sg.site_symmetry(xyz, lattice=lattice)}
    assert frac_ops == dist_ops
    assert sg.multiplicity(xyz) == sg.multiplicity(xyz, lattice=lattice)
    assert sg.multiplicity(xyz, lattice=lattice) * len(dist_ops) == sg.n_operations
    pc = position_constraint(sg, xyz, lattice=lattice)
    assert pc.symmetrized == pytest.approx(np.asarray(xyz), abs=1e-9)
    ac = adp_constraint(sg, xyz, lattice=lattice)
    assert ac.n_free == adp_constraint(sg, xyz).n_free


def test_rounded_special_positions_across_table() -> None:
    """3 桁に丸めた (1/3, 2/3, 1/4) が、三方晶・六方晶の全設定で丸めない場合と同じ多重度になる。"""
    lat = Lattice.hexagonal(5.0, 8.0)
    exact = np.array([1 / 3, 2 / 3, 0.25])
    rounded = np.array([0.333, 0.667, 0.25])
    for gsg in gemmi.spacegroup_table():
        if gsg.crystal_system_str() not in ("trigonal", "hexagonal") or gsg.ext == "R":
            continue
        sg = SpaceGroup.from_gemmi(gsg)
        assert sg.multiplicity(rounded, lattice=lat) == sg.multiplicity(exact, lattice=lat), (
            gsg.xhm()
        )
        pc = position_constraint(sg, rounded, lattice=lat)
        assert sg.multiplicity(pc.symmetrized, lattice=lat) == sg.multiplicity(exact), gsg.xhm()


def test_crystal_system_without_table_entry() -> None:
    """gemmi の表に無い操作の組(原点をずらした P 2/m・P -1)でも晶系が決まる(物理 #8)。"""
    shifted_p2m = SpaceGroup.from_symops(["x,y,z", "-x+1/2,y,-z", "-x+1/2,-y,-z", "x,-y,z"])
    assert shifted_p2m.number == 0
    assert shifted_p2m.crystal_system is CrystalSystem.MONOCLINIC
    shifted_p1bar = SpaceGroup.from_symops(["x,y,z", "-x+1/2,-y,-z"])
    assert shifted_p1bar.number == 0
    assert shifted_p1bar.crystal_system is CrystalSystem.TRICLINIC
    b2m = SpaceGroup.from_symops(["x,y,z", "-x,-y,z", "-x,-y,-z", "x,y,-z", "x+1/2,y,z+1/2"])
    assert b2m.crystal_system is CrystalSystem.MONOCLINIC


def test_crystal_system_from_rotations_matches_gemmi_for_all_settings() -> None:
    for gsg in gemmi.spacegroup_table():
        sg = SpaceGroup.from_gemmi(gsg)
        assert sg._crystal_system_from_rotations().value == gsg.crystal_system_str(), gsg.xhm()
