"""実在の CIF のゆらぎへの耐性: 元素記号列が無い、酸化数 0、data_ 見出し無し、混合サイト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaisetsu.crystal.cif import read_structure
from kaisetsu.refine.project import Project

HEAD = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P m -3 m'
"""


def test_label_only_atom_sites(tmp_path: Path) -> None:
    """_atom_site_type_symbol が無く、ラベル 'Ca1' などしか無い CIF(AMCSD 由来の COD に多い)。"""
    p = tmp_path / "a.cif"
    p.write_text(
        HEAD
        + """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ca1 0 0 0
O2 0.5 0.5 0.5
""",
        encoding="utf-8",
    )
    st = read_structure(p)
    assert [s.species for s in st.sites] == ["Ca", "O"]
    assert [s.label for s in st.sites] == ["Ca1", "O2"]


def test_oxidation_state_zero_and_charged(tmp_path: Path) -> None:
    p = tmp_path / "b.cif"
    p.write_text(
        HEAD
        + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
La1 La0 0 0 0
B1 B0 0.5 0.5 0.2
O1 O2- 0.5 0 0
""",
        encoding="utf-8",
    )
    st = read_structure(p)
    assert [s.species for s in st.sites] == ["La", "B", "O2-"]


def test_missing_data_block_header(tmp_path: Path) -> None:
    p = tmp_path / "Ni.cif"
    body = HEAD.replace("data_test\n", "")
    p.write_text(
        body
        + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
""",
        encoding="utf-8",
    )
    st = read_structure(p)
    assert st.n_sites == 1 and st.sites[0].species == "Ni"


def test_mixed_site_occupancy_tie(tmp_path: Path) -> None:
    """同じ座標の 2 種は占有率の和が保たれる(後の種が前の種に従属)。"""
    p = tmp_path / "c.cif"
    p.write_text(
        HEAD
        + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0 0 0 0.6
Al1 Al 0 0 0 0.4
O1 O 0.5 0.5 0.5 1.0
""",
        encoding="utf-8",
    )
    proj = Project()
    proj.add_phase(read_structure(p), name="mix")
    ps = proj.params
    assert ps["mix:Al1:occ"].tie is not None
    assert ps["mix:O1:occ"].tie is None
    # 座標(ここでは対称性で固定)と U_iso も先頭の種に揃う
    assert ps["mix:Al1:x"].locked
    assert ps["mix:Al1:uiso"].tie is not None
    assert ps["mix:Al1:uiso"].tie.masters == ("mix:Fe1:uiso",)
    ps["mix:Fe1:uiso"].value = 0.012
    ps.evaluate_ties()
    ps.apply_all()
    assert proj.phases["mix"].structure.sites[1].u_iso == pytest.approx(0.012)
    ps["mix:Fe1:occ"].value = 0.75
    ps.evaluate_ties()
    ps.apply_all()
    assert ps["mix:Al1:occ"].value == pytest.approx(0.25)
    assert proj.phases["mix"].structure.sites[1].occupancy == pytest.approx(0.25)
