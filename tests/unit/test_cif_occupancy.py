"""CIF の占有率 0 が 1 に化けないこと(外部レビュー 2026-10-17 #4 の回帰テスト)。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaisetsu.api import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.crystal.cif import read_structure, write_structure

HEAD = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"""


def test_round_trip_keeps_zero_and_partial_occupancy(tmp_path: Path) -> None:
    st = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("P 1"),
        [
            AtomSite("Si1", "Si", [0, 0, 0], occupancy=1.0),
            AtomSite("O1", "O", [0.2, 0.3, 0.4], occupancy=0.5),
            AtomSite("O2", "O", [0.6, 0.7, 0.8], occupancy=0.0),
        ],
        name="occ",
    )
    p = tmp_path / "occ.cif"
    write_structure(st, p)
    loaded = read_structure(p)
    assert [s.occupancy for s in loaded.sites] == pytest.approx([1.0, 0.5, 0.0])
    assert loaded.cell_mass() == pytest.approx(st.cell_mass())


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("", 1.0),  # 列が無い
        ("?", 1.0),  # 不明
        (".", 1.0),  # 該当なし
        ("0", 0.0),
        ("0.25", 0.25),
        ("1.2", 1.0),  # 1 超は 1 に丸める(混合サイトの慣習)
    ],
)
def test_occupancy_values(tmp_path: Path, column: str, expected: float) -> None:
    p = tmp_path / "a.cif"
    if column:
        p.write_text(HEAD + "_atom_site_occupancy\nSi1 Si 0 0 0 " + column + "\n", encoding="utf-8")
    else:
        p.write_text(HEAD + "Si1 Si 0 0 0\n", encoding="utf-8")
    assert read_structure(p).sites[0].occupancy == pytest.approx(expected)


def test_negative_occupancy_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "neg.cif"
    p.write_text(HEAD + "_atom_site_occupancy\nSi1 Si 0 0 0 -0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="占有率が負"):
        read_structure(p)
