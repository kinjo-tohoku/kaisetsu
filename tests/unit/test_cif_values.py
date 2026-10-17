"""CIF の座標 '1/3' や '?'、ラベルの重複、多重度の照合(レビュー 2026-10-17 物理 #4, #7)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.cif import read_structure

GRAPHITE = """data_graphite
_cell_length_a 2.46
_cell_length_b 2.46
_cell_length_c 6.70
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'P 63/m m c'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0.25
C2 C {x} {y} 0.25
"""

RUTILE_HEAD = """data_rutile
_cell_length_a 4.5937
_cell_length_b 4.5937
_cell_length_c 2.9587
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 42/m n m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"""


def test_fraction_coordinates_are_accepted_with_warning(tmp_path: Path) -> None:
    p = tmp_path / "g.cif"
    p.write_text(GRAPHITE.format(x="1/3", y="2/3"), encoding="utf-8")
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert st.sites[1].position == pytest.approx([1 / 3, 2 / 3, 0.25])
    assert st.multiplicities().tolist() == [2, 2]
    assert any("'1/3'" in m and "0.333333" in m for m in notes)


@pytest.mark.parametrize("bad", ["?", ".", "abc"])
def test_non_numeric_coordinate_is_an_error(tmp_path: Path, bad: str) -> None:
    p = tmp_path / "g.cif"
    p.write_text(GRAPHITE.format(x=bad, y="0.6667"), encoding="utf-8")
    with pytest.raises(ValueError, match="小数で書いてください"):
        read_structure(p)


def test_fraction_occupancy_is_accepted(tmp_path: Path) -> None:
    p = tmp_path / "r.cif"
    p.write_text(
        RUTILE_HEAD + "_atom_site_occupancy\nTi1 Ti 0 0 0 1/2\nO1 O 0.3053 0.3053 0 1\n",
        encoding="utf-8",
    )
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert st.sites[0].occupancy == pytest.approx(0.5)
    assert any("占有率" in m and "'1/2'" in m for m in notes)


def test_atom_site_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="小数で書いてください"):
        AtomSite("C2", "C", [np.nan, 0.5, 0.25])
    with pytest.raises(ValueError, match="占有率"):
        AtomSite("C2", "C", [0, 0, 0], occupancy=float("nan"))
    with pytest.raises(ValueError, match="U_iso"):
        AtomSite("C2", "C", [0, 0, 0], u_iso=float("inf"))
    with pytest.raises(ValueError, match="異方性"):
        AtomSite("C2", "C", [0, 0, 0], u_aniso=[0.01, np.nan, 0.01, 0, 0, 0])


def test_duplicate_labels_are_renamed(tmp_path: Path) -> None:
    p = tmp_path / "dup.cif"
    p.write_text(
        RUTILE_HEAD + "Ti Ti 0 0 0\nO O 0.3053 0.3053 0\nO O 0.1 0.2 0.3\n", encoding="utf-8"
    )
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert [s.label for s in st.sites] == ["Ti", "O_1", "O_2"]
    assert any("ラベル 'O' が 2 回" in m and "O_1, O_2" in m for m in notes)


def test_duplicate_labels_avoid_existing_names(tmp_path: Path) -> None:
    p = tmp_path / "dup.cif"
    p.write_text(
        RUTILE_HEAD + "O_1 O 0.1 0.1 0.1\nO O 0.3053 0.3053 0\nO O 0.2 0.2 0.2\n", encoding="utf-8"
    )
    st = read_structure(p)
    labels = [s.label for s in st.sites]
    assert len(set(labels)) == 3
    assert labels[0] == "O_1" and "O_1" not in labels[1:]


def test_multiplicity_mismatch_is_reported(tmp_path: Path) -> None:
    """CIF の多重度と計算が合わず、近くの特殊位置でも合わなければ警告する(座標の打ち間違いなど)。"""
    p = tmp_path / "r.cif"
    p.write_text(
        RUTILE_HEAD + "_atom_site_symmetry_multiplicity\nTi1 Ti 0 0 0 2\nO1 O 0.3053 0.2 0 4\n",
        encoding="utf-8",
    )
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert st.multiplicities().tolist() == [2, 8]
    assert any("O1" in m and "多重度が CIF の 4 と合いません" in m for m in notes)
    assert not any("Ti1" in m for m in notes)


def test_clean_cif_produces_no_warnings(tmp_path: Path) -> None:
    p = tmp_path / "r.cif"
    p.write_text(
        RUTILE_HEAD + "_atom_site_symmetry_multiplicity\nTi1 Ti 0 0 0 2\nO1 O 0.3053 0.3053 0 4\n",
        encoding="utf-8",
    )
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert st.multiplicities().tolist() == [2, 4]
    assert notes == []


ZNO = """data_ZnO
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
"""


def test_two_digit_coordinates_snap_when_cif_gives_multiplicity(tmp_path: Path) -> None:
    """(0.33, 0.67) は 0.01 Å の許容には入らないが、CIF の多重度 2 に合わせて乗せ直す(警告付き)。"""
    p = tmp_path / "zno.cif"
    p.write_text(
        ZNO + "_atom_site_symmetry_multiplicity\nZn1 Zn 0.33 0.67 0 2\nO1 O 0.33 0.67 0.3819 2\n",
        encoding="utf-8",
    )
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert st.multiplicities().tolist() == [2, 2]
    for s in st.sites:
        assert s.position[:2] == pytest.approx([1 / 3, 2 / 3], abs=1e-9)
    assert sum("乗せ直しました" in m for m in notes) == 2


def test_two_digit_coordinates_without_multiplicity_warn_about_overlap(tmp_path: Path) -> None:
    """多重度の列が無ければ乗せ直さないが、像どうしが近すぎる警告を出す。"""
    p = tmp_path / "zno.cif"
    p.write_text(ZNO + "Zn1 Zn 0.33 0.67 0\nO1 O 0.33 0.67 0.3819\n", encoding="utf-8")
    notes: list[str] = []
    st = read_structure(p, warnings=notes)
    assert st.multiplicities().tolist() == [6, 6]
    assert np.all(st.min_image_distances() < 0.3)
    assert sum("像どうしが" in m for m in notes) == 2
