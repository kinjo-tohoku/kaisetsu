"""対称操作の無い CIF の原点選択(レビュー 2026-10-17 物理 #3 の回帰テスト)。

H-M 記号だけの 'F d -3 m' は入口(記号か番号か)で原点選択が違い、原点選択 1 の座標で書いた
スピネルが :2 と解釈されて多重度が入れ替わっていた。CIF の多重度、無ければ最短原子間距離で選ぶ。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaisetsu.crystal.cif import read_structure

SPINEL = """data_spinel
_cell_length_a 8.0831
_cell_length_b 8.0831
_cell_length_c 8.0831
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
{sg}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{mult_tag}{atoms}"""
# 原点選択 1(8a が原点)と原点選択 2(反転中心が原点)の MgAl2O4
ORIGIN1 = [("Mg1 Mg 0 0 0", 8), ("Al1 Al 0.625 0.625 0.625", 16), ("O1 O 0.3874 0.3874 0.3874", 32)]
ORIGIN2 = [
    ("Mg1 Mg 0.125 0.125 0.125", 8),
    ("Al1 Al 0.5 0.5 0.5", 16),
    ("O1 O 0.2624 0.2624 0.2624", 32),
]


def _write(path: Path, sg_line: str, atoms: list[tuple[str, int]], *, with_mult: bool) -> Path:
    body = "".join(f"{a} {m}\n" if with_mult else f"{a}\n" for a, m in atoms)
    tag = "_atom_site_symmetry_multiplicity\n" if with_mult else ""
    path.write_text(SPINEL.format(sg=sg_line, mult_tag=tag, atoms=body), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "sg_line", ["_symmetry_space_group_name_H-M 'F d -3 m'", "_space_group_IT_number 227"]
)
@pytest.mark.parametrize(("atoms", "expected"), [(ORIGIN1, "F d -3 m:1"), (ORIGIN2, "F d -3 m:2")])
def test_origin_chosen_by_cif_multiplicity(
    tmp_path: Path, sg_line: str, atoms: list[tuple[str, int]], expected: str
) -> None:
    notes: list[str] = []
    st = read_structure(_write(tmp_path / "sp.cif", sg_line, atoms, with_mult=True), warnings=notes)
    assert st.spacegroup.symbol == expected
    assert st.multiplicities().tolist() == [8, 16, 32]
    assert st.composition() == {"Mg": 8.0, "Al": 16.0, "O": 32.0}
    assert any("原点選択" in m and expected in m for m in notes)
    assert not any("合いません" in m for m in notes)


@pytest.mark.parametrize(("atoms", "expected"), [(ORIGIN1, "F d -3 m:1"), (ORIGIN2, "F d -3 m:2")])
def test_origin_chosen_by_shortest_distance_without_multiplicity(
    tmp_path: Path, atoms: list[tuple[str, int]], expected: str
) -> None:
    """多重度の列が無ければ、間違った原点で原子が近すぎる(0.2〜0.3 Å)ことから選ぶ。"""
    notes: list[str] = []
    p = _write(
        tmp_path / "sp.cif", "_symmetry_space_group_name_H-M 'F d -3 m'", atoms, with_mult=False
    )
    st = read_structure(p, warnings=notes)
    assert st.spacegroup.symbol == expected
    assert st.multiplicities().tolist() == [8, 16, 32]
    assert any("最短原子間距離" in m and expected in m for m in notes)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("F d -3 m:1", "F d -3 m:1"),
        ("F d -3 m:2", "F d -3 m:2"),
        ("F d -3 m S", "F d -3 m:1"),  # ICSD 流: S = 原点選択 1
        ("F d -3 m Z", "F d -3 m:2"),  # ICSD 流: Z = 原点選択 2
    ],
)
def test_explicit_origin_is_respected_without_guessing(
    tmp_path: Path, symbol: str, expected: str
) -> None:
    """原点が明示されていれば座標に関係なくそれを使い、推定の警告は出ない。"""
    notes: list[str] = []
    p = _write(
        tmp_path / "sp.cif", f"_space_group_name_H-M_alt '{symbol}'", ORIGIN2, with_mult=False
    )
    st = read_structure(p, warnings=notes)
    assert st.spacegroup.symbol == expected
    assert not any("原点選択が書かれていません" in m for m in notes)


def test_wrong_explicit_origin_is_reported_by_multiplicity(tmp_path: Path) -> None:
    """原点選択 1 の座標なのに 'S'(= :1)と書かれていれば従うが、多重度が合わないと警告する。"""
    notes: list[str] = []
    p = _write(
        tmp_path / "sp.cif", "_space_group_name_H-M_alt 'F d -3 m S'", ORIGIN2, with_mult=True
    )
    st = read_structure(p, warnings=notes)
    assert st.spacegroup.symbol == "F d -3 m:1"
    assert any("多重度が CIF の 8 と合いません" in m for m in notes)


@pytest.mark.parametrize(
    ("cell", "symbol", "expected", "mults"),
    [
        ((4.759, 4.759, 12.991, 90, 90, 120), "R -3 c H", "R -3 c:H", [12, 18]),
        ((4.759, 4.759, 12.991, 90, 90, 120), "R -3 c", "R -3 c:H", [12, 18]),
    ],
)
def test_rhombohedral_hexagonal_axes_symbols(
    tmp_path: Path,
    cell: tuple[float, ...],
    symbol: str,
    expected: str,
    mults: list[int],
) -> None:
    a, b, c, al, be, ga = cell
    text = f"""data_cor
_cell_length_a {a}
_cell_length_b {b}
_cell_length_c {c}
_cell_angle_alpha {al}
_cell_angle_beta {be}
_cell_angle_gamma {ga}
_symmetry_space_group_name_H-M '{symbol}'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al 0 0 0.35216
O1 O 0.30624 0 0.25
"""
    p = tmp_path / "cor.cif"
    p.write_text(text, encoding="utf-8")
    st = read_structure(p)
    assert st.spacegroup.symbol == expected
    assert st.multiplicities().tolist() == mults


def test_anatase_origin_choice_by_distance(tmp_path: Path) -> None:
    """I41/amd のアナターゼ: 原点選択 2 の座標 (0, 1/4, 3/8) を H-M だけの CIF から正しく読む。"""
    text = """data_anatase
_cell_length_a 3.785
_cell_length_b 3.785
_cell_length_c 9.514
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'I 41/a m d'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 Ti 0 0.25 0.875
O1 O 0 0.25 0.0831
"""
    p = tmp_path / "anatase.cif"
    p.write_text(text, encoding="utf-8")
    st = read_structure(p)
    assert st.spacegroup.symbol == "I 41/a m d:2"
    assert st.multiplicities().tolist() == [4, 8]
    assert st.composition() == {"Ti": 4.0, "O": 8.0}


def test_unknown_symbol_still_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "bad.cif", "_symmetry_space_group_name_H-M 'Q x y z'", ORIGIN2, with_mult=False
    )
    with pytest.raises(ValueError, match="空間群記号を解釈できません"):
        read_structure(p)
