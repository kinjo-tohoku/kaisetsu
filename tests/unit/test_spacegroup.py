import gemmi
import numpy as np
import pytest

from kaisetsu.crystal.lattice import CrystalSystem
from kaisetsu.crystal.spacegroup import SpaceGroup, SymOp


def test_from_symbol_variants() -> None:
    for sym in ("Pnma", "P n m a", "P N M A"):
        sg = SpaceGroup.from_symbol(sym)
        assert sg.number == 62
        assert sg.n_operations == 8
    assert SpaceGroup.from_symbol("Fd-3m").number == 227
    assert SpaceGroup.from_symbol("F d 3 m").number == 227
    assert SpaceGroup.from_number(194).hm == "P 63/m m c"
    with pytest.raises(ValueError):
        SpaceGroup.from_symbol("Q x y z")


def test_basic_properties() -> None:
    sg = SpaceGroup.from_symbol("Pnma")
    assert sg.crystal_system is CrystalSystem.ORTHORHOMBIC
    assert sg.laue_class == "mmm"
    assert sg.centering == "P"
    assert sg.is_centrosymmetric
    fd = SpaceGroup.from_symbol("Fd-3m")
    assert fd.n_operations == 192
    assert fd.centering == "F"
    assert len(fd.laue_rotations) == 48
    p63 = SpaceGroup.from_symbol("P 63 m c")
    assert not p63.is_centrosymmetric
    assert len(p63.laue_rotations) == 24  # 6/mmm


def test_symop_roundtrip() -> None:
    op = SymOp.from_triplet("-x+1/2,y,z+1/2")
    assert np.array_equal(op.rot, np.diag([-1, 1, 1]))
    assert np.allclose(op.trans, [0.5, 0.0, 0.5])
    assert SymOp.from_triplet(op.triplet) == op or np.allclose(
        SymOp.from_triplet(op.triplet).trans, op.trans
    )
    x = np.array([0.1, 0.2, 0.3])
    assert np.allclose(op.apply(x), [0.4, 0.2, 0.8])


def test_from_symops_matches_table() -> None:
    sg = SpaceGroup.from_symbol("Pnma")
    rebuilt = SpaceGroup.from_symops([op.triplet for op in sg.operations])
    assert rebuilt.number == 62
    assert rebuilt == sg
    # 生成元だけから閉じる
    gen = SpaceGroup.from_symops(["x,y,z", "-x,-y,-z", "-x+1/2,-y,z+1/2", "x,-y+1/2,z"])
    assert gen.number == 62


@pytest.mark.parametrize(
    ("symbol", "hkl", "absent"),
    [
        ("Fd-3m", (0, 0, 2), True),  # d 映進: 00l は l = 4n
        ("Fd-3m", (0, 0, 4), False),
        ("Fd-3m", (1, 1, 1), False),
        ("Fd-3m", (2, 0, 0), True),
        ("Fd-3m", (2, 2, 2), False),  # 空間群としては許容(ダイヤモンドでは基底で消える)
        ("Fd-3m", (1, 1, 0), True),  # F 中心化: 偶奇混在
        ("Fd-3m", (3, 1, 1), False),
        ("Pnma", (0, 1, 2), True),  # n 映進 (0kl): k + l = 2n
        ("Pnma", (0, 1, 1), False),
        ("Pnma", (0, 2, 2), False),
        ("Pnma", (1, 0, 1), False),  # m
        ("Pnma", (1, 1, 0), True),  # a 映進 (hk0): h = 2n
        ("Pnma", (2, 1, 0), False),
        ("Pnma", (0, 1, 0), True),  # 2_1 軸: 0k0 は k = 2n
        ("I4/mmm", (1, 0, 0), True),  # I 中心化
        ("I4/mmm", (1, 1, 0), False),
        ("P 63/m m c", (0, 0, 1), True),  # 6_3: 00l は l = 2n
        ("P 63/m m c", (0, 0, 2), False),
    ],
)
def test_systematic_absences(symbol: str, hkl: tuple[int, int, int], absent: bool) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    assert bool(sg.is_absent(hkl)[0]) is absent


@pytest.mark.parametrize(
    "symbol", ["P-1", "P21/c", "Pnma", "I41/amd", "R-3c", "P63/mmc", "Fd-3m", "Ia-3d"]
)
def test_absences_agree_with_gemmi(symbol: str) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    gops = gemmi.find_spacegroup_by_name(symbol).operations()
    rng = np.random.default_rng(1)
    hkl = rng.integers(-6, 7, size=(400, 3))
    ours = sg.is_absent(hkl)
    theirs = np.array([gops.is_systematically_absent([int(v) for v in h]) for h in hkl])
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize(
    ("symbol", "xyz", "mult"),
    [
        # Fd-3m は gemmi の既定で原点選択 1(8a が原点)
        ("Fd-3m", (0, 0, 0), 8),
        ("Fd-3m", (0.5, 0.5, 0.5), 8),  # 8b
        ("Fd-3m", (0.125, 0.125, 0.125), 16),  # 16c
        ("Fd-3m", (0.625, 0.625, 0.625), 16),  # 16d
        ("Fd-3m", (0.1, 0.1, 0.1), 32),  # 32e
        ("Fd-3m", (0.3, 0.0, 0.0), 48),  # 48f
        ("Fd-3m", (0.1, 0.1, 0.3), 96),  # 96g
        ("Fd-3m", (0.11, 0.23, 0.37), 192),  # 192i(0.1,0.2,0.3 は偶然 96h に乗る)
        ("Pnma", (0.188, 0.25, 0.167), 4),
        ("Pnma", (0.085, 0.026, 0.806), 8),
        ("Pnma", (0, 0, 0), 4),
        ("P63/mmc", (1 / 3, 2 / 3, 0.25), 2),
        ("P63/mmc", (0, 0, 0), 2),
        ("P63/mmc", (0.3, 0.6, 0.25), 6),
        ("R-3c", (0, 0, 0.25), 6),
        ("R-3c", (0.3, 0, 0.25), 18),
    ],
)
def test_site_multiplicity(symbol: str, xyz: tuple[float, float, float], mult: int) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    assert sg.multiplicity(xyz) == mult
    # 多重度 × サイト対称の位数 = 群の位数
    assert mult * len(sg.site_symmetry(xyz)) == sg.n_operations


@pytest.mark.parametrize(
    ("symbol", "hkl", "mult"),
    [
        ("Pm-3m", (1, 0, 0), 6),
        ("Pm-3m", (1, 1, 0), 12),
        ("Pm-3m", (1, 1, 1), 8),
        ("Pm-3m", (1, 2, 3), 48),
        ("P6/mmm", (1, 0, 0), 6),
        ("P6/mmm", (0, 0, 1), 2),
        ("P6/mmm", (1, 0, 1), 12),
        ("P6/mmm", (2, 1, 0), 12),
        ("P6/mmm", (2, 1, 1), 24),
        ("Pnma", (1, 0, 0), 2),
        ("Pnma", (1, 1, 0), 4),
        ("Pnma", (1, 1, 1), 8),
        ("P-1", (1, 2, 3), 2),
        ("P4/mmm", (1, 0, 0), 4),
        ("P4/mmm", (1, 1, 0), 4),
        ("P4/mmm", (1, 2, 0), 8),
        ("P4/mmm", (1, 2, 3), 16),
    ],
)
def test_reflection_multiplicity(symbol: str, hkl: tuple[int, int, int], mult: int) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    assert sg.reflection_multiplicity(hkl) == mult


def test_equivalent_positions_fd3m_8a() -> None:
    sg = SpaceGroup.from_symbol("Fd-3m")
    pos, ops = sg.equivalent_positions([0, 0, 0])
    assert pos.shape == (8, 3)
    assert len(ops) == 8
    expected = {
        (0, 0, 0),
        (0, 0.5, 0.5),
        (0.5, 0, 0.5),
        (0.5, 0.5, 0),
        (0.25, 0.25, 0.25),
        (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75),
        (0.75, 0.75, 0.25),
    }
    got = {tuple(np.round(p, 6)) for p in pos}
    assert got == expected
