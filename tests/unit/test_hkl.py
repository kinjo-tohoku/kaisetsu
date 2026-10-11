import numpy as np
import pytest

from kaisetsu.crystal import Lattice, SpaceGroup
from kaisetsu.reflections import canonical_hkl, generate_reflections


def _brute_force_count(lattice: Lattice, sg: SpaceGroup, d_min: float) -> int:
    """球内の全格子点のうち消滅しないものの個数(多重度の和に一致するはず)。"""
    n = int(max(lattice.a, lattice.b, lattice.c) / d_min) + 2
    r = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    d = lattice.d_spacing(grid)
    inside = grid[d >= d_min * (1 - 1e-12)]
    return int(np.sum(~sg.is_absent(inside)))


@pytest.mark.parametrize(
    ("lattice", "symbol"),
    [
        (Lattice.cubic(4.0), "Pm-3m"),
        (Lattice.cubic(5.43), "Fd-3m"),
        (Lattice.cubic(3.0), "Im-3m"),
        (Lattice.tetragonal(4.0, 6.0), "I41/amd"),
        (Lattice.hexagonal(3.2, 5.2), "P63/mmc"),
        (Lattice.hexagonal(5.0, 13.0), "R-3c"),
        (Lattice.orthorhombic(8.48, 5.398, 6.958), "Pnma"),
        (Lattice.monoclinic(5.0, 6.0, 7.0, 100.0), "P21/c"),
        (Lattice(5.1, 6.2, 7.3, 81.0, 95.0, 103.0), "P-1"),
    ],
)
def test_multiplicity_sum_matches_brute_force(lattice: Lattice, symbol: str) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    d_min = 1.2
    refl = generate_reflections(lattice, sg, d_min)
    assert len(refl) > 0
    assert refl.d.min() >= d_min * (1 - 1e-9)
    assert np.all(np.diff(refl.d) <= 1e-12)  # d 降順
    assert int(refl.multiplicity.sum()) == _brute_force_count(lattice, sg, d_min)
    # 代表 hkl は自分自身が代表
    assert np.array_equal(canonical_hkl(sg, refl.hkl), refl.hkl)
    # 消滅反射は含まれない
    assert not sg.is_absent(refl.hkl).any()
    # 重複なし
    assert len({tuple(h) for h in refl.hkl}) == len(refl)


def test_simple_cubic_first_reflections() -> None:
    refl = generate_reflections(Lattice.cubic(4.0), SpaceGroup.from_symbol("Pm-3m"), 1.0)
    first = [tuple(h) for h in refl.hkl[:6]]
    assert first == [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0), (2, 1, 1)]
    assert list(refl.multiplicity[:6]) == [6, 12, 8, 6, 24, 24]
    assert refl.d[0] == pytest.approx(4.0)
    assert refl.index_of(1, 1, 0) == 1
    assert refl.index_of(9, 9, 9) == -1


def test_fcc_absences_and_d_max() -> None:
    refl = generate_reflections(Lattice.cubic(5.43), SpaceGroup.from_symbol("Fd-3m"), 0.8)
    hkls = {tuple(h) for h in refl.hkl}
    assert (1, 1, 1) in hkls and (2, 2, 0) in hkls and (3, 1, 1) in hkls
    assert (1, 0, 0) not in hkls and (2, 0, 0) not in hkls and (1, 1, 0) not in hkls
    limited = generate_reflections(
        Lattice.cubic(5.43), SpaceGroup.from_symbol("Fd-3m"), 0.8, d_max=2.0
    )
    assert limited.d.max() <= 2.0 and (1, 1, 1) not in {tuple(h) for h in limited.hkl}
    with_absent = generate_reflections(
        Lattice.cubic(5.43), SpaceGroup.from_symbol("Fd-3m"), 0.8, include_absent=True
    )
    assert len(with_absent) > len(refl)


def test_canonical_hkl_maps_equivalents() -> None:
    sg = SpaceGroup.from_symbol("Pm-3m")
    eq = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    assert np.array_equal(canonical_hkl(sg, eq), np.tile([1, 0, 0], (4, 1)))
    hexa = SpaceGroup.from_symbol("P6/mmm")
    assert np.array_equal(canonical_hkl(hexa, [[-1, 1, 0]]), [[1, 0, 0]])


def test_two_theta_and_invalid_dmin() -> None:
    refl = generate_reflections(Lattice.cubic(5.43094), SpaceGroup.from_symbol("Fd-3m"), 1.0)
    tt = refl.two_theta(1.5405929)
    assert tt[0] == pytest.approx(28.44, abs=0.01)
    with pytest.raises(ValueError):
        generate_reflections(Lattice.cubic(4.0), SpaceGroup.from_symbol("Pm-3m"), 0.0)
