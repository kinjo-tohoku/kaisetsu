"""特殊位置の拘束が並進成分(定数項)を保つこと(外部レビュー 2026-10-17 #1 の回帰テスト)。

y = x/2 + 1/2 のような関係を持つ特殊位置で、拘束を評価しても座標と構造因子が変わらないこと。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.api import AtomSite, Lattice, Project, Radiation, SpaceGroup, Structure
from kaisetsu.refine.constraints import position_constraint
from kaisetsu.reflections.structure_factor import structure_factors

HKL = np.array([[1, 0, 0], [1, 1, 0], [2, 0, 0], [1, 0, 1], [2, 1, 1]])


def _wrap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5


CASES = [
    # 空間群, 格子, 座標(並進付きの特殊位置)
    ("P 6/m m m", Lattice.hexagonal(4.0, 6.0), [0.2, 0.6, 0.0]),  # y = x/2 + 1/2 の鏡映線
    ("I a -3 d", Lattice.cubic(12.0), [0.125, 0.6629, 0.5871]),  # 48g (1/8, y, 1/4 - y)
    ("I 4/m c m", Lattice.tetragonal(5.5, 7.9), [0.7741, 0.2741, 0.0]),  # 8h (x, x + 1/2, 0)
    ("P 4 b m", Lattice.tetragonal(12.4, 3.9), [0.3282, 0.8282, 0.017]),  # 4c (x, x + 1/2, z)
    ("P 21 3", Lattice.cubic(6.6), [0.6989, 0.1989, 0.3011]),  # 4a (x, x, x) の並進付きの像
]


@pytest.mark.parametrize(("symbol", "lattice", "xyz"), CASES)
def test_position_constraint_keeps_affine_offset(
    symbol: str, lattice: Lattice, xyz: list[float]
) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    pc = position_constraint(sg, xyz)
    xs = np.asarray(pc.symmetrized)
    assert np.allclose(_wrap(xs, xyz), 0.0, atol=2e-3)
    assert pc.dependent, "並進付きの特殊位置なので従属座標があるはず"
    for j, rel in pc.dependent.items():
        val = sum(c * xs[p] for p, c in zip(rel.pivots, rel.coeffs, strict=True)) + rel.offset
        assert val == pytest.approx(xs[j], abs=1e-9), f"{symbol}: xyz[{j}] の定数項が違う"


@pytest.mark.parametrize(("symbol", "lattice", "xyz"), CASES)
def test_structure_unchanged_after_tie_evaluation(
    symbol: str, lattice: Lattice, xyz: list[float]
) -> None:
    """Project に入れて拘束を評価しても座標・多重度・|F|² が変わらない。"""
    st = Structure(
        lattice,
        SpaceGroup.from_symbol(symbol),
        [AtomSite("Si1", "Si", xyz, u_iso=0.01)],
        name="affine",
    )
    before = st.copy()
    rad = Radiation.neutron(1.5)
    f2_before = np.abs(structure_factors(before, HKL, rad)) ** 2
    proj = Project()
    proj.add_phase(st, name="phase")
    proj.params.evaluate_ties()
    proj.params.apply_all()
    assert np.allclose(_wrap(st.sites[0].position, before.sites[0].position), 0.0, atol=1e-6)
    assert st.multiplicities().tolist() == before.multiplicities().tolist()
    f2_after = np.abs(structure_factors(st, HKL, rad)) ** 2
    assert np.allclose(f2_after, f2_before, rtol=1e-9, atol=1e-9)


def test_dependent_follows_pivot_with_offset() -> None:
    """主座標を動かすと従属座標が「係数 × 主 + 定数」で追従する(I4/mcm 8h: y = x + 1/2)。"""
    st = Structure(
        Lattice.tetragonal(5.5, 7.9),
        SpaceGroup.from_symbol("I 4/m c m"),
        [AtomSite("O1", "O", [0.7741, 0.2741, 0.0])],
        name="tilt",
    )
    proj = Project()
    proj.add_phase(st, name="p")
    names = [f"p:O1:{c}" for c in "xyz"]
    free = [n for n in names if proj.params[n].tie is None and not proj.params[n].locked]
    assert len(free) == 1
    proj.params[free[0]].value += 0.01
    proj.params.evaluate_ties()
    proj.params.apply_all()
    x, y, _z = st.sites[0].position
    assert _wrap(y, x + 0.5) == pytest.approx(0.0, abs=1e-9)
