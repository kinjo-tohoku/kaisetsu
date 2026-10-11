"""粉末回折の反射リスト生成。

d_min 以上の全ての格子点 hkl を列挙し、ラウエ群で等価なものを 1 つにまとめ(代表 hkl)、
多重度を数え、消滅則で F = 0 となる反射を除く。

代表 hkl は等価反射のうち (h, k, l) を辞書式に比較して最大のもの。
多重度 = |ラウエ群| / |hkl を不変に保つ回転の数|。

出典: Pecharsky & Zavalij, Ch. 2.9 (多重度), Giacovazzo Ch. 3 (消滅則)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ReflectionList:
    """独立反射のリスト(d の降順)。"""

    hkl: IntArray
    d: FloatArray
    multiplicity: IntArray

    def __post_init__(self) -> None:
        n = len(self.hkl)
        if self.d.shape != (n,) or self.multiplicity.shape != (n,):
            raise ValueError("hkl, d, multiplicity の長さが一致しません")

    def __len__(self) -> int:
        return len(self.hkl)

    @property
    def h(self) -> IntArray:
        return self.hkl[:, 0]

    @property
    def k(self) -> IntArray:
        return self.hkl[:, 1]

    @property
    def l(self) -> IntArray:
        return self.hkl[:, 2]

    def two_theta(self, wavelength: float) -> FloatArray:
        """2θ [deg]。到達不能な反射は NaN。"""
        s = wavelength / (2.0 * self.d)
        with np.errstate(invalid="ignore"):
            return np.asarray(np.degrees(2.0 * np.arcsin(np.where(s <= 1.0, s, np.nan))))

    def subset(self, mask: npt.ArrayLike) -> ReflectionList:
        m = np.asarray(mask)
        return ReflectionList(self.hkl[m], self.d[m], self.multiplicity[m])

    def index_of(self, h: int, k: int, l: int) -> int:
        """指定 hkl(代表でなくてもよい)の添字。無ければ -1。"""
        match = np.where(np.all(self.hkl == np.array([h, k, l]), axis=1))[0]
        return int(match[0]) if len(match) else -1

    def __repr__(self) -> str:
        if len(self) == 0:
            return "ReflectionList(empty)"
        return f"ReflectionList(n={len(self)}, d={self.d.max():.4f}..{self.d.min():.4f} Å)"


def _hkl_key(hkl: IntArray, base: int, offset: int) -> IntArray:
    """(h, k, l) を辞書式順序を保つ整数に符号化する。"""
    return (hkl[:, 0] + offset) * base * base + (hkl[:, 1] + offset) * base + (hkl[:, 2] + offset)


def canonical_hkl(spacegroup: SpaceGroup, hkl: npt.ArrayLike) -> IntArray:
    """各 hkl をラウエ群における代表 hkl((h,k,l) 辞書式最大)に写す。"""
    h = np.atleast_2d(np.asarray(hkl, dtype=np.int64))
    hmax = int(np.abs(h).max()) if h.size else 0
    base = 2 * hmax + 1
    best = h.copy()
    best_key = _hkl_key(h, base, hmax)
    for rot in spacegroup.laue_rotations:
        hr = h @ rot
        key = _hkl_key(hr, base, hmax)
        better = key > best_key
        best[better] = hr[better]
        best_key = np.where(better, key, best_key)
    return best


def generate_reflections(
    lattice: Lattice,
    spacegroup: SpaceGroup,
    d_min: float,
    d_max: float = math.inf,
    include_absent: bool = False,
) -> ReflectionList:
    """d_min ≤ d ≤ d_max の独立反射を生成する。

    Args:
        lattice: 格子
        spacegroup: 空間群
        d_min: 最小面間隔 [Å]
        d_max: 最大面間隔 [Å](既定は無限大)
        include_absent: 真なら消滅則で消える反射も含める(Le Bail 等で使うことは無いが検証用)
    """
    if d_min <= 0:
        raise ValueError(f"d_min は正でなければなりません: {d_min}")
    # |h| ≤ a/d_min(h = a·s, |s| = 1/d のコーシー・シュワルツ)
    hmax = math.floor(lattice.a / d_min)
    kmax = math.floor(lattice.b / d_min)
    lmax = math.floor(lattice.c / d_min)
    hr = np.arange(-hmax, hmax + 1, dtype=np.int64)
    kr = np.arange(-kmax, kmax + 1, dtype=np.int64)
    lr = np.arange(-lmax, lmax + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(hr, kr, lr, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]

    inv_d2 = lattice.inverse_d_squared(grid)
    keep = inv_d2 <= 1.0 / d_min**2 * (1.0 + 1e-12)
    if math.isfinite(d_max):
        keep &= inv_d2 >= 1.0 / d_max**2 * (1.0 - 1e-12)
    hkl_all = grid[keep]
    if len(hkl_all) == 0:
        return ReflectionList(
            np.zeros((0, 3), dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64)
        )

    # 代表 hkl と安定化部分群の大きさ
    m = max(hmax, kmax, lmax)
    base = 2 * m + 1
    own_key = _hkl_key(hkl_all, base, m)
    best_key = own_key.copy()
    stabilizer = np.zeros(len(hkl_all), dtype=np.int64)
    laue = spacegroup.laue_rotations
    for rot in laue:
        hr_ = hkl_all @ rot
        stabilizer += np.all(hr_ == hkl_all, axis=1)
        np.maximum(best_key, _hkl_key(hr_, base, m), out=best_key)
    is_rep = own_key == best_key
    hkl = hkl_all[is_rep]
    mult = len(laue) // stabilizer[is_rep]

    if not include_absent:
        present = ~spacegroup.is_absent(hkl)
        hkl = hkl[present]
        mult = mult[present]

    d = lattice.d_spacing(hkl)
    order = np.lexsort((-hkl[:, 2], -hkl[:, 1], -hkl[:, 0], -d))
    return ReflectionList(hkl[order], d[order], mult[order].astype(np.int64))
