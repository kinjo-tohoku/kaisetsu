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
from kaisetsu.errors import PatternSizeError

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

#: hkl の候補(全格子点 (2h+1)(2k+1)(2l+1))の上限。格子定数が発散したときに
#: メモリを使い尽くす前に止める(2e7 で int64 の格子が 480 MB)
MAX_HKL_CANDIDATES = 20_000_000
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
        """格納されている代表 hkl と完全に一致する行の添字。無ければ -1。

        等価な別の hkl は見つからない。代表に写してから探すには `canonical_hkl` を使う。
        """
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
    # 位取りは回転像も含めた最大の成分で決める(六方晶の h + k などは入力の最大を超えるので、
    # 入力だけで決めると桁が繰り上がって順序が狂う。DeepSeek レビュー 2026-10-17 #2)
    images = [h @ rot for rot in spacegroup.laue_rotations]
    hmax = max((int(np.abs(im).max()) for im in images if im.size), default=0)
    base = 2 * hmax + 1
    best = h.copy()
    best_key = _hkl_key(h, base, hmax)
    for hr in images:
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
    max_candidates: int = MAX_HKL_CANDIDATES,
) -> ReflectionList:
    """d_min ≤ d ≤ d_max の独立反射を生成する。

    Args:
        lattice: 格子
        spacegroup: 空間群
        d_min: 最小面間隔 [Å]
        d_max: 最大面間隔 [Å](既定は無限大)
        include_absent: 真なら消滅則で消える反射も含める(Le Bail 等で使うことは無いが検証用)
        max_candidates: hkl の候補の上限。超えたら `PatternSizeError`(格子定数の発散)
    """
    if d_min <= 0:
        raise ValueError(f"d_min は正でなければなりません: {d_min}")
    # |h| ≤ a/d_min(h = a·s, |s| = 1/d のコーシー・シュワルツ)。採否の判定(下の keep)と
    # 同じ相対誤差を見込む(a/d_min がちょうど整数のとき端の反射が候補から落ちないように)
    hmax = math.floor(lattice.a / d_min * (1.0 + 1e-12))
    kmax = math.floor(lattice.b / d_min * (1.0 + 1e-12))
    lmax = math.floor(lattice.c / d_min * (1.0 + 1e-12))
    n_cand = (2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1)
    if n_cand > max_candidates:
        raise PatternSizeError(
            f"hkl の候補が {n_cand:.2e} 個になり上限 {max_candidates:.1e} 個を超えました"
            f"(a, b, c = {lattice.a:.2f}, {lattice.b:.2f}, {lattice.c:.2f} Å、"
            f"d_min = {d_min:.3f} Å)。格子定数が発散しているか、データ範囲に対して"
            "格子が大きすぎます。格子定数を確かめるか、2θ(d_min)の範囲を狭めてください"
        )
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
