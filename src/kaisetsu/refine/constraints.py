"""対称性に由来する拘束の導出。

- 原子座標: サイト対称群 G の回転部 R に対し、許される変位 δ は全ての R で R δ = δ。
  射影 P = (1/|G|) Σ R の像が許される部分空間。その基底を行簡約して
  「主座標」と「従属座標 = Σ c × 主座標 + 定数」に分ける(例 (x, x, z) → y = x)。
- 異方性変位 U^{ij}: U' = R U R^T で不変な 6 次元部分空間を同様に求める。
- 格子定数: 晶系ごとの自由パラメータと等値関係。

出典: International Tables Vol. A, Section 1.4 (Wyckoff positions);
      Giacovazzo Ch. 1 (site symmetry)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import CrystalSystem, Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

FloatArray = npt.NDArray[np.float64]

#: U^{ij} の 6 成分の順序 (11, 22, 33, 12, 13, 23) と 3x3 の添字
_U6_INDEX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class LinearRelation:
    """従属成分 = Σ coeffs[k] × pivots[k] + offset。"""

    pivots: tuple[int, ...]
    coeffs: tuple[float, ...]
    offset: float


@dataclass(frozen=True)
class SymmetryConstraint:
    """n 成分のベクトルに対する対称拘束。

    Attributes:
        free: 自由に動かせる成分の添字(主成分)
        dependent: 従属成分の添字 → 主成分との線形関係
        fixed: 対称性で固定される成分の添字(動かせない)
        symmetrized: 対称化した現在値
    """

    free: tuple[int, ...]
    dependent: dict[int, LinearRelation]
    fixed: tuple[int, ...]
    symmetrized: FloatArray

    @property
    def n_free(self) -> int:
        return len(self.free)


def _rref_pivots(basis: FloatArray, tol: float = 1e-8) -> tuple[list[int], FloatArray]:
    """部分空間の基底(行ベクトル) r×n を行簡約し、主成分の列と簡約行列を返す。"""
    a = np.array(basis, dtype=np.float64)
    r, n = a.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        if row >= r:
            break
        k = row + int(np.argmax(np.abs(a[row:, col])))
        if abs(a[k, col]) < tol:
            continue
        a[[row, k]] = a[[k, row]]
        a[row] /= a[row, col]
        for i in range(r):
            if i != row:
                a[i] -= a[i, col] * a[row]
        pivots.append(col)
        row += 1
    return pivots, a[: len(pivots)]


def constraint_from_projector(
    proj: FloatArray, values: FloatArray, tol: float = 1e-8, *, project_values: bool = True
) -> SymmetryConstraint:
    """射影行列 P (n×n) と現在値から対称拘束を組み立てる。

    拘束は「許される値の集合 = sym + P の像」というアフィン部分空間で、従属成分の定数項は
    sym から求める。project_values が真なら sym = P @ values(ADP や歪みのように原点を通る
    線形な部分空間)。偽なら values をそのまま不変点 sym として使う。原子座標のように
    並進を含む対称操作の不変点(例: y = x/2 + 1/2)は P @ values では再現できないので、
    呼び出し側で求めた不変点を渡す(レビュー 2026-10-17 #1)。
    """
    n = proj.shape[0]
    vals = np.asarray(values, dtype=np.float64).reshape(n)
    sym = proj @ vals if project_values else vals.copy()
    # P の像の基底(列空間)
    u, s, _ = np.linalg.svd(proj)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return SymmetryConstraint((), {}, tuple(range(n)), sym)
    basis = u[:, :rank].T  # rank × n
    pivots, red = _rref_pivots(basis, tol)
    # red の各行は主成分 pivots[k] に対する係数行。
    # 従属成分 j の変位 = Σ_k red[k, j] × (主成分 k の変位)
    dependent: dict[int, LinearRelation] = {}
    fixed: list[int] = []
    for j in range(n):
        if j in pivots:
            continue
        coeffs = red[:, j]
        if np.all(np.abs(coeffs) < tol):
            fixed.append(j)
            continue
        nz = [(pivots[k], float(coeffs[k])) for k in range(len(pivots)) if abs(coeffs[k]) >= tol]
        piv = tuple(p for p, _ in nz)
        cf = tuple(c for _, c in nz)
        offset = float(sym[j] - sum(c * sym[p] for p, c in nz))
        dependent[j] = LinearRelation(piv, cf, offset)
    return SymmetryConstraint(tuple(pivots), dependent, tuple(fixed), sym)


def position_constraint(
    spacegroup: SpaceGroup,
    xyz: npt.ArrayLike,
    tol: float | None = None,
    *,
    lattice: Lattice | None = None,
) -> SymmetryConstraint:
    """原子座標 (x, y, z) の対称拘束。

    特殊位置と見なす許容差 tol は、lattice を与えたときは直交座標の距離 [Å]
    (既定 `DISTANCE_TOL` = 0.01 Å)、与えないときは分率座標の成分ごとの差(既定 5e-4)。
    CIF の丸め(3〜4 桁)程度のずれは特殊位置に乗せ直す。分率座標の成分比較では六方晶の
    (−y, x−y, z) で誤差が 2 倍になり (0.333, 0.667, z) を取りこぼすので、格子が分かるときは
    必ず lattice を渡す(レビュー 2026-10-17 物理 #1)。
    """
    x = np.asarray(xyz, dtype=np.float64).reshape(3)
    if lattice is None and tol is None:
        tol = 5e-4
    ops = spacegroup.site_symmetry(x, tol, lattice=lattice)
    proj = np.zeros((3, 3))
    for op in ops:
        proj += op.rot
    proj /= len(ops)
    # 射影の像で座標を対称化(特殊位置から僅かにずれていれば乗せ直す)。
    # 変位の射影 P(x_ref - x) ではなく、サイト対称操作の平均像 (1/n) Σ (R x + t - n) を使う
    # (格子並進の違いを補正した、並進を含む不変点)。
    xs = spacegroup.symmetrized_position(x, tol, lattice=lattice)
    # xs は並進を含む不変点なので射影し直さない(y = x/2 + 1/2 の 1/2 が落ちる)
    c = constraint_from_projector(proj, xs, 1e-8, project_values=False)
    return SymmetryConstraint(c.free, c.dependent, c.fixed, xs)


def _u6_to_mat(u6: FloatArray) -> FloatArray:
    m = np.zeros((3, 3))
    for k, (i, j) in enumerate(_U6_INDEX):
        m[i, j] = m[j, i] = u6[k]
    return m


def _mat_to_u6(m: FloatArray) -> FloatArray:
    return np.array([m[i, j] for i, j in _U6_INDEX])


def adp_constraint(
    spacegroup: SpaceGroup,
    xyz: npt.ArrayLike,
    u6: npt.ArrayLike | None = None,
    tol: float | None = None,
    *,
    lattice: Lattice | None = None,
) -> SymmetryConstraint:
    """異方性変位パラメータ (U11, U22, U33, U12, U13, U23) の対称拘束。

    tol と lattice の意味は `position_constraint` と同じ。
    """
    if lattice is None and tol is None:
        tol = 5e-4
    ops = spacegroup.site_symmetry(np.asarray(xyz, dtype=np.float64), tol, lattice=lattice)
    proj = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = 1.0
        m = _u6_to_mat(e)
        acc = np.zeros((3, 3))
        for op in ops:
            acc += op.rot @ m @ op.rot.T
        proj[:, k] = _mat_to_u6(acc / len(ops))
    vals = np.zeros(6) if u6 is None else np.asarray(u6, dtype=np.float64).reshape(6)
    return constraint_from_projector(proj, vals, 1e-8)


@dataclass(frozen=True)
class LatticeConstraint:
    """晶系による格子定数 (a, b, c, α, β, γ) の拘束。"""

    free: tuple[int, ...]
    ties: dict[int, int]  # 従属 → 主(等値)
    fixed: tuple[int, ...]


_LATTICE_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


def lattice_constraint(
    system: CrystalSystem, unique_axis: str = "b", rhombohedral: bool = False
) -> LatticeConstraint:
    """晶系ごとの格子定数拘束。"""
    if system is CrystalSystem.CUBIC:
        return LatticeConstraint((0,), {1: 0, 2: 0}, (3, 4, 5))
    if system is CrystalSystem.TETRAGONAL:
        return LatticeConstraint((0, 2), {1: 0}, (3, 4, 5))
    if system is CrystalSystem.HEXAGONAL or (system is CrystalSystem.TRIGONAL and not rhombohedral):
        return LatticeConstraint((0, 2), {1: 0}, (3, 4, 5))
    if system is CrystalSystem.TRIGONAL:  # 菱面体軸
        return LatticeConstraint((0, 3), {1: 0, 2: 0, 4: 3, 5: 3}, ())
    if system is CrystalSystem.ORTHORHOMBIC:
        return LatticeConstraint((0, 1, 2), {}, (3, 4, 5))
    if system is CrystalSystem.MONOCLINIC:
        angle = {"a": 3, "b": 4, "c": 5}.get(unique_axis, 4)
        fixed = tuple(i for i in (3, 4, 5) if i != angle)
        return LatticeConstraint((0, 1, 2, angle), {}, fixed)
    return LatticeConstraint((0, 1, 2, 3, 4, 5), {}, ())


def lattice_parameter_names() -> tuple[str, ...]:
    return _LATTICE_NAMES
