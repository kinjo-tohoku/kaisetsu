"""空間群と対称操作。

対称操作のテーブルは gemmi(MPL-2.0)から取得し、等価位置・多重度・消滅則・
ラウエ等価反射・サイト対称の計算はこのモジュールで NumPy により行う。

出典:
    - International Tables for Crystallography Vol. A(空間群表)
    - Giacovazzo, *Fundamentals of Crystallography*, Ch. 1(対称操作)、Ch. 3(消滅則)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

import gemmi
import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import CrystalSystem

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_DEN = float(gemmi.Op.DEN)  # gemmi は回転・並進を 1/24 単位の整数で持つ


@dataclass(frozen=True, eq=False)
class SymOp:
    """対称操作 x' = R x + t(分率座標に作用)。"""

    rot: IntArray
    trans: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rot", np.asarray(self.rot, dtype=np.int64).reshape(3, 3))
        object.__setattr__(self, "trans", np.asarray(self.trans, dtype=np.float64).reshape(3))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymOp):
            return NotImplemented
        dt = self.trans - other.trans
        dt -= np.round(dt)
        return bool(np.array_equal(self.rot, other.rot) and np.all(np.abs(dt) < 1e-9))

    def __hash__(self) -> int:
        return hash((self.rot.tobytes(), tuple(np.round(self.trans % 1.0, 6))))

    @classmethod
    def from_gemmi(cls, op: gemmi.Op) -> SymOp:
        rot = np.asarray(op.rot, dtype=np.float64) / _DEN
        if not np.allclose(rot, np.round(rot)):
            raise ValueError(f"整数でない回転行列は扱えません: {op.triplet()}")
        return cls(np.round(rot).astype(np.int64), np.asarray(op.tran, dtype=np.float64) / _DEN)

    @classmethod
    def from_triplet(cls, triplet: str) -> SymOp:
        """'x,1/2-y,z' 形式の文字列から構築する。"""
        return cls.from_gemmi(gemmi.Op(triplet))

    def to_gemmi(self) -> gemmi.Op:
        op = gemmi.Op()
        op.rot = [[int(v * _DEN) for v in row] for row in self.rot]
        op.tran = [round(v * _DEN) for v in self.trans]
        return op

    @property
    def triplet(self) -> str:
        """'x,-y+1/2,z' 形式の表記。"""
        return self.to_gemmi().triplet()

    def apply(self, xyz: npt.ArrayLike) -> FloatArray:
        """分率座標 (..., 3) に作用させる。"""
        x = np.asarray(xyz, dtype=np.float64)
        return np.asarray(x @ self.rot.T + self.trans)

    def apply_to_hkl(self, hkl: npt.ArrayLike) -> IntArray:
        """反射指数 (..., 3) に作用させる(h' = h R)。"""
        h = np.asarray(hkl, dtype=np.int64)
        return np.asarray(h @ self.rot)

    def transform_uij(self, u: FloatArray) -> FloatArray:
        """異方性変位テンソル U^{ij}(3x3, 反変成分)を変換する: U' = R U R^T。"""
        return np.asarray(self.rot @ u @ self.rot.T)

    def __repr__(self) -> str:
        return f"SymOp({self.triplet!r})"


class SpaceGroup:
    """空間群。

    `operations` には格子中心化を含む全ての対称操作(一般位置の数と同じ)が入る。
    """

    def __init__(self, ops: gemmi.GroupOps, sg: gemmi.SpaceGroup | None = None) -> None:
        self._gemmi_ops = ops
        self._gemmi_sg = sg if sg is not None else gemmi.find_spacegroup_by_ops(ops)
        self._ops: tuple[SymOp, ...] = tuple(SymOp.from_gemmi(op) for op in ops)
        if not self._ops:
            raise ValueError("対称操作が 1 つもありません")

    # ------------------------------------------------------------------
    # 構築
    # ------------------------------------------------------------------
    @classmethod
    def from_symbol(cls, symbol: str) -> SpaceGroup:
        """Hermann-Mauguin 記号(例 'P n m a', 'Pnma', 'Fd-3m', 'R -3 c:H')から構築する。"""
        sg = gemmi.find_spacegroup_by_name(symbol)
        if sg is None:
            raise ValueError(
                f"空間群記号を解釈できません: {symbol!r}。"
                "例: 'P n m a', 'Fd-3m', 'P 63/m m c', 'R -3 c:H'"
            )
        return cls(sg.operations(), sg)

    @classmethod
    def from_number(cls, number: int) -> SpaceGroup:
        """国際番号(1-230)から構築する。設定は gemmi の既定(通常は標準設定)。"""
        sg = gemmi.find_spacegroup_by_number(number)
        if sg is None:
            raise ValueError(f"空間群番号は 1〜230 で指定してください: {number}")
        return cls(sg.operations(), sg)

    @classmethod
    def from_symops(cls, triplets: Iterable[str]) -> SpaceGroup:
        """CIF の _space_group_symop_operation_xyz のリストから構築する。

        生成元だけが与えられた場合は群を閉じて補完する。
        """
        ops = gemmi.GroupOps([gemmi.Op(t) for t in triplets])
        ops.add_missing_elements()
        return cls(ops)

    @classmethod
    def from_gemmi(cls, sg: gemmi.SpaceGroup) -> SpaceGroup:
        return cls(sg.operations(), sg)

    # ------------------------------------------------------------------
    # 基本情報
    # ------------------------------------------------------------------
    @property
    def gemmi_spacegroup(self) -> gemmi.SpaceGroup | None:
        return self._gemmi_sg

    @property
    def number(self) -> int:
        """国際番号。表にない操作集合の場合は 0。"""
        return int(self._gemmi_sg.number) if self._gemmi_sg is not None else 0

    @property
    def symbol(self) -> str:
        """Hermann-Mauguin 記号(設定記号付き、例 'F d -3 m:1')。"""
        return str(self._gemmi_sg.xhm()) if self._gemmi_sg is not None else "?"

    @property
    def hm(self) -> str:
        return str(self._gemmi_sg.hm) if self._gemmi_sg is not None else "?"

    @property
    def hall(self) -> str:
        return str(self._gemmi_sg.hall) if self._gemmi_sg is not None else "?"

    @property
    def crystal_system(self) -> CrystalSystem:
        if self._gemmi_sg is None:
            raise ValueError("既知の空間群に対応しない対称操作のため晶系を決められません")
        return CrystalSystem(self._gemmi_sg.crystal_system_str())

    @property
    def laue_class(self) -> str:
        if self._gemmi_sg is None:
            raise ValueError("既知の空間群に対応しない対称操作のためラウエ群を決められません")
        return str(self._gemmi_sg.laue_str())

    @property
    def point_group(self) -> str:
        if self._gemmi_sg is None:
            raise ValueError("既知の空間群に対応しない対称操作のため点群を決められません")
        return str(self._gemmi_sg.point_group_hm())

    @property
    def centering(self) -> str:
        """格子中心化の記号(P, A, B, C, I, F, R, H)。"""
        return str(self._gemmi_ops.find_centering())

    @property
    def is_centrosymmetric(self) -> bool:
        return bool(self._gemmi_ops.is_centrosymmetric())

    @property
    def operations(self) -> tuple[SymOp, ...]:
        """全対称操作(中心化を含む)。"""
        return self._ops

    @property
    def n_operations(self) -> int:
        return len(self._ops)

    @cached_property
    def rotations(self) -> IntArray:
        """全操作の回転部 (N, 3, 3)。"""
        return np.stack([op.rot for op in self._ops])

    @cached_property
    def translations(self) -> FloatArray:
        """全操作の並進部 (N, 3)。"""
        return np.stack([op.trans for op in self._ops])

    @cached_property
    def laue_rotations(self) -> IntArray:
        """ラウエ群(点群 + 反転)の相異なる回転行列 (M, 3, 3)。反射の等価性判定に使う。"""
        rots = np.concatenate([self.rotations, -self.rotations])
        uniq = np.unique(rots.reshape(len(rots), 9), axis=0)
        return uniq.reshape(-1, 3, 3)

    # ------------------------------------------------------------------
    # 反射に関する操作
    # ------------------------------------------------------------------
    def is_absent(self, hkl: npt.ArrayLike) -> BoolArray:
        """消滅則により F(hkl) = 0 となる反射を判定する(ベクトル化)。

        操作 (R, t) について h R = h かつ h·t が整数でないなら消滅
        (Giacovazzo 3.3 節)。中心化による消滅もこの判定に含まれる。
        """
        h = np.atleast_2d(np.asarray(hkl, dtype=np.int64))
        absent = np.zeros(len(h), dtype=bool)
        for rot, tr in zip(self.rotations, self.translations, strict=True):
            same = np.all(h @ rot == h, axis=1)
            if not same.any():
                continue
            phase = h @ tr
            frac = np.abs(phase - np.round(phase))
            absent |= same & (frac > 1e-6)
        return absent

    def laue_equivalents(self, hkl: npt.ArrayLike) -> IntArray:
        """1 つの hkl に対しラウエ群で等価な反射の集合 (m, 3) を返す(重複なし)。"""
        h = np.asarray(hkl, dtype=np.int64).reshape(3)
        eq = np.einsum("j,mjk->mk", h, self.laue_rotations)
        return np.unique(eq, axis=0)

    def reflection_multiplicity(self, hkl: npt.ArrayLike) -> int:
        """粉末回折における反射多重度。"""
        return len(self.laue_equivalents(hkl))

    # ------------------------------------------------------------------
    # 原子位置に関する操作
    # ------------------------------------------------------------------
    def equivalent_positions(
        self, xyz: npt.ArrayLike, tol: float = 1e-4
    ) -> tuple[FloatArray, IntArray]:
        """等価位置の集合と、それぞれを生成した操作の添字を返す。

        Returns:
            positions: (m, 3)。単位胞内 [0, 1) に還元した相異なる位置。
            op_index: (m,)。positions[i] = ops[op_index[i]](xyz) となる最初の操作の添字。
        """
        x = np.asarray(xyz, dtype=np.float64).reshape(3)
        pos = (self.rotations @ x + self.translations) % 1.0
        pos = np.where(pos > 1.0 - tol, 0.0, pos)  # 1.0 近傍の丸め
        keep: list[int] = []
        kept: list[FloatArray] = []
        for i, p in enumerate(pos):
            if any(_same_position(p, q, tol) for q in kept):
                continue
            keep.append(i)
            kept.append(p)
        return np.array(kept).reshape(-1, 3), np.array(keep, dtype=np.int64)

    def multiplicity(self, xyz: npt.ArrayLike, tol: float = 1e-4) -> int:
        """サイト多重度(単位胞あたりの等価原子数)。"""
        return len(self.equivalent_positions(xyz, tol)[0])

    def site_symmetry(self, xyz: npt.ArrayLike, tol: float = 1e-4) -> tuple[SymOp, ...]:
        """位置 xyz を(格子並進を除いて)不変に保つ操作の集合(サイト対称群)。"""
        x = np.asarray(xyz, dtype=np.float64).reshape(3)
        result: list[SymOp] = []
        for op in self._ops:
            dx = op.apply(x) - x
            if np.all(np.abs(dx - np.round(dx)) < tol):
                result.append(op)
        return tuple(result)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"SpaceGroup({self.symbol!r}, number={self.number}, n_ops={self.n_operations})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpaceGroup):
            return NotImplemented
        if self.n_operations != other.n_operations:
            return False
        return bool(
            np.array_equal(self.rotations, other.rotations)
            and np.allclose(self.translations, other.translations)
        )

    def __hash__(self) -> int:
        return hash((self.number, self.n_operations))


def _same_position(p: FloatArray, q: FloatArray, tol: float) -> bool:
    d = p - q
    d -= np.round(d)
    return bool(np.all(np.abs(d) < tol))


def unique_positions(positions: Sequence[FloatArray] | FloatArray, tol: float = 1e-4) -> FloatArray:
    """周期境界を考慮して重複を除いた位置の配列 (m, 3) を返す。"""
    kept: list[FloatArray] = []
    for p in np.asarray(positions, dtype=np.float64).reshape(-1, 3):
        pm = p % 1.0
        if not any(_same_position(pm, q, tol) for q in kept):
            kept.append(pm)
    return np.array(kept).reshape(-1, 3)
