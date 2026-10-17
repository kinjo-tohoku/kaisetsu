"""格子(単位胞)の幾何学。

計量テンソル・逆格子・面間隔 d(hkl)・直交座標変換を扱う。

出典:
    - C. Giacovazzo (ed.), *Fundamentals of Crystallography*, 3rd ed., Ch. 2.
    - V. K. Pecharsky & P. Y. Zavalij, *Fundamentals of Powder Diffraction and
      Structural Characterization of Materials*, 2nd ed., Ch. 1 (式 1.8-1.14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class CrystalSystem(StrEnum):
    """晶系。格子定数の拘束条件を決める。"""

    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    TRIGONAL = "trigonal"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


@dataclass(frozen=True)
class Lattice:
    """単位胞。長さは Å、角度は度。

    Attributes:
        a, b, c: 格子定数 [Å]
        alpha, beta, gamma: 軸角 [deg]。alpha は b-c 間、beta は a-c 間、gamma は a-b 間。
    """

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            if getattr(self, name) <= 0:
                raise ValueError(f"格子定数 {name} は正でなければなりません: {getattr(self, name)}")
        for name in ("alpha", "beta", "gamma"):
            v = getattr(self, name)
            if not 0.0 < v < 180.0:
                raise ValueError(f"軸角 {name} は 0° と 180° の間でなければなりません: {v}")
        if self.volume <= 0.0 or not math.isfinite(self.volume):
            raise ValueError("この格子定数の組では実在する単位胞になりません(体積が正になりません)")

    # ------------------------------------------------------------------
    # 構築ヘルパー
    # ------------------------------------------------------------------
    @classmethod
    def cubic(cls, a: float) -> Lattice:
        return cls(a, a, a)

    @classmethod
    def tetragonal(cls, a: float, c: float) -> Lattice:
        return cls(a, a, c)

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float) -> Lattice:
        return cls(a, b, c)

    @classmethod
    def hexagonal(cls, a: float, c: float) -> Lattice:
        return cls(a, a, c, 90.0, 90.0, 120.0)

    @classmethod
    def rhombohedral(cls, a: float, alpha: float) -> Lattice:
        return cls(a, a, a, alpha, alpha, alpha)

    @classmethod
    def monoclinic(cls, a: float, b: float, c: float, beta: float) -> Lattice:
        """b 軸ユニークの単斜晶。"""
        return cls(a, b, c, 90.0, beta, 90.0)

    @classmethod
    def from_parameters(cls, params: npt.ArrayLike) -> Lattice:
        """(a, b, c, alpha, beta, gamma) の配列から構築する。"""
        p = np.asarray(params, dtype=float)
        if p.shape != (6,):
            raise ValueError("格子定数は 6 要素 (a, b, c, alpha, beta, gamma) で指定してください")
        return cls(*(float(x) for x in p))

    @property
    def parameters(self) -> FloatArray:
        """(a, b, c, alpha, beta, gamma) を配列で返す。"""
        return np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])

    # ------------------------------------------------------------------
    # 計量テンソル
    # ------------------------------------------------------------------
    @property
    def metric_tensor(self) -> FloatArray:
        """実格子の計量テンソル G (Pecharsky 式 1.11)。

        G_ij = a_i · a_j。 d(hkl) や原子間距離の計算の基礎。
        """
        ca, cb, cg = (math.cos(math.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        a, b, c = self.a, self.b, self.c
        return np.array(
            [
                [a * a, a * b * cg, a * c * cb],
                [a * b * cg, b * b, b * c * ca],
                [a * c * cb, b * c * ca, c * c],
            ]
        )

    @property
    def reciprocal_metric_tensor(self) -> FloatArray:
        """逆格子の計量テンソル G* = G^-1。"""
        return np.linalg.inv(self.metric_tensor)

    @property
    def volume(self) -> float:
        """単位胞体積 [Å^3]。V = sqrt(det G)。"""
        det = float(np.linalg.det(self.metric_tensor))
        return math.sqrt(det) if det > 0 else float("nan")

    @property
    def reciprocal(self) -> Lattice:
        """逆格子(2π を含まない結晶学の定義)。"""
        gs = self.reciprocal_metric_tensor
        ar, br, cr = (math.sqrt(gs[i, i]) for i in range(3))

        def _angle(cos_value: float) -> float:
            # 退化に近い格子では丸めで |cos| が 1 をわずかに超える
            return math.degrees(math.acos(min(1.0, max(-1.0, cos_value))))

        alpha_r = _angle(gs[1, 2] / (br * cr))
        beta_r = _angle(gs[0, 2] / (ar * cr))
        gamma_r = _angle(gs[0, 1] / (ar * br))
        return Lattice(ar, br, cr, alpha_r, beta_r, gamma_r)

    # ------------------------------------------------------------------
    # 面間隔・回折角
    # ------------------------------------------------------------------
    def inverse_d_squared(self, hkl: npt.ArrayLike) -> FloatArray:
        """1/d^2 = h^T G* h を返す。hkl は (..., 3) 配列。"""
        h = np.asarray(hkl, dtype=float)
        gs = self.reciprocal_metric_tensor
        return np.asarray(np.einsum("...i,ij,...j->...", h, gs, h), dtype=np.float64)

    def d_spacing(self, hkl: npt.ArrayLike) -> FloatArray:
        """面間隔 d(hkl) [Å]。hkl は (..., 3) 配列。"""
        inv_d2 = self.inverse_d_squared(hkl)
        with np.errstate(divide="ignore"):
            return np.asarray(1.0 / np.sqrt(inv_d2))

    def two_theta(self, hkl: npt.ArrayLike, wavelength: float) -> FloatArray:
        """Bragg 角 2θ [deg]。Bragg 式 λ = 2 d sinθ。到達不能な反射は NaN。"""
        d = self.d_spacing(hkl)
        s = wavelength / (2.0 * d)
        with np.errstate(invalid="ignore"):
            return np.asarray(np.degrees(2.0 * np.arcsin(np.where(s <= 1.0, s, np.nan))))

    # ------------------------------------------------------------------
    # 直交座標変換
    # ------------------------------------------------------------------
    @property
    def orthogonalization_matrix(self) -> FloatArray:
        """分率座標 → 直交座標 [Å] の変換行列 M (r_cart = M @ r_frac)。

        a 軸を x 軸に、b 軸を xy 平面に取る標準的な設定
        (Giacovazzo 2.3 節、PDB/CIF と同じ慣例)。
        """
        ca, cb, cg = (math.cos(math.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        sg = math.sin(math.radians(self.gamma))
        v = self.volume / (self.a * self.b * self.c)
        return np.array(
            [
                [self.a, self.b * cg, self.c * cb],
                [0.0, self.b * sg, self.c * (ca - cb * cg) / sg],
                [0.0, 0.0, self.c * v / sg],
            ]
        )

    def cartesian(self, frac: npt.ArrayLike) -> FloatArray:
        """分率座標 (..., 3) を直交座標 [Å] に変換する。"""
        f = np.asarray(frac, dtype=float)
        return np.asarray(f @ self.orthogonalization_matrix.T)

    def fractional(self, cart: npt.ArrayLike) -> FloatArray:
        """直交座標 [Å] (..., 3) を分率座標に変換する。"""
        c = np.asarray(cart, dtype=float)
        return np.asarray(c @ np.linalg.inv(self.orthogonalization_matrix).T)

    def distance(self, frac1: npt.ArrayLike, frac2: npt.ArrayLike) -> FloatArray:
        """2 点間距離 [Å](周期境界は考慮しない)。"""
        dv = np.asarray(frac2, dtype=float) - np.asarray(frac1, dtype=float)
        return np.asarray(np.sqrt(np.einsum("...i,ij,...j->...", dv, self.metric_tensor, dv)))

    # ------------------------------------------------------------------
    # 晶系
    # ------------------------------------------------------------------
    def constrained(self, system: CrystalSystem) -> Lattice:
        """晶系の拘束を満たすように格子定数を整えて返す。

        精密化で a のみ動かし b, c を追従させる、といった用途に使う。
        """
        a, b, c = self.a, self.b, self.c
        al, be, ga = self.alpha, self.beta, self.gamma
        if system is CrystalSystem.CUBIC:
            return Lattice(a, a, a)
        if system is CrystalSystem.TETRAGONAL:
            return Lattice(a, a, c)
        if system is CrystalSystem.ORTHORHOMBIC:
            return Lattice(a, b, c)
        if system is CrystalSystem.HEXAGONAL:
            return Lattice(a, a, c, 90.0, 90.0, 120.0)
        if system is CrystalSystem.TRIGONAL:
            # 六方軸設定を既定とする。菱面体軸は rhombohedral() を使う。
            if abs(ga - 120.0) < 1e-6 or abs(al - 90.0) < 1e-6:
                return Lattice(a, a, c, 90.0, 90.0, 120.0)
            return Lattice(a, a, a, al, al, al)
        if system is CrystalSystem.MONOCLINIC:
            return Lattice(a, b, c, 90.0, be, 90.0)
        return Lattice(a, b, c, al, be, ga)

    def __repr__(self) -> str:
        return (
            f"Lattice(a={self.a:.5f}, b={self.b:.5f}, c={self.c:.5f}, "
            f"alpha={self.alpha:.4f}, beta={self.beta:.4f}, gamma={self.gamma:.4f})"
        )
