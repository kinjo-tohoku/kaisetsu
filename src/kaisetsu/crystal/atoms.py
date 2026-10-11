"""原子サイト。

分率座標・占有率・原子変位パラメータ(ADP)を保持する。
ADP の慣例は CIF と同じ: U_iso [Å^2]、異方性は U^{ij} [Å^2](逆格子基底に関する反変成分)、
温度因子 T = exp(-2π^2 Σ_ij h_i h_j a*_i a*_j U^{ij})。

出典: Trueblood et al., Acta Cryst. A52, 770-781 (1996)(ADP の表記法)。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

EIGHT_PI2 = 8.0 * math.pi**2

_SPECIES_RE = re.compile(r"^([A-Z][a-z]?)(?:([+-])?(\d*)([+-])?)?$")


def split_species(species: str) -> tuple[str, int]:
    """散乱種のラベルを (元素記号, 電荷) に分ける。

    'Pb2+', 'Pb+2', 'O2-', 'O-2', 'Fe', 'H1-' などを受け付ける。
    """
    s = species.strip()
    m = _SPECIES_RE.match(s)
    if not m:
        raise ValueError(f"散乱種のラベルを解釈できません: {species!r}(例: 'Pb2+', 'O2-', 'Fe')")
    el, sign1, digits, sign2 = m.groups()
    sign = sign1 or sign2
    if (sign1 and sign2) or (digits and not sign):
        raise ValueError(f"散乱種のラベルを解釈できません: {species!r}(例: 'Pb2+', 'O2-', 'Fe')")
    if not sign:
        return el, 0
    n = int(digits) if digits else 1
    return el, n if sign == "+" else -n


def normalize_species(species: str) -> str:
    """ラベルを CIF 流の表記('Pb2+', 'O2-', 'Fe')に正規化する。"""
    el, q = split_species(species)
    if q == 0:
        return el
    return f"{el}{abs(q)}{'+' if q > 0 else '-'}"


@dataclass
class AtomSite:
    """結晶学的に独立な原子サイト。

    Attributes:
        label: サイト名(例 'Pb1')
        species: 散乱種(例 'Pb2+', 'O')。X 線の原子散乱因子の選択に使う。
            中性子では元素記号部分のみ使う。
        position: 分率座標 (3,)
        occupancy: 占有率(0-1)。サイト多重度は含めない。
        u_iso: 等方性変位パラメータ U_iso [Å^2]
        u_aniso: 異方性変位パラメータ (U11, U22, U33, U12, U13, U23) [Å^2]。None なら等方性。
    """

    label: str
    species: str
    position: FloatArray
    occupancy: float = 1.0
    u_iso: float = 0.0
    u_aniso: FloatArray | None = field(default=None)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.species = normalize_species(self.species)
        if self.u_aniso is not None:
            self.u_aniso = np.asarray(self.u_aniso, dtype=np.float64).reshape(6)
        if not 0.0 <= self.occupancy <= 1.0 + 1e-9:
            raise ValueError(f"{self.label}: 占有率は 0〜1 で指定してください: {self.occupancy}")

    @property
    def element(self) -> str:
        return split_species(self.species)[0]

    @property
    def charge(self) -> int:
        return split_species(self.species)[1]

    @property
    def is_anisotropic(self) -> bool:
        return self.u_aniso is not None

    @property
    def b_iso(self) -> float:
        """B_iso = 8π^2 U_iso [Å^2]。"""
        return EIGHT_PI2 * self.u_iso

    def u_matrix(self) -> FloatArray:
        """U^{ij} を対称 3x3 行列で返す。等方性サイトでは例外(U_iso を使うこと)。"""
        if self.u_aniso is None:
            raise ValueError(f"{self.label} は等方性サイトです。u_iso を使ってください")
        u11, u22, u33, u12, u13, u23 = (float(v) for v in self.u_aniso)
        return np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])

    def copy(self) -> AtomSite:
        return AtomSite(
            self.label,
            self.species,
            self.position.copy(),
            self.occupancy,
            self.u_iso,
            None if self.u_aniso is None else self.u_aniso.copy(),
        )

    def __repr__(self) -> str:
        x, y, z = self.position
        if self.u_aniso is not None:
            adp = f"u_aniso={np.array2string(self.u_aniso, precision=4)}"
        else:
            adp = f"u_iso={self.u_iso:.4f}"
        return (
            f"AtomSite({self.label!r}, {self.species!r}, ({x:.4f}, {y:.4f}, {z:.4f}), "
            f"occ={self.occupancy:.3f}, {adp})"
        )
