"""結晶構造 = 格子 + 空間群 + 独立原子サイト。

対称操作で単位胞内の全原子に展開する機能(構造因子計算の前処理)を持つ。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import gemmi
import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

AVOGADRO = 6.02214076e23  # 1/mol

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ExpandedAtoms:
    """単位胞内の全原子(対称操作で展開済み)。

    Attributes:
        positions: (N, 3) 分率座標
        site_index: (N,) 元の独立サイトの添字
        species: 独立サイトごとの散乱種(重複なし、順序固定)
        species_index: (N,) 各原子の species 内添字
        occupancy: (N,) 占有率
        u_iso: (N,) U_iso [Å^2](異方性原子では U_eq 相当ではなく 0)
        beta: (N, 3, 3) 異方性温度因子 β_ij(T = exp(-h^T β h))。等方性原子では 0。
        is_anisotropic: (N,) 真なら beta を、偽なら u_iso を使う
        multiplicity: 独立サイトごとの多重度 (n_sites,)
    """

    positions: FloatArray
    site_index: IntArray
    species: tuple[str, ...]
    species_index: IntArray
    occupancy: FloatArray
    u_iso: FloatArray
    beta: FloatArray
    is_anisotropic: BoolArray
    multiplicity: IntArray

    def __len__(self) -> int:
        return len(self.positions)


class Structure:
    """結晶構造。

    Args:
        lattice: 単位胞
        spacegroup: 空間群
        sites: 独立原子サイトのリスト
        name: 相の名前(表示用)
    """

    def __init__(
        self,
        lattice: Lattice,
        spacegroup: SpaceGroup,
        sites: Iterable[AtomSite],
        name: str = "",
    ) -> None:
        self.lattice = lattice
        self.spacegroup = spacegroup
        self.sites: list[AtomSite] = list(sites)
        self.name = name
        labels = [s.label for s in self.sites]
        if len(set(labels)) != len(labels):
            raise ValueError("原子サイトのラベルが重複しています: " + ", ".join(labels))

    # ------------------------------------------------------------------
    @classmethod
    def from_cif(cls, path: str, block: str | None = None) -> Structure:
        from kaisetsu.crystal.cif import read_structure

        return read_structure(path, block)

    # ------------------------------------------------------------------
    @property
    def n_sites(self) -> int:
        return len(self.sites)

    def site(self, label: str) -> AtomSite:
        for s in self.sites:
            if s.label == label:
                return s
        raise KeyError(f"サイト {label!r} はありません")

    def multiplicities(self) -> IntArray:
        return np.array(
            [self.spacegroup.multiplicity(s.position) for s in self.sites], dtype=np.int64
        )

    def species_list(self) -> tuple[str, ...]:
        seen: list[str] = []
        for s in self.sites:
            if s.species not in seen:
                seen.append(s.species)
        return tuple(seen)

    def composition(self) -> dict[str, float]:
        """単位胞あたりの原子数(占有率込み)を元素ごとに返す。"""
        comp: dict[str, float] = {}
        for s, m in zip(self.sites, self.multiplicities(), strict=True):
            comp[s.element] = comp.get(s.element, 0.0) + s.occupancy * float(m)
        return comp

    def cell_mass(self) -> float:
        """単位胞の質量 [g/mol](占有率込み)。原子量は gemmi の表を使う。"""
        return sum(n * gemmi.Element(el).weight for el, n in self.composition().items())

    def density(self) -> float:
        """計算密度 [g/cm³] = M_cell / (N_A V)。"""
        v = self.lattice.volume
        return self.cell_mass() / (AVOGADRO * v * 1e-24) if v > 0 else 0.0

    def linear_absorption_coefficient(self, wavelength: float) -> float:
        """X 線の線吸収係数 μ [1/cm](光電吸収のみ。Cromer-Liberman の f'' から)。

        原子 1 個の光電吸収断面積 σ_a = 2 r_e λ f''(λ) を単位胞内で足し、体積で割る。
        コンプトン散乱・弾性散乱の寄与(軽元素で数 %〜1 割)は含まない。

        出典: International Tables Vol. C, Section 4.2.6(f'' と吸収の関係)。
        """
        from kaisetsu.scattering.xray import XrayScatteringFactors

        xt = XrayScatteringFactors()
        r_e = 2.8179403262e-5  # 古典電子半径 [Å]
        total = 0.0
        for s, m in zip(self.sites, self.multiplicities(), strict=True):
            fpp = xt.anomalous(s.species, wavelength).imag
            total += s.occupancy * float(m) * 2.0 * r_e * wavelength * fpp
        v = self.lattice.volume
        return (total / v) * 1.0e8 if v > 0 else 0.0  # 1/Å → 1/cm

    def formula_units(self) -> float:
        """最小公約の式単位数 Z の推定値(整数とは限らない)。"""
        counts = list(self.composition().values())
        if not counts:
            return 0.0
        g = counts[0]
        for c in counts[1:]:
            g = math.gcd(round(g * 1000), round(c * 1000)) / 1000
        return g

    # ------------------------------------------------------------------
    def expand(self, tol: float = 1e-4) -> ExpandedAtoms:
        """対称操作で単位胞内の全原子に展開する。

        異方性 U^{ij} は操作の回転部 R で U' = R U R^T と変換し、
        β_ij = 2π^2 a*_i a*_j U'^{ij} に直して保持する。
        """
        species = self.species_list()
        rec = self.lattice.reciprocal
        astar = np.array([rec.a, rec.b, rec.c])
        scale = 2.0 * math.pi**2 * np.outer(astar, astar)

        pos_list: list[FloatArray] = []
        site_idx: list[int] = []
        sp_idx: list[int] = []
        occ: list[float] = []
        uiso: list[float] = []
        beta: list[FloatArray] = []
        aniso: list[bool] = []
        mult: list[int] = []
        for i, s in enumerate(self.sites):
            positions, op_index = self.spacegroup.equivalent_positions(s.position, tol)
            m = len(positions)
            mult.append(m)
            u_mat = s.u_matrix() if s.is_anisotropic else None
            for p, k in zip(positions, op_index, strict=True):
                pos_list.append(p)
                site_idx.append(i)
                sp_idx.append(species.index(s.species))
                occ.append(s.occupancy)
                if u_mat is not None:
                    op = self.spacegroup.operations[int(k)]
                    beta.append(scale * op.transform_uij(u_mat))
                    uiso.append(0.0)
                    aniso.append(True)
                else:
                    beta.append(np.zeros((3, 3)))
                    uiso.append(s.u_iso)
                    aniso.append(False)
        n = len(pos_list)
        return ExpandedAtoms(
            positions=np.array(pos_list, dtype=np.float64).reshape(n, 3),
            site_index=np.array(site_idx, dtype=np.int64),
            species=species,
            species_index=np.array(sp_idx, dtype=np.int64),
            occupancy=np.array(occ, dtype=np.float64),
            u_iso=np.array(uiso, dtype=np.float64),
            beta=np.array(beta, dtype=np.float64).reshape(n, 3, 3),
            is_anisotropic=np.array(aniso, dtype=bool),
            multiplicity=np.array(mult, dtype=np.int64),
        )

    def copy(self) -> Structure:
        return Structure(self.lattice, self.spacegroup, [s.copy() for s in self.sites], self.name)

    def __repr__(self) -> str:
        return (
            f"Structure({self.name!r}, {self.spacegroup.symbol}, {self.lattice!r}, "
            f"{self.n_sites} sites)"
        )
