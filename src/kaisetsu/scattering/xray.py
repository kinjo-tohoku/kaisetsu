"""X 線原子散乱因子 f(s, λ) = f0(s) + f'(λ) + i f''(λ)。

- f0(s): 解析近似 f0 = Σ_i a_i exp(-b_i s^2) + c, s = sinθ/λ [Å^-1]。
  既定は International Tables Vol. C Table 6.1.1.4 の 4 ガウス(Cromer-Mann)係数。
  GSAS-II と同じテーブルなので突き合わせ検証に向く。
  'wk1995' を選ぶと Waasmaier & Kirfel (1995) の 5 ガウス係数(高角まで精度が良い)。
- f', f'': Cromer & Liberman の方法(gemmi の実装)。
  D. T. Cromer & D. Liberman, J. Chem. Phys. 53, 1891 (1970); Acta Cryst. A37, 267 (1981)。
"""

from __future__ import annotations

import logging
from typing import Literal

import gemmi
import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.atoms import normalize_species, split_species
from kaisetsu.scattering.radiation import HC_EV_ANGSTROM
from kaisetsu.scattering.tables import IT1992, WK1995

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

log = logging.getLogger(__name__)

TableName = Literal["it1992", "wk1995"]


class XrayScatteringFactors:
    """X 線原子散乱因子の計算。

    Args:
        table: f0 の係数テーブル('it1992' または 'wk1995')
    """

    def __init__(self, table: TableName = "it1992") -> None:
        if table == "it1992":
            self._table = IT1992
        elif table == "wk1995":
            self._table = WK1995
        else:
            raise ValueError(f"未知のテーブル: {table!r}('it1992' か 'wk1995')")
        self.table_name: TableName = table

    # ------------------------------------------------------------------
    def resolve(self, species: str) -> str:
        """テーブルに存在するキーに解決する。イオンが無ければ中性原子に落とす。

        'Cval'(価電子状態の C)、'Hiso'、'Sival' のようなテーブル固有のラベルは、
        元素 + 電荷の正規化を通さずそのまま引く。
        """
        raw = species.strip()
        if raw in self._table:
            return raw
        key = normalize_species(species)
        if key in self._table:
            return key
        el, q = split_species(key)
        if el in ("D", "T"):
            # 重水素・三重水素は電子が 1 個で、X 線の散乱因子は H と同じ(中性子の CIF に多い)
            return "H"
        if q != 0 and el in self._table:
            log.warning(
                "散乱因子テーブル %s に %s が無いので中性原子 %s を使います",
                self.table_name,
                key,
                el,
            )
            return el
        raise KeyError(f"散乱因子テーブル {self.table_name} に {species!r} がありません")

    def coefficients(self, species: str) -> tuple[FloatArray, FloatArray, float]:
        a, b, c = self._table[self.resolve(species)]
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), float(c)

    @staticmethod
    def atomic_number(species: str) -> int:
        el = split_species(species)[0]
        z = gemmi.Element(el).atomic_number
        if z == 0:
            raise ValueError(f"元素記号を解釈できません: {species!r}")
        return int(z)

    # ------------------------------------------------------------------
    def f0(self, species: str, s: npt.ArrayLike) -> FloatArray:
        """f0(s)。s = sinθ/λ [Å^-1] の配列を受け取る。"""
        a, b, c = self.coefficients(species)
        s2 = np.square(np.asarray(s, dtype=np.float64))
        return np.asarray(
            np.sum(a * np.exp(-np.outer(s2.ravel(), b)), axis=1).reshape(s2.shape) + c
        )

    def anomalous(self, species: str, wavelength: float) -> complex:
        """異常分散項 f' + i f''(Cromer-Liberman)。"""
        z = self.atomic_number(species)
        energy = HC_EV_ANGSTROM / wavelength
        fp, fpp = gemmi.cromer_liberman(z=z, energy=energy)
        return complex(fp, fpp)

    def f(self, species: str, s: npt.ArrayLike, wavelength: float | None = None) -> ComplexArray:
        """f0(s) + f' + i f''。wavelength が None なら異常分散を含めない。"""
        f0 = self.f0(species, s).astype(np.complex128)
        if wavelength is None:
            return f0
        return np.asarray(f0 + self.anomalous(species, wavelength))

    def __repr__(self) -> str:
        return f"XrayScatteringFactors(table={self.table_name!r})"
