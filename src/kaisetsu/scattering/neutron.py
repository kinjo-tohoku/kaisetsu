"""中性子の束縛コヒーレント散乱長 b [fm]。

出典: V. F. Sears, Neutron News 3(3), 26-37 (1992)。
散乱長は角度に依存しない(点核)。磁気散乱は扱わない(将来)。
単位は fm (10^-15 m)。GSAS や FullProf の 10^-12 cm 単位とは 10 倍の関係。
"""

from __future__ import annotations

import logging

from kaisetsu.crystal.atoms import normalize_species, split_species
from kaisetsu.scattering.tables import SEARS1992

log = logging.getLogger(__name__)


class NeutronScatteringLengths:
    """中性子散乱長テーブル(Sears 1992)。"""

    def resolve(self, species: str) -> str:
        key = species.strip()
        if key in SEARS1992:  # 'D' のような同位体ラベルもそのまま通す
            return key
        key = normalize_species(species)
        if key in SEARS1992:
            return key
        el, _ = split_species(key)
        if el in SEARS1992:
            return el
        raise KeyError(f"中性子散乱長テーブルに {species!r} がありません")

    def scattering_length(self, species: str) -> complex:
        """束縛コヒーレント散乱長 b_c [fm](複素数。吸収の強い核種では虚部あり)。"""
        re_, im_, _ = SEARS1992[self.resolve(species)]
        return complex(re_, im_)

    def absorption_cross_section(self, species: str) -> float:
        """吸収断面積 σ_abs [barn](2200 m/s, λ = 1.798 Å)。"""
        return SEARS1992[self.resolve(species)][2]

    def __repr__(self) -> str:
        return "NeutronScatteringLengths(Sears1992)"
