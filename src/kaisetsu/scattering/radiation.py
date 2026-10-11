"""放射線(X 線・中性子)と波長の定義。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: h c / e [eV·Å](CODATA 2018)。E[eV] = HC_EV_ANGSTROM / λ[Å]
HC_EV_ANGSTROM = 12398.419843320026

#: 代表的な特性 X 線の波長 [Å](Hölzer et al., Phys. Rev. A 56, 4554 (1997))
CHARACTERISTIC_WAVELENGTHS: dict[str, tuple[float, float]] = {
    "Cu": (1.5405929, 1.5444274),
    "Co": (1.7889960, 1.7928350),
    "Cr": (2.2897260, 2.2936060),
    "Fe": (1.9360410, 1.9399730),
    "Mo": (0.7093171, 0.7136100),
    "Ag": (0.5594218, 0.5638131),
}


class RadiationType(StrEnum):
    XRAY = "xray"
    NEUTRON = "neutron"


@dataclass(frozen=True)
class Radiation:
    """放射線の種類と波長。

    Attributes:
        kind: X 線か中性子か
        wavelength: 主波長 λ1 [Å](Kα1、放射光、中性子)
        wavelength2: 副波長 λ2 [Å](Kα2)。無ければ None。
        ratio2: 強度比 I(λ2)/I(λ1)。Kα2/Kα1 は通常 0.5 前後。
    """

    kind: RadiationType
    wavelength: float
    wavelength2: float | None = None
    ratio2: float = 0.0

    def __post_init__(self) -> None:
        if self.wavelength <= 0:
            raise ValueError(f"波長は正でなければなりません: {self.wavelength}")
        if self.wavelength2 is not None and self.wavelength2 <= 0:
            raise ValueError(f"副波長は正でなければなりません: {self.wavelength2}")

    @classmethod
    def xray(
        cls, wavelength: float, wavelength2: float | None = None, ratio2: float = 0.5
    ) -> Radiation:
        return cls(RadiationType.XRAY, wavelength, wavelength2, ratio2 if wavelength2 else 0.0)

    @classmethod
    def characteristic(
        cls, target: str = "Cu", ratio2: float = 0.5, kalpha2: bool = True
    ) -> Radiation:
        """特性 X 線(例 'Cu')。kalpha2=False なら Kα1 単色。"""
        if target not in CHARACTERISTIC_WAVELENGTHS:
            raise ValueError(
                f"未登録のターゲット: {target!r}。選択肢: {', '.join(CHARACTERISTIC_WAVELENGTHS)}"
            )
        l1, l2 = CHARACTERISTIC_WAVELENGTHS[target]
        return cls.xray(l1, l2 if kalpha2 else None, ratio2)

    @classmethod
    def synchrotron(cls, wavelength: float) -> Radiation:
        return cls(RadiationType.XRAY, wavelength)

    @classmethod
    def neutron(cls, wavelength: float) -> Radiation:
        return cls(RadiationType.NEUTRON, wavelength)

    @property
    def is_xray(self) -> bool:
        return self.kind is RadiationType.XRAY

    @property
    def is_neutron(self) -> bool:
        return self.kind is RadiationType.NEUTRON

    @property
    def energy_ev(self) -> float:
        """主波長に対応するエネルギー [eV]。"""
        return HC_EV_ANGSTROM / self.wavelength

    @property
    def energy_kev(self) -> float:
        return self.energy_ev / 1000.0

    @property
    def wavelengths(self) -> tuple[tuple[float, float], ...]:
        """(波長, 相対強度) の組。λ1 は 1.0。"""
        if self.wavelength2 is None or self.ratio2 <= 0:
            return ((self.wavelength, 1.0),)
        return ((self.wavelength, 1.0), (self.wavelength2, self.ratio2))
