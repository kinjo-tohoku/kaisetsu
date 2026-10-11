"""装置パラメータ(定波長)と試料形状。

ピーク幅の慣例(FullProf / RIETAN-FP と同じ、単位は deg):
    Γ_G^2 = U tan^2θ + V tanθ + W + P / cos^2θ
    Γ_L   = X tanθ + Y / cosθ + Z
X は歪み的(tanθ)、Y は結晶子サイズ的(1/cosθ)な広がり。

GSAS-II からの換算(`from_gsas2`):
    GSAS-II は σ^2 [centideg^2] = U tan^2θ + V tanθ + W、
    γ [centideg] = X / cosθ + Y tanθ + Z(X と Y の役割が逆)なので
    U_here = 8 ln2 · U_gsas / 10^4(V, W も同様)、X_here = Y_gsas / 100、Y_here = X_gsas / 100。

出典: G. Caglioti, A. Paoletti & F. P. Ricci, Nucl. Instrum. 3, 223-228 (1958)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from kaisetsu.profile.corrections import bragg_brentano_shift, debye_scherrer_shift
from kaisetsu.scattering.radiation import Radiation

FloatArray = npt.NDArray[np.float64]

EIGHT_LN2 = 8.0 * math.log(2.0)

_MIN_FWHM_G = (
    2.3548200450309493 * math.sqrt(0.001) / 100.0
)  # GSAS-II の下限 σ^2 = 0.001 cdeg^2 相当
_MIN_FWHM_L = 1.0e-5


class GeometryKind(StrEnum):
    BRAGG_BRENTANO = "bragg-brentano"
    DEBYE_SCHERRER = "debye-scherrer"


@dataclass
class CWInstrument:
    """定波長(角度分散)回折計の装置パラメータ。

    Attributes:
        radiation: 放射線と波長
        zero: ゼロ点シフト [deg](計算位置に加える)
        u, v, w, p: ガウス成分の幅 [deg^2]
        x, y, z: ローレンツ成分の幅 [deg]
        s_l, h_l: FCJ 軸発散パラメータ S/L, H/L
        polarization: 偏光係数 P(中性子では無視)
    """

    radiation: Radiation
    zero: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.01
    p: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    s_l: float = 0.0
    h_l: float = 0.0
    polarization: float = 0.5

    # ------------------------------------------------------------------
    def fwhm_gauss(self, two_theta: npt.ArrayLike) -> FloatArray:
        th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
        t = np.tan(th)
        g2 = self.u * t * t + self.v * t + self.w + self.p / np.cos(th) ** 2
        return np.asarray(np.sqrt(np.maximum(g2, _MIN_FWHM_G**2)))

    def fwhm_lorentz(self, two_theta: npt.ArrayLike) -> FloatArray:
        th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
        lo = self.x * np.tan(th) + self.y / np.cos(th) + self.z
        return np.asarray(np.maximum(lo, _MIN_FWHM_L))

    # ------------------------------------------------------------------
    @classmethod
    def from_gsas2(cls, inst: Mapping[str, Any]) -> CWInstrument:
        """GSAS-II の装置パラメータ辞書(値のみ、または [初期値, 現在値, フラグ])から構築する。"""

        def val(key: str, default: float = 0.0) -> float:
            v = inst.get(key, default)
            if isinstance(v, list | tuple):
                v = v[1] if len(v) == 3 else v[0]
            return float(v)

        itype = str(inst.get("Type", "PXC"))
        if isinstance(inst.get("Type"), list | tuple):
            itype = str(inst["Type"][1])
        if "N" in itype:
            rad = Radiation.neutron(val("Lam"))
        else:
            lam2 = val("Lam2") if "Lam2" in inst else 0.0
            if "Lam1" in inst:
                rad = Radiation.xray(val("Lam1"), lam2 or None, val("I(L2)/I(L1)", 0.5))
            else:
                rad = Radiation.xray(val("Lam"))
        shl = val("SH/L")
        return cls(
            radiation=rad,
            zero=val("Zero"),
            u=EIGHT_LN2 * val("U") / 1.0e4,
            v=EIGHT_LN2 * val("V") / 1.0e4,
            w=EIGHT_LN2 * val("W") / 1.0e4,
            x=val("Y") / 100.0,
            y=val("X") / 100.0,
            z=val("Z") / 100.0,
            s_l=shl / 2.0,
            h_l=shl / 2.0,
            polarization=val("Polariz.", 0.5),
        )

    def to_gsas2_widths(self) -> dict[str, float]:
        """GSAS-II の U, V, W, X, Y, Z, SH/L に換算する(確認用)。"""
        return {
            "U": self.u * 1.0e4 / EIGHT_LN2,
            "V": self.v * 1.0e4 / EIGHT_LN2,
            "W": self.w * 1.0e4 / EIGHT_LN2,
            "X": self.y * 100.0,
            "Y": self.x * 100.0,
            "Z": self.z * 100.0,
            "SH/L": self.s_l + self.h_l,
        }


@dataclass
class SampleGeometry:
    """試料・光学系の形状に由来する補正パラメータ。

    Attributes:
        kind: Bragg-Brentano(反射、平板)か Debye-Scherrer(透過、キャピラリ)
        radius_mm: ゴニオメータ半径 [mm]
        displacement_mm: (BB) 試料高さずれ [mm]
        mu_eff_inv_mm: (BB) 透過補正の有効線吸収係数 [1/mm]。0 で補正なし
        roughness_a, roughness_b: (BB) Suortti 表面粗さ
        displace_x_mm, displace_y_mm: (DS) 試料位置ずれ [mm]
        mu_r: (DS) 円筒試料の μR
    """

    kind: GeometryKind = GeometryKind.BRAGG_BRENTANO
    radius_mm: float = 200.0
    displacement_mm: float = 0.0
    mu_eff_inv_mm: float = 0.0
    roughness_a: float = 0.0
    roughness_b: float = 0.0
    displace_x_mm: float = 0.0
    displace_y_mm: float = 0.0
    mu_r: float = 0.0

    def position_shift(self, two_theta: npt.ArrayLike) -> FloatArray:
        """試料に由来するピークシフト Δ2θ [deg]。"""
        if self.kind is GeometryKind.BRAGG_BRENTANO:
            return bragg_brentano_shift(
                two_theta, self.displacement_mm, self.radius_mm, self.mu_eff_inv_mm
            )
        return debye_scherrer_shift(
            two_theta, self.displace_x_mm, self.displace_y_mm, self.radius_mm
        )

    @classmethod
    def from_gsas2(cls, sample: Mapping[str, Any]) -> SampleGeometry:
        """GSAS-II の Sample Parameters から構築する。

        GSAS-II の Shift / DisplaceX / DisplaceY は μm、Transparency は 1/μ_eff を
        特有の単位で持つので換算する。
        """

        def val(key: str, default: float = 0.0) -> float:
            v = sample.get(key, default)
            if isinstance(v, list | tuple):
                v = v[0]
            return float(v)

        radius = val("Gonio. radius", 200.0)
        kind = (
            GeometryKind.DEBYE_SCHERRER
            if "Debye" in str(sample.get("Type", ""))
            else GeometryKind.BRAGG_BRENTANO
        )
        transparency = val("Transparency")
        # GSAS-II: Δ2θ = -(9/(π R)) T sin2θ [deg] ⇔ 1/(2 μ_eff) = T/20 mm
        mu_eff = 10.0 / transparency if transparency > 0 else 0.0
        return cls(
            kind=kind,
            radius_mm=radius,
            displacement_mm=val("Shift") / 1000.0,
            mu_eff_inv_mm=mu_eff,
            roughness_a=val("SurfRoughA"),
            roughness_b=val("SurfRoughB"),
            displace_x_mm=val("DisplaceX") / 1000.0,
            displace_y_mm=val("DisplaceY") / 1000.0,
            mu_r=val("Absorption"),
        )
