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
from pathlib import Path
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

    @property
    def sh_l(self) -> float:
        """FCJ 非対称の SH/L = S/L + H/L(GSAS-II と同じ 1 つの量)。

        FCJ の式は S と H の入れ替えに対称で、S/L = H/L のときは両者の微分が一致して
        同時には決められない(FAP の実データで相関 −1.000)。精密化では和だけを動かし、
        内部では S/L = H/L = SH/L / 2 とする。
        """
        return self.s_l + self.h_l

    @sh_l.setter
    def sh_l(self, value: float) -> None:
        self.s_l = self.h_l = 0.5 * float(value)

    # ------------------------------------------------------------------
    def gauss_fwhm2(self, two_theta: npt.ArrayLike) -> FloatArray:
        """Caglioti のガウス幅の 2 乗 Γ_G² [deg²]。下限で止めない生の値(負になりうる)。"""
        th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
        t = np.tan(th)
        return np.asarray(self.u * t * t + self.v * t + self.w + self.p / np.cos(th) ** 2)

    def lorentz_fwhm_raw(self, two_theta: npt.ArrayLike) -> FloatArray:
        """ローレンツ幅 Γ_L [deg]。下限で止めない生の値(負になりうる)。"""
        th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
        return np.asarray(self.x * np.tan(th) + self.y / np.cos(th) + self.z)

    def fwhm_gauss(self, two_theta: npt.ArrayLike) -> FloatArray:
        """ガウス幅 [deg]。Γ_G² が負のところは下限(GSAS-II と同じ)で止める。

        止めた範囲があるかは `kaisetsu.refine.width_check` で点検・罰則する。
        """
        return np.asarray(np.sqrt(np.maximum(self.gauss_fwhm2(two_theta), _MIN_FWHM_G**2)))

    def fwhm_lorentz(self, two_theta: npt.ArrayLike) -> FloatArray:
        return np.asarray(np.maximum(self.lorentz_fwhm_raw(two_theta), _MIN_FWHM_L))

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

    @classmethod
    def from_instprm(cls, path: str | Path) -> CWInstrument:
        """GSAS-II の .instprm(``key:value`` 行、CW 用)を読む。"""
        p = Path(path)
        vals = read_instprm_lines(p)
        if "C" not in vals.get("Type", ""):
            raise ValueError(
                f"{p.name}: 定波長(CW)の装置パラメータではありません(Type {vals.get('Type')})"
            )
        nums: dict[str, Any] = {k: float(v) for k, v in vals.items() if _is_number(v)}
        nums["Type"] = vals["Type"]
        return cls.from_gsas2(nums)

    @classmethod
    def from_gsas_prm(cls, path: str | Path, bank: int = 1) -> CWInstrument:
        """GSAS(旧)の装置パラメータファイル(.prm / .inst / .iparm)を読む。CW(PXC / PNC)用。

        APS 11-BM などが配っている形式。GSAS-II の SetPowderInstParms と同じ対応で読む:
        ICONS 行は固定桁(λ1, λ2, ゼロ点 [cdeg]、λ2 があれば比 I2/I1 と偏光、無ければ偏光)、
        PRCF1 行のプロファイル型が 3 なら PRCF11 = U V W、PRCF12 = X Y S/L H/L(SH/L はその和)、
        それ以外は X = Y = 0、SH/L = 0.002。U, V, W は cdeg²、X, Y は cdeg(GSAS の単位)。
        """
        p = Path(path)
        recs: dict[str, str] = {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("INS"):
                recs[line[:12]] = line[12:]
        htype = recs.get("INS   HTYPE ", "").strip()[:3]
        if not htype or "C" not in htype:
            raise ValueError(
                f"{p.name}: 定波長(CW)の GSAS 装置ファイルではありません(HTYPE {htype!r})"
            )
        pre = f"INS{bank:3d}"
        try:
            s = recs[pre + " ICONS"]
        except KeyError as exc:
            raise ValueError(f"{p.name}: バンク {bank} の ICONS 行がありません") from exc
        lam1 = _sfloat(s[:10])
        lam2 = _sfloat(s[10:20])
        zero = _sfloat(s[20:30]) / 100.0
        d: dict[str, Any] = {"Type": htype, "Zero": zero, "Z": 0.0}
        if lam2 > 0.0:
            d.update({"Lam1": lam1, "Lam2": lam2, "I(L2)/I(L1)": _sfloat(s[40:50]) or 0.5})
            d["Polariz."] = _sfloat(s[55:65])
        else:
            d["Lam"] = lam1
            d["Polariz."] = _sfloat(s[40:50])
        if "N" in htype:
            d["Polariz."] = 0.0
        ptype = 0
        if pre + "PRCF1 " in recs:
            ptype = int(float(recs[pre + "PRCF1 "].split()[0]))
        c1 = recs.get(pre + "PRCF11", "").split()
        c2 = recs.get(pre + "PRCF12", "").split()
        d["U"], d["V"], d["W"] = (float(c1[i]) if i < len(c1) else 0.0 for i in range(3))
        if ptype == 3 and len(c2) >= 4:
            d["X"], d["Y"] = float(c2[0]), float(c2[1])
            d["SH/L"] = float(c2[2]) + float(c2[3])
        else:
            d["X"], d["Y"], d["SH/L"] = 0.0, 0.0, 0.002
        return cls.from_gsas2(d)

    @classmethod
    def from_instrument_file(cls, path: str | Path, bank: int = 1) -> CWInstrument:
        """GSAS-II の .instprm か GSAS の .prm/.inst かを中身で判定して読む。"""
        p = Path(path)
        head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        if "INS   HTYPE" in head or head.lstrip().startswith("INS"):
            return cls.from_gsas_prm(p, bank)
        return cls.from_instprm(p)

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


def read_instprm_lines(path: str | Path) -> dict[str, str]:
    """GSAS-II の .instprm(``key:value`` 行)を辞書にする(値は文字列のまま)。"""
    vals: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        vals[key.strip()] = val.strip()
    return vals


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _sfloat(s: str) -> float:
    """空白や空文字なら 0。GSAS の固定桁フィールド用。"""
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


@dataclass
class SampleGeometry:
    """試料・光学系の形状に由来する補正パラメータ。

    Attributes:
        kind: Bragg-Brentano(反射、平板)か Debye-Scherrer(透過、キャピラリ)
        radius_mm: ゴニオメータ半径 [mm]
        displacement_mm: (BB) 試料高さずれ [mm]
        transparency_mm: (BB) 透過補正の有効侵入深さ 1/μ_eff [mm]。0 で補正なし
        roughness_a, roughness_b: (BB) Suortti 表面粗さ。a = 1 で補正なし、小さいほど粗い。
            b は角度依存の鋭さで、a と相関が強い(FAP で −0.997)ので通常 1 に固定する
        displace_x_mm, displace_y_mm: (DS) 試料位置ずれ [mm]
        mu_r: (DS) 円筒試料の μR
    """

    kind: GeometryKind = GeometryKind.BRAGG_BRENTANO
    radius_mm: float = 200.0
    displacement_mm: float = 0.0
    transparency_mm: float = 0.0
    roughness_a: float = 1.0
    roughness_b: float = 1.0
    displace_x_mm: float = 0.0
    displace_y_mm: float = 0.0
    mu_r: float = 0.0

    def __post_init__(self) -> None:
        # 旧来の「a = b = 0 で補正なし」は a = b = 1 に読み替える(a の微分が 0 にならないように)
        if self.roughness_a == 0.0 and self.roughness_b == 0.0:
            self.roughness_a = 1.0
            self.roughness_b = 1.0

    def position_shift(self, two_theta: npt.ArrayLike) -> FloatArray:
        """試料に由来するピークシフト Δ2θ [deg]。"""
        if self.kind is GeometryKind.BRAGG_BRENTANO:
            return bragg_brentano_shift(
                two_theta, self.displacement_mm, self.radius_mm, self.transparency_mm
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
        # GSAS-II: Δ2θ = -(9/(π R)) T sin2θ [deg] ⇔ 1/μ_eff = T/10 mm
        t_mm = transparency / 10.0
        return cls(
            kind=kind,
            radius_mm=radius,
            displacement_mm=val("Shift") / 1000.0,
            transparency_mm=t_mm,
            roughness_a=val("SurfRoughA"),
            roughness_b=val("SurfRoughB"),
            displace_x_mm=val("DisplaceX") / 1000.0,
            displace_y_mm=val("DisplaceY") / 1000.0,
            mu_r=val("Absorption"),
        )
