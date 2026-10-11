"""TOF(飛行時間)中性子回折計の装置パラメータとパターン計算。

ピーク位置: TOF [μs] = DIFC d + DIFA d² + DIFB / d + ZERO(GSAS-II と同じ)
ピーク幅(GSAS-II の TOF type 3 と同じ定義、d は Å、TOF は μs):
    α = alpha / d                              立ち上がり [1/μs]
    β = beta0 + beta1 / d⁴ + betaq / d²        減衰 [1/μs]
    σ² = sig0 + sig1 d² + sig2 d⁴ + sigq d     ガウス分散 [μs²]
    γ = x d + y d² + z                         ローレンツ FWHM [μs]
強度: y(t) = y_b(t) + Σ_p S_p / V_p Σ_h m_h |F_h|² L_h P(t - t_h)、L = d⁴ sinθ(TOF の Lorentz)。
入射スペクトルはデータ側でバナジウム規格化されている前提(GSAS-II の FXYE と同じ)。

GSAS-II の .instprm(テキスト)を読む `TOFInstrument.from_instprm` を用意する。

出典: R. B. Von Dreele, J. D. Jorgensen & C. G. Windsor, J. Appl. Cryst. 15, 581 (1982);
      GSAS-II のドキュメント "Instrument parameters (TOF)"。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kaisetsu.pattern.calculator import (
    PatternResult,
    PeakTable,
    PhaseContribution,
    _empty_peak_table,
)
from kaisetsu.profile.corrections import march_dollase
from kaisetsu.profile.tof_profile import tof_peak, tof_peak_fwhm
from kaisetsu.reflections.hkl import generate_reflections
from kaisetsu.reflections.structure_factor import structure_factors
from kaisetsu.scattering.radiation import Radiation

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class TOFInstrument:
    """TOF 回折計の 1 バンク分のパラメータ。"""

    difc: float
    difa: float = 0.0
    difb: float = 0.0
    zero: float = 0.0
    alpha: float = 0.1
    beta0: float = 0.03
    beta1: float = 0.002
    betaq: float = 0.0
    sig0: float = 0.0
    sig1: float = 50.0
    sig2: float = 0.0
    sigq: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    two_theta: float = 90.0
    flight_path_m: float = 0.0

    @property
    def radiation(self) -> Radiation:
        """構造因子計算用(中性子。波長は使わない)。"""
        return Radiation.neutron(1.0)

    @property
    def is_tof(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def position(self, d: npt.ArrayLike) -> FloatArray:
        da = np.asarray(d, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.asarray(self.difc * da + self.difa * da * da + self.difb / da + self.zero)

    def d_from_tof(self, tof: npt.ArrayLike) -> FloatArray:
        """TOF → d(ニュートン法。DIFA, DIFB は小さい補正)。"""
        t = np.asarray(tof, dtype=np.float64)
        d = (t - self.zero) / self.difc
        d = np.maximum(d, 1e-6)
        for _ in range(8):
            f = self.position(d) - t
            df = self.difc + 2.0 * self.difa * d - self.difb / (d * d)
            d = np.maximum(d - f / df, 1e-6)
        return np.asarray(d)

    def alpha_of(self, d: FloatArray) -> FloatArray:
        return np.asarray(self.alpha / d)

    def beta_of(self, d: FloatArray) -> FloatArray:
        return np.asarray(self.beta0 + self.beta1 / d**4 + self.betaq / d**2)

    def sigma2_of(self, d: FloatArray) -> FloatArray:
        return np.asarray(
            np.maximum(self.sig0 + self.sig1 * d**2 + self.sig2 * d**4 + self.sigq * d, 1e-6)
        )

    def gamma_of(self, d: FloatArray) -> FloatArray:
        return np.asarray(np.maximum(self.x * d + self.y * d * d + self.z, 1e-5))

    # ------------------------------------------------------------------
    @classmethod
    def from_instprm(cls, path: str | Path) -> TOFInstrument:
        """GSAS-II の .instprm(``key:value`` 行)を読む。"""
        p = Path(path)
        vals: dict[str, str] = {}
        for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, val = line.split(":", 1)
            vals[key.strip()] = val.strip()
        if "T" not in vals.get("Type", ""):
            raise ValueError(
                f"{p.name}: TOF の装置パラメータではありません(Type {vals.get('Type')})"
            )
        return cls.from_gsas2({k: _num(v) for k, v in vals.items() if _is_num(v)})

    @classmethod
    def from_gsas2(cls, inst: Mapping[str, object]) -> TOFInstrument:
        """GSAS-II の辞書(値のみ、または [初期値, 現在値, フラグ])から構築する。"""

        def val(key: str, default: float = 0.0) -> float:
            v = inst.get(key, default)
            if isinstance(v, list | tuple):
                v = v[1] if len(v) == 3 else v[0]
            return float(v)  # type: ignore[arg-type]

        return cls(
            difc=val("difC"),
            difa=val("difA"),
            difb=val("difB"),
            zero=val("Zero"),
            alpha=val("alpha", 0.1),
            beta0=val("beta-0", 0.03),
            beta1=val("beta-1", 0.002),
            betaq=val("beta-q"),
            sig0=val("sig-0"),
            sig1=val("sig-1", 50.0),
            sig2=val("sig-2"),
            sigq=val("sig-q"),
            x=val("X"),
            y=val("Y"),
            z=val("Z"),
            two_theta=val("2-theta", 90.0),
            flight_path_m=val("fltPath"),
        )

    def to_gsas2(self) -> dict[str, float]:
        return {
            "difC": self.difc, "difA": self.difa, "difB": self.difb, "Zero": self.zero,
            "alpha": self.alpha, "beta-0": self.beta0, "beta-1": self.beta1, "beta-q": self.betaq,
            "sig-0": self.sig0, "sig-1": self.sig1, "sig-2": self.sig2, "sig-q": self.sigq,
            "X": self.x, "Y": self.y, "Z": self.z, "2-theta": self.two_theta,
        }  # fmt: skip


def _sample_broadening_tof(
    ph: PhaseContribution, hkl: IntArray, d: FloatArray, difc: float
) -> tuple[FloatArray, FloatArray]:
    """試料由来の広がり(TOF、μs)。(ガウス分散 σ² への加算, ローレンツ FWHM への加算)。"""
    eight_ln2 = 8.0 * math.log(2.0)
    dd_size = np.zeros_like(d)
    if ph.size_um is not None and ph.size_um > 0:
        dd_size = d * d / (1.0e4 * ph.size_um)
    if ph.stephens is not None:
        from kaisetsu.profile.stephens import monomial_values

        var = monomial_values(hkl) @ np.asarray(ph.stephens, dtype=np.float64)
        m = 1.0 / (d * d)
        rel = np.where(m > 0, np.sqrt(np.maximum(var, 0.0)) / m, 0.0) * 1e-6
    else:
        rel = np.full_like(d, ph.strain_e6 * 1e-6)
    dd_strain = rel * d  # GSAS-II の TOF 定義 ΔT = DIFC d ε×10⁻⁶(CW の Δd/d = ε/2 の 2 倍)
    fs, fm = ph.size_lorentz_fraction, ph.strain_lorentz_fraction
    t_size = difc * dd_size
    t_strain = difc * dd_strain
    add_l = fs * t_size + fm * t_strain
    add_g2 = (((1.0 - fs) * t_size) ** 2 + ((1.0 - fm) * t_strain) ** 2) / eight_ln2
    return np.asarray(add_g2), np.asarray(add_l)


def _is_num(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _num(s: str) -> float:
    return float(s)


def calculate_pattern_tof(
    x: npt.ArrayLike,
    phases: Sequence[PhaseContribution],
    instrument: TOFInstrument,
    background: Callable[[FloatArray], FloatArray] | None = None,
    *,
    d_min: float | None = None,
    d_max: float | None = None,
    window_widths: float = 30.0,
    anomalous: Mapping[str, complex] | None = None,
    scattering_override: Mapping[str, complex] | None = None,
) -> PatternResult:
    """TOF パターンを計算する。x は TOF [μs](昇順)。"""
    xa = np.asarray(x, dtype=np.float64).ravel()
    d_lo, d_hi = (float(v) for v in instrument.d_from_tof(np.array([xa[0], xa[-1]])))
    dmin = d_min if d_min is not None else d_lo * 0.98
    dmax = d_max if d_max is not None else d_hi * 1.02
    y_bkg = (
        np.asarray(background(xa), dtype=np.float64)
        if background is not None
        else np.zeros_like(xa)
    )
    y_total = y_bkg.copy()
    y_phases: list[FloatArray] = []
    tables: list[PeakTable] = []
    sin_theta = math.sin(math.radians(instrument.two_theta / 2.0))
    rad = instrument.radiation
    for ph in phases:
        st = ph.structure
        refl = generate_reflections(st.lattice, st.spacegroup, dmin, dmax)
        yp = np.zeros_like(xa)
        if len(refl) == 0:
            y_phases.append(yp)
            tables.append(_empty_peak_table())
            continue
        if ph.f2_table is not None:
            f2 = np.array(
                [ph.f2_table.get((int(h), int(k), int(l)), ph.f2_default) for h, k, l in refl.hkl]
            )
        else:
            f = structure_factors(
                st, refl.hkl, rad, anomalous=anomalous, override=scattering_override
            )
            f2 = np.abs(f) ** 2
        d = refl.d
        pos = instrument.position(d)
        lorentz = d**4 * sin_theta
        po = march_dollase(
            refl.hkl, st.lattice, st.spacegroup, ph.march_dollase_axis, ph.march_dollase_r
        )
        inten = ph.scale / st.lattice.volume * refl.multiplicity * f2 * lorentz * po
        alpha = instrument.alpha_of(d)
        beta = instrument.beta_of(d)
        sig2 = instrument.sigma2_of(d)
        gam = instrument.gamma_of(d)
        # 試料由来の広がり(サイズ・歪み・Stephens)を d 空間で求めて TOF に換算する:
        #   サイズ: Δd = d²/L(Scherrer、K = 1)、歪み: Δd/d = ε×10⁻⁶(GSAS-II の定義)
        add_g2, add_l = _sample_broadening_tof(ph, refl.hkl, d, instrument.difc)
        sig2 = sig2 + add_g2
        gam = gam + add_l
        n_refl = len(refl)
        w_start = np.zeros(n_refl, dtype=np.int64)
        w_stop = np.zeros(n_refl, dtype=np.int64)
        contrib: list[FloatArray] = []
        fwhm_all = np.zeros(n_refl)
        for k in range(n_refl):
            width = tof_peak_fwhm(float(alpha[k]), float(beta[k]), float(sig2[k]), float(gam[k]))
            fwhm_all[k] = width
            lo = pos[k] - window_widths * width
            hi = pos[k] + window_widths * width * 1.5  # 減衰側に長い裾
            i0 = int(np.searchsorted(xa, lo))
            i1 = int(np.searchsorted(xa, hi))
            if i1 <= i0 or inten[k] <= 0 or not np.isfinite(pos[k]):
                contrib.append(np.zeros(0))
                continue
            part = inten[k] * tof_peak(xa[i0:i1] - pos[k], alpha[k], beta[k], sig2[k], gam[k])
            yp[i0:i1] += part
            w_start[k], w_stop[k] = i0, i1
            contrib.append(part)
        tables.append(
            PeakTable(
                hkl=refl.hkl,
                d=refl.d,
                multiplicity=refl.multiplicity,
                two_theta=pos,
                f2=f2,
                intensity=inten,
                fwhm=fwhm_all,
                fwhm_gauss=np.sqrt(sig2) * 2.3548200450309493,
                fwhm_lorentz=gam,
                intensity_total=inten.copy(),
                window_start=w_start,
                window_stop=w_stop,
                contributions=contrib,
            )
        )
        y_phases.append(yp)
        y_total += yp
    return PatternResult(x=xa, y_calc=y_total, y_background=y_bkg, y_phases=y_phases, peaks=tables)


__all__ = ["TOFInstrument", "calculate_pattern_tof"]
