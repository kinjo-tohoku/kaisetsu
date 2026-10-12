"""TOF(飛行時間)中性子回折計の装置パラメータとパターン計算。

ピーク位置: TOF [μs] = DIFC d + DIFA d² + DIFB / d + ZERO(GSAS-II と同じ)
ピーク幅(GSAS-II の TOF type 3 と同じ定義、d は Å、TOF は μs):
    α = alpha / d                              立ち上がり [1/μs]
    β = beta0 + beta1 / d⁴ + betaq / d²        減衰 [1/μs]
    σ² = sig0 + sig1 d² + sig2 d⁴ + sigq d     ガウス分散 [μs²]
    γ = x d + y d² + z                         ローレンツ FWHM [μs]
強度: y(t) = y_b(t) + Σ_p S_p / V_p Σ_h m_h |F_h|² L_h P(t - t_h)、L = d⁴ sinθ(TOF の Lorentz)。
入射スペクトルはデータ側でバナジウム規格化されている前提(GSAS-II の FXYE と同じ)。

Z-Rietveld の Type0m(J-PARC の iMATERIA、SuperHRPD などの装置ファイル)も同じ部品で計算する
(`profile_kind = "zrietveld"`。alpha, beta* は使わない):
    プロファイル = TCH 擬 Voigt ⊗ [(1 − R) B2B(α, β₁) + R B2B(α, β₀)]
    B2B は接合点をピーク位置とする両側指数関数(上と同じ)、R は長い裾(β₀)の割合
    時定数の単位 τ [μs] = (TOF / 1000) × ρ(d)、ρ(d) = zr_scale0 + zr_scale1 ln d
    立ち上がり率 α = 1 / (τ α')、α' = 1 / (Alpha1 + Alpha2 d)
    減衰率 β₁ = 1 / (τ Beta1_0)、β₀ = 1 / (τ (Beta0_0 + Beta0_1 / d))
    R = r01 exp(−r02 / d^r03)
    σ² = SigmaSquare0 + SigmaSquare1 d² + SigmaSquare2 d⁴、γ = Gamma0 + Gamma1 d + Gamma2 d²
    (sig0, sig1, sig2 と z, x, y に写す)
    定義書が公開されていないので、Z-Rietveld 2.9.1 が iMATERIA BS の装置ファイルで描いた
    計算曲線(プロジェクトファイルから取り出した。validation/reference/zrietveld_type0m.json)
    から逆算した: 169 反射で α', β₀', β₁', R, σ², γ の d 依存は 1e-8 で一致、関数形は Si 111 で
    残差 2e-7、ρ(d) は 18 反射から求めた経験式(0.1% 以内)。

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
from kaisetsu.pattern.instrument import SampleGeometry
from kaisetsu.profile.corrections import cylinder_absorption_array
from kaisetsu.profile.tof_batch import tof_profile_batch
from kaisetsu.profile.tof_profile import tof_peak_fwhm
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
    #: プロファイル関数: "gsas3"(上の alpha, beta*)か "zrietveld"(Type0m。下の zr_*)
    profile_kind: str = "gsas3"
    #: Z-Rietveld Type0m(既定値は iMATERIA BS 155° バンクの配布ファイル)
    zr_alpha1: float = 5.2576
    zr_alpha2: float = -0.8175
    zr_beta1: float = 0.4552
    zr_beta0_0: float = 2.5788
    zr_beta0_1: float = 1.1534
    zr_r01: float = 0.1440
    zr_r02: float = 1.4737
    zr_r03: float = 5.2035
    #: 時定数の単位 τ = TOF/1000 × (zr_scale0 + zr_scale1 ln d)。Z-Rietveld の計算曲線からの経験式
    zr_scale0: float = 1.0544
    zr_scale1: float = -0.0306

    @property
    def is_zrietveld(self) -> bool:
        return self.profile_kind == "zrietveld"

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

    def sigma2_raw(self, d: FloatArray) -> FloatArray:
        """ガウス分散 σ² [μs²]。下限で止めない生の値(負になりうる)。"""
        return np.asarray(self.sig0 + self.sig1 * d**2 + self.sig2 * d**4 + self.sigq * d)

    def sigma2_of(self, d: FloatArray) -> FloatArray:
        return np.asarray(np.maximum(self.sigma2_raw(d), 1e-6))

    def gamma_of(self, d: FloatArray) -> FloatArray:
        return np.asarray(np.maximum(self.x * d + self.y * d * d + self.z, 1e-5))

    def zr_time_scale(self, d: FloatArray) -> FloatArray:
        """Type0m の時定数の単位 τ [μs] = TOF/1000 × (zr_scale0 + zr_scale1 ln d)。"""
        dd = np.maximum(np.asarray(d, dtype=np.float64), 1e-6)
        t = np.maximum(self.position(dd), 1.0)
        rho = np.maximum(self.zr_scale0 + self.zr_scale1 * np.log(dd), 1e-3)
        return np.asarray(t / 1000.0 * rho)

    def zr_rates(self, d: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Type0m の (立ち上がり率 α, 減衰率 β₀, 減衰率 β₁, 割合 R) [1/μs, 1/μs, 1/μs, -]。"""
        dd = np.maximum(np.asarray(d, dtype=np.float64), 1e-6)
        tau = self.zr_time_scale(dd)
        # α' = 1 / (Alpha1 + Alpha2 d) が時定数の係数(Z-Rietveld が反射ごとに書き出す値)
        a_ = 1.0 / np.maximum(self.zr_alpha1 + self.zr_alpha2 * dd, 1e-3)
        b0_ = np.maximum(self.zr_beta0_0 + self.zr_beta0_1 / dd, 1e-3)
        b1_ = max(self.zr_beta1, 1e-3)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            r = self.zr_r01 * np.exp(-self.zr_r02 / dd ** max(self.zr_r03, 0.0))
        r = np.clip(np.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return (
            np.asarray(1.0 / (tau * a_)),
            np.asarray(1.0 / (tau * b0_)),
            np.asarray(1.0 / (tau * b1_)),
            np.asarray(r),
        )

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
    def from_zdiffractometer(cls, path: str | Path) -> TOFInstrument:
        """Z-Rietveld の .zDiffractometer(TOF)を読む。

        [Conversion parameters] の c0, c1, c2(TOF = c0 + c1 d + c2 d²)をゼロ点・DIFC・DIFA に、
        [Diffraction angle] をバンクの 2θ にする。

        プロファイル Type0m / Type0 のうち、ガウス分散 σ² = SigmaSquare0 + SigmaSquare1 d²
        + SigmaSquare2 d⁴ [μs²] と、ローレンツ幅 γ = Gamma0 + Gamma1 d + Gamma2 d² [μs] は
        GSAS の type 3 と同じ形なので sig0, sig1, sig2 と z, x, y に写す(Z-Rietveld の
        クイックガイドにある iMATERIA BS の s1² = 45 μs² と、配布ファイルの 33.7 が整合する)。
        Type0m の立ち上がり・減衰(Alpha1, Alpha2, Beta1_0, Beta0_0, Beta0_1)と R(r01〜r03)が
        揃っていれば `profile_kind = "zrietveld"` にして zr_* に写す(モジュールの docstring)。
        無ければ GSAS type 3 の既定値のまま(標準手順が alpha, beta0, beta1 を動かす)。
        """
        from kaisetsu.io.readers.zrietveld import parse_zdiffractometer

        p = Path(path)
        z = parse_zdiffractometer(p)
        if not z.is_tof:
            kind = z.method or "種類不明"
            raise ValueError(f"{p.name}: TOF の装置ファイルではありません({kind})")
        c = z.conversion
        if len(c) < 2 or c[1] <= 0:
            raise ValueError(f"{p.name}: [Conversion parameters](DIFC)が読めません")
        prof = z.profile("Type0m", "Type0", "Type1")
        # Type0 / Type1 は "sigma square 0" のような名前なので Type0m の名前に揃える
        prof = {
            k.replace("sigma square ", "SigmaSquare").replace("gamma ", "Gamma"): v
            for k, v in prof.items()
        }
        inst = cls(
            difc=c[1],
            difa=c[2] if len(c) > 2 else 0.0,
            zero=c[0],
            two_theta=z.two_theta if z.two_theta is not None else 90.0,
        )
        sig = [prof.get(k, 0.0) for k in ("SigmaSquare0", "SigmaSquare1", "SigmaSquare2")]
        if any(k in prof for k in ("SigmaSquare0", "SigmaSquare1", "SigmaSquare2")) and (
            max(sig) > 0
        ):
            # SuperHRPD(BL08)の配布ファイルは σ² が d⁴ の項だけ(ローレンツ幅が主)
            inst.sig0, inst.sig1, inst.sig2 = sig
        if any(k in prof for k in ("Gamma0", "Gamma1", "Gamma2")):
            inst.z = prof.get("Gamma0", 0.0)
            inst.x = prof.get("Gamma1", 0.0)
            inst.y = prof.get("Gamma2", 0.0)
        zr_keys = ("Alpha1", "Alpha2", "Beta1_0", "Beta0_0", "Beta0_1", "r01", "r02", "r03")
        if all(k in prof for k in zr_keys):
            inst.profile_kind = "zrietveld"
            inst.zr_alpha1 = prof["Alpha1"]
            inst.zr_alpha2 = prof["Alpha2"]
            inst.zr_beta1 = prof["Beta1_0"]
            inst.zr_beta0_0 = prof["Beta0_0"]
            inst.zr_beta0_1 = prof["Beta0_1"]
            inst.zr_r01 = prof["r01"]
            inst.zr_r02 = prof["r02"]
            inst.zr_r03 = prof["r03"]
        return inst

    @classmethod
    def from_instrument_file(cls, path: str | Path) -> TOFInstrument:
        """GSAS-II の .instprm か Z-Rietveld の .zDiffractometer かを中身で判定して読む。"""
        from kaisetsu.io.readers.zrietveld import is_zdiffractometer

        p = Path(path)
        head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        if is_zdiffractometer(head):
            return cls.from_zdiffractometer(p)
        return cls.from_instprm(p)

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
    dd_size = d * d * ph.inverse_size(hkl, len(d)) / 1.0e4
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
    sample: SampleGeometry | None = None,
) -> PatternResult:
    """TOF パターンを計算する。x は TOF [μs](昇順)。

    sample.mu_r は円筒試料の μR を λ = 1 Å で与えたもの。反射ごとに λ = 2 d sinθ に比例させて
    Lobanov の近似式で透過率を求める(GSAS-II と同じ)。
    """
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
        po = ph.orientation_factor(refl.hkl)
        lam = 2.0 * d * sin_theta
        corr = lorentz * po
        if sample is not None and sample.mu_r > 0.0:
            corr = corr * cylinder_absorption_array(sample.mu_r * lam, instrument.two_theta)
        corr = corr * ph.extinction_factor(f2, lam, instrument.two_theta, False)
        inten = ph.scale / st.lattice.volume * refl.multiplicity * f2 * corr
        if instrument.is_zrietveld:
            alpha, beta0_zr, beta1_zr, r_zr = instrument.zr_rates(d)
            beta = np.minimum(beta0_zr, beta1_zr)  # 窓の決定には遅い方の減衰
        else:
            alpha = instrument.alpha_of(d)
            beta = instrument.beta_of(d)
            beta0_zr = beta1_zr = beta
            r_zr = np.zeros_like(d)
        sig2 = instrument.sigma2_of(d)
        gam = instrument.gamma_of(d)
        # 試料由来の広がり(サイズ・歪み・Stephens)を d 空間で求めて TOF に換算する:
        #   サイズ: Δd = d²/L(Scherrer、K = 1)、歪み: Δd/d = ε×10⁻⁶(GSAS-II の定義)
        add_g2, add_l = _sample_broadening_tof(ph, refl.hkl, d, instrument.difc)
        sig2 = sig2 + add_g2
        gam = gam + add_l
        n_refl = len(refl)
        fwhm_all = np.array(
            [
                tof_peak_fwhm(float(alpha[k]), float(beta[k]), float(sig2[k]), float(gam[k]))
                for k in range(n_refl)
            ]
        )
        ok = np.isfinite(pos) & (inten > 0.0)
        pos_safe = np.where(ok, pos, 0.0)
        lo = pos_safe - window_widths * fwhm_all
        hi = pos_safe + window_widths * fwhm_all * 1.5  # 減衰側に長い裾
        i0 = np.searchsorted(xa, lo).astype(np.int64)
        i1 = np.searchsorted(xa, hi).astype(np.int64)
        i1 = np.where(ok & (i1 > i0), i1, i0)
        if instrument.is_zrietveld:
            # Type0m: 短い裾(β₁)と長い裾(β₀)の両側指数を R で混ぜる
            out, offsets, lengths = tof_profile_batch(
                xa, pos_safe, alpha, beta1_zr, sig2, gam, inten * (1.0 - r_zr), i0, i1
            )
            out2, _, _ = tof_profile_batch(
                xa, pos_safe, alpha, beta0_zr, sig2, gam, inten * r_zr, i0, i1
            )
            out = out + out2
        else:
            out, offsets, lengths = tof_profile_batch(
                xa, pos_safe, alpha, beta, sig2, gam, inten, i0, i1
            )
        w_start = np.where(lengths > 0, i0, 0).astype(np.int64)
        w_stop = np.where(lengths > 0, i1, 0).astype(np.int64)
        contrib: list[FloatArray] = []
        for k in range(n_refl):
            if lengths[k] <= 0:
                contrib.append(np.zeros(0))
                continue
            part = out[offsets[k] : offsets[k] + lengths[k]]
            yp[i0[k] : i1[k]] += part
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


def zrietveld_peak(
    instrument: TOFInstrument, x: npt.ArrayLike, d: float, sigma2: float, gamma: float
) -> FloatArray:
    """Type0m の 1 ピーク(面積 1)を x [μs] 上に計算する(検証用)。

    位置は instrument.position(d)。sigma2, gamma は試料の広がりを足した値を渡す。
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    dd = np.array([float(d)])
    pos = instrument.position(dd)
    alpha, beta0, beta1, r = instrument.zr_rates(dd)
    i0 = np.zeros(1, dtype=np.int64)
    i1 = np.array([xa.size], dtype=np.int64)
    s2 = np.array([max(sigma2, 1e-6)])
    g = np.array([max(gamma, 1e-5)])
    y1, _, _ = tof_profile_batch(xa, pos, alpha, beta1, s2, g, 1.0 - r, i0, i1)
    y2, _, _ = tof_profile_batch(xa, pos, alpha, beta0, s2, g, r, i0, i1)
    return np.asarray(y1 + y2)


__all__ = ["TOFInstrument", "calculate_pattern_tof", "zrietveld_peak"]
