"""粉末回折パターンのフォワード計算。

    y_calc(2θ_i) = y_b(2θ_i) + Σ_p S_p / V_p Σ_h m_h |F_h|^2 L(θ_h) P(θ_h) O_h A(θ_h) Ω(2θ_i - 2θ_h)

    S_p: 相のスケール因子、V_p: 単位胞体積、m: 多重度、L: Lorentz 因子、
    P: 偏光因子(X 線のみ)、O: 選択配向、A: 吸収・表面粗さ、Ω: 面積 1 のプロファイル関数。
    Kα2 があれば同じ反射を波長比の強度で λ2 の位置にも加える。

スケール因子の定義は GSAS-II と同じ(1/V を陽に含む)。GSAS-II のプロファイルは
centidegree あたりで規格化されているため、同じパターンに対して S_here = S_gsas / 100。

出典: R. A. Young (ed.), *The Rietveld Method*, Ch. 1, 式 (1.2)-(1.4); Pecharsky & Zavalij Ch. 2.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.structure import Structure
from kaisetsu.errors import PatternSizeError
from kaisetsu.pattern.instrument import EIGHT_LN2, CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.profile.corrections import (
    axis_cos2,
    cylinder_absorption,
    lorentz_factor,
    march_dollase,
    polarization_factor,
    sabine_extinction,
    surface_roughness_suortti,
)
from kaisetsu.profile.fcj_batch import fcj_profile_batch
from kaisetsu.profile.pseudo_voigt import tch_fwhm_eta
from kaisetsu.profile.stephens import stephens_fwhm
from kaisetsu.profile.texture import symmetrized_harmonics, texture_factor
from kaisetsu.reflections.hkl import ReflectionList, generate_reflections
from kaisetsu.reflections.structure_factor import powder_structure_factor_squared
from kaisetsu.scattering.xray import TableName

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class PhaseContribution:
    """パターンに寄与する 1 相とそのパラメータ。

    Attributes:
        structure: 結晶構造
        scale: スケール因子(1/V を陽に含む定義。GSAS-II の S_hist × S_phase / 100 に相当)
        march_dollase_r, march_dollase_axis: March-Dollase 選択配向(r = 1 で無し)
        size_um: 結晶子サイズ [μm]。None で広がりなし。
            Scherrer 式 Γ_size = (180/π) λ / (10^4 p cosθ) [deg](K = 1)
        size_axis, size_axial_um: 一軸異方的なサイズ。size_axial_um が None でなければ、軸
            (逆格子方向 hkl)と反射のなす角 φ で有効サイズを
            1/p(φ) = sqrt(sin²φ / size_um² + cos²φ / size_axial_um²) とする
            (GSAS-II の uniaxial と同じ。size_um は軸に垂直な方向のサイズ)
        strain_e6: 等方的な微小歪み(×10^-6、GSAS-II と同じ定義)。
            Γ_strain = (180/π) (ε × 10^-6) tanθ [deg]
        size_lorentz_fraction, strain_lorentz_fraction: 各広がりのローレンツ成分の割合。
            残りはガウス成分に Γ^2 として加える(GSAS-II と同じ扱い)。
        stephens: Stephens の異方的歪みの係数 S_HKL(15 個、×10⁻¹² Å⁻⁴)。None なら等方的な
            strain_e6 を使う。ローレンツ成分の割合は strain_lorentz_fraction を共用
        sh_order, sh_coefficients: 球面調和関数の集合組織(`profile.texture`)。sh_order が 0 なら
            使わない。係数はラウエ群で不変な調和関数(l = 2, 4, …, sh_order)の順
        extinction: Sabine の粉末消衰係数 E(GSAS-II の Extinction と同じ単位。0 で補正なし)
    """

    structure: Structure
    scale: float = 1.0
    march_dollase_r: float = 1.0
    march_dollase_axis: tuple[int, int, int] = (0, 0, 1)
    size_um: float | None = None
    strain_e6: float = 0.0
    size_lorentz_fraction: float = 1.0
    strain_lorentz_fraction: float = 1.0
    stephens: FloatArray | None = None
    size_axis: tuple[int, int, int] = (0, 0, 1)
    size_axial_um: float | None = None
    sh_order: int = 0
    sh_coefficients: FloatArray | None = None
    extinction: float = 0.0
    #: Le Bail / Pawley 用: 代表 hkl → |F|^2。None なら構造から計算する。
    f2_table: dict[tuple[int, int, int], float] | None = None
    #: f2_table に無い反射に使う |F|^2
    f2_default: float = 1.0

    @property
    def is_uniaxial_size(self) -> bool:
        return self.size_axial_um is not None

    def inverse_size(self, hkl: IntArray | None, n: int) -> FloatArray:
        """反射ごとの 1/p [1/μm](サイズ広がりなしなら 0)。一軸モデルは hkl が要る。"""
        if self.size_um is None or self.size_um <= 0:
            return np.zeros(n)
        if self.size_axial_um is None or hkl is None:
            return np.full(n, 1.0 / self.size_um)
        pa = max(float(self.size_axial_um), 1e-6)
        st = self.structure
        c2 = axis_cos2(hkl, st.lattice, st.spacegroup, self.size_axis)
        return np.asarray(np.sqrt((1.0 - c2) / self.size_um**2 + c2 / pa**2))

    def texture_basis(self) -> tuple[tuple[int, FloatArray], ...]:
        st = self.structure
        return symmetrized_harmonics(st.spacegroup, st.lattice, int(self.sh_order))

    def orientation_factor(
        self,
        hkl: IntArray,
        march_dollase_r: float | None = None,
        sh_coefficients: FloatArray | None = None,
    ) -> FloatArray:
        """選択配向の補正(March-Dollase × 球面調和関数)。引数で値を差し替えられる(微分用)。"""
        st = self.structure
        r = self.march_dollase_r if march_dollase_r is None else march_dollase_r
        po = march_dollase(hkl, st.lattice, st.spacegroup, self.march_dollase_axis, r)
        if self.sh_order > 0:
            coef = self.sh_coefficients if sh_coefficients is None else sh_coefficients
            basis = self.texture_basis()
            if coef is not None and len(coef) == sum(u.shape[1] for _, u in basis):
                po = po * texture_factor(hkl, st.lattice, basis, coef)
        return np.asarray(po)

    def extinction_factor(
        self,
        f2: FloatArray,
        wavelength: npt.ArrayLike,
        two_theta: npt.ArrayLike,
        is_xray: bool,
    ) -> FloatArray:
        """Sabine 消衰補正(反射ごと)。extinction = 0 なら 1。"""
        if self.extinction == 0.0:
            return np.ones(len(f2))
        return sabine_extinction(
            f2, wavelength, self.structure.lattice.volume, two_theta, self.extinction, is_xray
        )

    def sample_broadening(
        self,
        two_theta: FloatArray,
        wavelength: float,
        hkl: IntArray | None = None,
        inv_d2: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """試料由来の広がり。

        Args:
            hkl, inv_d2: Stephens の異方的歪みと一軸サイズに使う反射指数と 1/d²(None なら等方)

        Returns:
            (ガウス FWHM^2 への加算分 [deg^2], ローレンツ FWHM への加算分 [deg])
        """
        th = np.radians(two_theta / 2.0)
        deg = 180.0 / math.pi
        g_size = deg * wavelength * self.inverse_size(hkl, len(two_theta)) / (1.0e4 * np.cos(th))
        if self.stephens is not None and hkl is not None and inv_d2 is not None:
            g_strain = stephens_fwhm(hkl, inv_d2, two_theta, self.stephens)
        else:
            g_strain = deg * (self.strain_e6 * 1.0e-6) * np.tan(th)
        fs, fm = self.size_lorentz_fraction, self.strain_lorentz_fraction
        add_l = fs * g_size + fm * g_strain
        add_g2 = EIGHT_LN2 * (((1.0 - fs) * g_size) ** 2 + ((1.0 - fm) * g_strain) ** 2)
        return np.asarray(add_g2), np.asarray(add_l)

    @classmethod
    def from_gsas2(
        cls, structure: Structure, phase_hist: Mapping[str, Any], hist_scale: float = 1.0
    ) -> PhaseContribution:
        """GSAS-II の相-ヒストグラム辞書(Scale, Pref.Ori., Size, Mustrain)から構築する。"""
        scale = float(phase_hist.get("Scale", [1.0])[0]) * hist_scale / 100.0
        po = phase_hist.get("Pref.Ori.", ["MD", 1.0, False, [0, 0, 1]])
        r = 1.0
        axis: tuple[int, int, int] = (0, 0, 1)
        if po and po[0] == "MD":
            r = float(po[1])
            axis = (int(po[3][0]), int(po[3][1]), int(po[3][2]))
        size = phase_hist.get("Size", ["isotropic", [1.0, 1.0, 1.0]])
        mus = phase_hist.get("Mustrain", ["isotropic", [1000.0, 1000.0, 1.0]])
        if size[0] not in ("isotropic", "uniaxial") or mus[0] != "isotropic":
            raise NotImplementedError("GSAS-II の楕円体サイズ・異方的歪みは未対応です")
        pc = cls(
            structure,
            scale=scale,
            march_dollase_r=r,
            march_dollase_axis=axis,
            size_um=float(size[1][0]),
            strain_e6=float(mus[1][0]),
            size_lorentz_fraction=float(size[1][2]),
            strain_lorentz_fraction=float(mus[1][2]),
        )
        if size[0] == "uniaxial":
            pc.size_axial_um = float(size[1][1])
            ax = size[3] if len(size) > 3 else [0, 0, 1]
            pc.size_axis = (int(ax[0]), int(ax[1]), int(ax[2]))
        return pc


@dataclass
class PeakTable:
    """計算に使った反射ごとの量(相ごと)。"""

    hkl: IntArray
    d: FloatArray
    multiplicity: IntArray
    two_theta: FloatArray  # 補正済みピーク位置 (λ1)
    f2: FloatArray  # |F|^2
    intensity: FloatArray  # 積分強度(λ1 成分、プロファイル面積 1 に対する係数)
    fwhm: FloatArray  # 全 FWHM [deg]
    fwhm_gauss: FloatArray
    fwhm_lorentz: FloatArray
    #: 全波長成分を合わせた積分強度(λ1 成分 × (1 + Kα2 比))
    intensity_total: FloatArray = field(default_factory=lambda: np.zeros(0))
    #: 反射ごとの寄与の窓 [start, stop) と寄与配列(全波長成分の和)
    window_start: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    window_stop: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contributions: list[FloatArray] = field(default_factory=list)

    def windows(self) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for a, b in zip(self.window_start, self.window_stop, strict=True)]


@dataclass
class PatternResult:
    x: FloatArray
    y_calc: FloatArray
    y_background: FloatArray
    y_phases: list[FloatArray]
    peaks: list[PeakTable] = field(default_factory=list)

    @property
    def y_bragg(self) -> FloatArray:
        return np.asarray(self.y_calc - self.y_background)


def default_d_min(x_max: float, wavelength: float, margin_deg: float = 0.0) -> float:
    """2θ の上限から d_min を決める。"""
    tt = min(x_max + margin_deg, 179.9)
    return wavelength / (2.0 * math.sin(math.radians(tt / 2.0)))


def default_d_max(x_min: float, wavelength: float) -> float:
    tt = max(x_min, 0.01)
    return wavelength / (2.0 * math.sin(math.radians(tt / 2.0)))


#: ピーク窓の点数の合計(全相・全波長)の上限。幅が発散したときにメモリを使い尽くす前に
#: 止める(1e8 点で寄与配列 1 本が 800 MB。正常なら放射光の大きな格子でも 1e6 程度)
MAX_WINDOW_POINTS = 100_000_000


def window_size_message(
    n: int, limit: int, n_refl: int, n_points: int, width: FloatArray, unit: str
) -> str:
    """ピーク窓が上限を超えたときの説明(`PatternSizeError` の本文)。"""
    med = float(np.median(width)) if len(width) else float("nan")
    return (
        f"ピーク窓の点数の合計が {n:.2e} 点になり上限 {limit:.1e} 点を超えました"
        f"(反射 {n_refl} 本、データ {n_points} 点、幅の中央値 {med:.3g} {unit})。"
        "ピーク幅(U, V, W, X, Y や試料の幅)が発散している可能性があります。"
        "幅のパラメータを確かめるか固定してください"
    )


def peak_window(
    pos: float, fwhm: float, s_l: float, h_l: float, low_fwhm: float, high_fwhm: float
) -> tuple[float, float]:
    """ピークを計算する 2θ 範囲(GSAS-II と同じ規則: 低角側 50 幅 + 軸発散分、高角側 75 幅)。

    幅には Γ_G + Γ_L(TCH の全 FWHM より少し大きい)を使う。GSAS-II と同じ。
    """
    fwhm = max(fwhm, 1e-4)
    lo = low_fwhm * (fwhm + (s_l + h_l) * abs(math.cos(math.radians(pos))))
    hi = high_fwhm * fwhm
    if pos > 90.0:
        lo, hi = hi, lo
    return pos - lo, pos + hi


def calculate_pattern(
    x: npt.ArrayLike,
    phases: Sequence[PhaseContribution],
    instrument: CWInstrument,
    sample: SampleGeometry | None = None,
    background: Callable[[FloatArray], FloatArray] | None = None,
    *,
    d_min: float | None = None,
    d_max: float | None = None,
    window_fwhm: tuple[float, float] = (50.0, 75.0),
    kalpha2_mode: str = "exact",
    xray_table: TableName = "it1992",
    anomalous: Mapping[str, complex] | None = None,
    scattering_override: Mapping[str, complex] | None = None,
    fcj_method: str = "accurate",
    max_window_points: int = MAX_WINDOW_POINTS,
) -> PatternResult:
    """パターンを計算する。

    Args:
        x: 2θ [deg] の配列(昇順)
        phases: 相のリスト
        instrument: 装置パラメータ
        sample: 試料形状(None なら補正なしの Bragg-Brentano)
        background: 2θ 配列 → バックグラウンドの関数(None なら 0)
        d_min, d_max: 反射生成の範囲。None なら x の範囲と λ1 から決める。
        window_fwhm: ピーク計算範囲(低角側, 高角側)を FWHM の倍数で指定
        kalpha2_mode: 'exact' は λ2 から Bragg 式で位置を計算、
            'gsas' は GSAS-II と同じ近似 2θ2 = 2θ1 + (360/π)(Δλ/λ) tanθ
        xray_table: X 線 f0 テーブル
        anomalous: 散乱種 → f' + i f'' の上書き(検証用)
        scattering_override: 散乱種 → 散乱因子そのものの上書き(検証用、中性子の b など)
        max_window_points: ピーク窓の点数の合計の上限。超えたら `PatternSizeError`
            (ピーク幅の発散)
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    if sample is None:
        sample = SampleGeometry()
    rad = instrument.radiation
    lam1 = rad.wavelength
    dmin = d_min if d_min is not None else default_d_min(float(xa[-1]), lam1)
    dmax = d_max if d_max is not None else default_d_max(float(xa[0]), lam1)

    y_bkg = (
        np.asarray(background(xa), dtype=np.float64)
        if background is not None
        else np.zeros_like(xa)
    )
    y_total = y_bkg.copy()
    y_phases: list[FloatArray] = []
    peak_tables: list[PeakTable] = []
    n_window = 0

    for ph in phases:
        st = ph.structure
        refl: ReflectionList = generate_reflections(st.lattice, st.spacegroup, dmin, dmax)
        yp = np.zeros_like(xa)
        if len(refl) == 0:
            y_phases.append(yp)
            peak_tables.append(_empty_peak_table())
            continue
        if ph.f2_table is not None:
            f2 = np.array(
                [ph.f2_table.get((int(h), int(k), int(l)), ph.f2_default) for h, k, l in refl.hkl]
            )
        else:
            # Friedel 対の平均(非中心対称 + 異常分散で代表 1 本の |F|² と差が出る)
            f2 = powder_structure_factor_squared(
                st,
                refl.hkl,
                rad,
                xray_table=xray_table,
                anomalous=anomalous,
                override=scattering_override,
            )
        volume = st.lattice.volume
        n_refl = len(refl)
        w_start = np.zeros(n_refl, dtype=np.int64)
        w_stop = np.zeros(n_refl, dtype=np.int64)
        contrib: list[FloatArray | None] = [None] * n_refl
        inten_total = np.zeros(n_refl)

        po = ph.orientation_factor(refl.hkl)

        table: PeakTable | None = None
        corr1: FloatArray | None = None
        fg1: FloatArray | None = None
        fl1: FloatArray | None = None
        for wl, ratio in rad.wavelengths:
            reuse = kalpha2_mode == "gsas" and wl != lam1 and corr1 is not None
            if reuse:
                # GSAS-II 流: Kα1 の補正位置に (360/π)(Δλ/λ) tanθ を足し、
                # 強度補正と幅は Kα1 の位置で求めたものを使い回す
                pos1 = _corrected_position(_bragg_two_theta(refl.d, lam1), instrument, sample)
                pos = pos1 + (360.0 / math.pi) * ((wl - lam1) / lam1) * np.tan(
                    np.radians(pos1 / 2.0)
                )
                assert corr1 is not None and fg1 is not None and fl1 is not None
                corr, fg, fl = corr1, fg1, fl1
            else:
                pos = _corrected_position(_bragg_two_theta(refl.d, wl), instrument, sample)
                lp = lorentz_factor(pos)
                if rad.is_xray:
                    lp = lp * polarization_factor(pos, instrument.polarization)
                if sample.kind is GeometryKind.DEBYE_SCHERRER:
                    absorb = cylinder_absorption(sample.mu_r, pos)
                else:
                    absorb = surface_roughness_suortti(sample.roughness_a, sample.roughness_b, pos)
                corr = lp * absorb
                add_g2, add_l = ph.sample_broadening(pos, wl, refl.hkl, 1.0 / (refl.d * refl.d))
                fg = np.sqrt(instrument.fwhm_gauss(pos) ** 2 + add_g2)
                fl = instrument.fwhm_lorentz(pos) + add_l
                if wl == lam1:
                    corr1, fg1, fl1 = corr, fg, fl
            ok = np.isfinite(pos)
            ext = ph.extinction_factor(f2, wl, np.where(ok, pos, 90.0), rad.is_xray)
            inten = ph.scale * ratio / volume * refl.multiplicity * f2 * corr * po * ext
            fwhm = np.array(
                [tch_fwhm_eta(float(a), float(b))[0] for a, b in zip(fg, fl, strict=True)]
            )
            # ピーク窓(GSAS-II と同じ規則)をまとめて決め、全ピークを一括計算する
            pos_safe = np.where(ok, pos, 0.0)
            width = np.maximum(fg + fl, 1e-4)
            lo_w = window_fwhm[0] * (
                width + (instrument.s_l + instrument.h_l) * np.abs(np.cos(np.radians(pos_safe)))
            )
            hi_w = window_fwhm[1] * width
            high = pos_safe > 90.0
            lo_w, hi_w = np.where(high, hi_w, lo_w), np.where(high, lo_w, hi_w)
            i0 = np.searchsorted(xa, pos_safe - lo_w).astype(np.int64)
            i1 = np.searchsorted(xa, pos_safe + hi_w).astype(np.int64)
            use = ok & (inten > 0.0) & (i1 > i0)
            i1 = np.where(use, i1, i0)
            n_window += int(np.sum(i1 - i0))
            if n_window > max_window_points:
                raise PatternSizeError(
                    window_size_message(n_window, max_window_points, n_refl, len(xa), width, "°")
                )
            out, offsets, lengths = fcj_profile_batch(
                xa,
                pos_safe,
                fg,
                fl,
                inten,
                i0,
                i1,
                instrument.s_l,
                instrument.h_l,
                method=fcj_method,
            )
            for i in np.nonzero(lengths > 0)[0]:
                a, b = int(i0[i]), int(i1[i])
                part = out[offsets[i] : offsets[i] + lengths[i]]
                yp[a:b] += part
                inten_total[i] += inten[i]
                prev = contrib[i]
                if prev is None:
                    w_start[i], w_stop[i] = a, b
                    contrib[i] = part.copy()
                else:
                    s0, s1 = min(int(w_start[i]), a), max(int(w_stop[i]), b)
                    merged = np.zeros(s1 - s0)
                    merged[w_start[i] - s0 : w_stop[i] - s0] += prev
                    merged[a - s0 : b - s0] += part
                    w_start[i], w_stop[i] = s0, s1
                    contrib[i] = merged
            if table is None:
                table = PeakTable(
                    hkl=refl.hkl,
                    d=refl.d,
                    multiplicity=refl.multiplicity,
                    two_theta=pos,
                    f2=f2,
                    intensity=inten,
                    fwhm=fwhm,
                    fwhm_gauss=fg,
                    fwhm_lorentz=fl,
                )
        if table is not None:
            table.intensity_total = inten_total
            table.window_start = w_start
            table.window_stop = w_stop
            table.contributions = [c if c is not None else np.zeros(0) for c in contrib]
        y_phases.append(yp)
        y_total += yp
        peak_tables.append(table if table is not None else _empty_peak_table())

    return PatternResult(
        x=xa, y_calc=y_total, y_background=y_bkg, y_phases=y_phases, peaks=peak_tables
    )


def _bragg_two_theta(d: FloatArray, wavelength: float) -> FloatArray:
    s = wavelength / (2.0 * d)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.degrees(2.0 * np.arcsin(np.where(s <= 1.0, s, np.nan))))


def _corrected_position(
    pos0: FloatArray, instrument: CWInstrument, sample: SampleGeometry
) -> FloatArray:
    return np.asarray(pos0 + sample.position_shift(pos0) + instrument.zero)


def _empty_peak_table() -> PeakTable:
    z = np.zeros(0)
    return PeakTable(
        np.zeros((0, 3), dtype=np.int64), z, np.zeros(0, dtype=np.int64), z, z, z, z, z, z
    )
