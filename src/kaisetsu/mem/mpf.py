"""Rietveld 結果から MEM の入力を作る。MEM パターンフィッティング(MPF)。

観測構造因子: Rietveld 分配で得た "観測" 積分強度 I_obs,k から
    |F_obs|² = |F_calc|² I_obs / I_calc、位相はモデル(初回)または前回の MEM 密度(2 回目以降)から。
    σ(|F|) は I_obs の誤差伝播 σ(I_obs) から(σ(F²) = F²_calc σ(I)/I_calc、σ(|F|) = σ(F²)/(2|F|))。

MPF(Takata, Nishibori & Sakata, Z. Kristallogr. 216, 71 (2001)):
    1. Rietveld 精密化 → F_obs(位相はモデル)
    2. MEM → 密度 ρ_MEM → F_MEM(h)
    3. |F_MEM|² を固定した構造因子表としてパターンフィッティング(スケール・背景・格子・プロファイル)
       → 重なった反射を F_MEM の比で再分配した新しい F_obs(位相は F_MEM)
    4. 2 に戻る。Rwp と密度の変化が小さくなるまで繰り返す。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from kaisetsu.mem.core import MEMInput, MEMResult, run_mem
from kaisetsu.pdf.calculator import scattering_lengths
from kaisetsu.refine.project import Project
from kaisetsu.refine.rfactors import (
    partition_intensities,
    partition_uncertainties,
    reflections_inside,
)
from kaisetsu.reflections.structure_factor import structure_factors
from kaisetsu.scattering.xray import XrayScatteringFactors

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ComplexArray = npt.NDArray[np.complex128]

HklKey = tuple[int, int, int]


@dataclass
class ObservedStructureFactors:
    hkl: IntArray
    f_obs: ComplexArray
    sigma: FloatArray
    f2_calc: FloatArray
    f000: float
    f000_negative: float
    radiation: str


def total_scattering_f000(project: Project, phase: str, hist: int) -> tuple[float, float]:
    """F(000) の正側・負側の総量(X 線: 電子数、中性子: 散乱長 [fm])。"""
    ph = project.phases[phase]
    ph.sync()
    st = ph.structure
    ex = st.expand()
    rad = project.histograms[hist].instrument.radiation
    kind = "neutron" if rad.is_neutron else "xray"
    b = scattering_lengths(ex.species, kind)[ex.species_index] * ex.occupancy
    pos = float(np.sum(b[b > 0]))
    neg = float(-np.sum(b[b < 0]))
    return pos, neg


def observed_structure_factors(
    project: Project,
    hist: int,
    phase: str,
    *,
    phases: Mapping[HklKey, complex] | None = None,
    min_rel_sigma: float = 0.01,
    min_abs_sigma: float | None = None,
) -> ObservedStructureFactors:
    """Rietveld 分配から MEM 用の観測構造因子を作る。

    Args:
        phases: 代表 hkl → 位相を取る複素数(None ならモデルの構造因子。異常分散の虚部は除く)
        min_rel_sigma: σ(|F|) の下限(|F| に対する比)
        min_abs_sigma: σ(|F|) の絶対値の下限(None なら |F| の最大値の 1e-3)
    """
    h = project.histograms[hist]
    res = project.calculate(hist)
    names = list(h.contributions)
    k = names.index(phase)
    tbl = res.peaks[k]
    # フィット範囲の点だけで分配し、中心が範囲外の反射は計算値 ± 100% にする
    mask = h.mask()
    inside = reflections_inside(h.data.x, tbl.two_theta, mask)
    i_obs = partition_intensities(
        h.data.y,
        res.y_calc,
        res.y_background,
        tbl.windows(),
        tbl.contributions,
        tbl.intensity_total,
        mask=mask,
        inside=inside,
    )
    sig_i = partition_uncertainties(
        h.data.sigma, res.y_calc, res.y_background, tbl.windows(), tbl.contributions,
        tbl.intensity_total, mask=mask, inside=inside,
    )  # fmt: skip
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tbl.intensity_total > 0, i_obs / tbl.intensity_total, 0.0)
        ratio_s = np.where(tbl.intensity_total > 0, sig_i / tbl.intensity_total, 0.0)
    f2_obs = np.maximum(tbl.f2 * ratio, 0.0)
    f_abs = np.sqrt(f2_obs)
    sig_f2 = tbl.f2 * ratio_s
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_f = np.where(f_abs > 0, sig_f2 / (2.0 * f_abs), np.sqrt(np.maximum(sig_f2, 0.0)))
    fmax = float(f_abs.max()) if len(f_abs) else 1.0
    floor_abs = min_abs_sigma if min_abs_sigma is not None else 1e-3 * fmax
    sig_f = np.maximum(np.maximum(sig_f, min_rel_sigma * f_abs), floor_abs)

    ph = project.phases[phase]
    ph.sync()
    rad = h.instrument.radiation
    if phases is None:
        anomalous = h.anomalous
        if rad.is_xray and anomalous is None:
            xt = XrayScatteringFactors()
            anomalous = {
                sp: complex(xt.anomalous(sp, rad.wavelength).real, 0.0)
                for sp in ph.structure.species_list()
            }
        fc = structure_factors(
            ph.structure, tbl.hkl, rad, anomalous=anomalous, override=h.scattering_override
        )
        unit = np.where(np.abs(fc) > 0, fc / np.maximum(np.abs(fc), 1e-300), 1.0 + 0j)
    else:
        unit = np.array(
            [_unit(phases.get((int(a), int(b), int(c)), 1.0 + 0j)) for a, b, c in tbl.hkl],
            dtype=np.complex128,
        )
    f_obs = f_abs * unit
    pos, neg = total_scattering_f000(project, phase, hist)
    return ObservedStructureFactors(
        hkl=tbl.hkl.copy(),
        f_obs=f_obs,
        sigma=sig_f,
        f2_calc=tbl.f2.copy(),
        f000=pos - neg,
        f000_negative=neg,
        radiation="neutron" if rad.is_neutron else "xray",
    )


def _unit(z: complex) -> complex:
    a = abs(z)
    return z / a if a > 0 else 1.0 + 0j


def overlap_groups_from_project(
    project: Project, hist: int, phase: str, overlap_fwhm: float = 0.5
) -> list[tuple[IntArray, float, float, IntArray]]:
    """ピーク位置の差が FWHM × overlap_fwhm 以内で連なる反射を群にまとめる(G 拘束用)。

    群の観測値 G_obs = Σ_h m_h |F_obs,h|²(Rietveld 分配の値の和。分配の仮定によらず群の和は決まる)、
    σ_G は各反射の σ(|F|²) m_h を二乗和した値。
    """
    h = project.histograms[hist]
    res = project.calculate(hist)
    k = list(h.contributions).index(phase)
    tbl = res.peaks[k]
    pos = tbl.two_theta
    order = np.argsort(pos)
    groups: list[list[int]] = []
    current: list[int] = []
    for j in order:
        if not np.isfinite(pos[j]):
            continue
        if current:
            prev = current[-1]
            if abs(pos[j] - pos[prev]) <= overlap_fwhm * 0.5 * (tbl.fwhm[j] + tbl.fwhm[prev]):
                current.append(int(j))
                continue
            if len(current) > 1:
                groups.append(current)
        current = [int(j)]
    if len(current) > 1:
        groups.append(current)
    if not groups:
        return []
    obs = observed_structure_factors(project, hist, phase)
    f_abs = np.abs(obs.f_obs)
    sig_f2 = 2.0 * f_abs * obs.sigma
    out: list[tuple[IntArray, float, float, IntArray]] = []
    for g in groups:
        idx = np.asarray(g, dtype=np.int64)
        mult = tbl.multiplicity[idx].astype(np.int64)
        g_obs = float(np.sum(mult * f_abs[idx] ** 2))
        sig_g = float(np.sqrt(np.sum((mult * sig_f2[idx]) ** 2)))
        out.append((idx, g_obs, max(sig_g, 1e-3 * max(g_obs, 1.0)), mult))
    return out


def mem_from_project(
    project: Project,
    hist: int,
    phase: str,
    grid: tuple[int, int, int] = (64, 64, 64),
    *,
    two_channel: bool | None = None,
    phases: Mapping[HklKey, complex] | None = None,
    min_rel_sigma: float = 0.01,
    group_overlaps: bool = False,
    overlap_fwhm: float = 0.5,
    **mem_kwargs: Any,
) -> MEMResult:
    """Rietveld 結果(現在のモデルと観測)から MEM 密度を求める。

    two_channel を省略すると、負の散乱長があるとき(中性子)だけ 2 チャンネルにする。
    group_overlaps が真なら、重なった反射(位置の差が FWHM × overlap_fwhm 以内)を G 拘束にする。
    """
    obs = observed_structure_factors(
        project, hist, phase, phases=phases, min_rel_sigma=min_rel_sigma
    )
    ph = project.phases[phase]
    two = two_channel if two_channel is not None else obs.f000_negative > 0
    groups = (
        overlap_groups_from_project(project, hist, phase, overlap_fwhm) if group_overlaps else []
    )
    inp = MEMInput(
        lattice=ph.structure.lattice,
        spacegroup=ph.structure.spacegroup,
        hkl=obs.hkl,
        f_obs=obs.f_obs,
        sigma=obs.sigma,
        f000=obs.f000,
        grid=grid,
        two_channel=two,
        f000_negative=obs.f000_negative if two else 0.0,
        overlap_groups=groups,
    )
    return run_mem(inp, **mem_kwargs)


@dataclass
class MPFResult:
    densities: list[MEMResult]
    rwp_history: list[float]
    chi2_history: list[float]
    f_mem: dict[HklKey, complex]
    log: list[str] = field(default_factory=list)

    @property
    def final(self) -> MEMResult:
        return self.densities[-1]


def run_mpf(
    project: Project,
    hist: int,
    phase: str,
    *,
    grid: tuple[int, int, int] = (64, 64, 64),
    n_cycles: int = 3,
    fit_patterns: Sequence[str] = ("h{i}:bkg:*", "h{i}:{p}:scale", "{p}:cell:*"),
    max_cycles_fit: int = 10,
    rwp_tol: float = 1e-4,
    progress: Callable[[str], None] | None = None,
    **mem_kwargs: Any,
) -> MPFResult:
    """MEM パターンフィッティングを行う。

    Args:
        grid: MEM の格子点数
        n_cycles: MEM ↔ パターンフィッティングの往復回数の上限
        fit_patterns: パターンフィッティングで動かすパラメータ(h{i}, {p} は置き換える)
        rwp_tol: Rwp の変化がこれ以下なら終了
        mem_kwargs: run_mem に渡す(method, max_outer, target_chi2 など)

    終了後、相は |F_MEM|² を固定した表(le_bail=True, f2_update=False)のままになる。
    モデルに戻すには ``project.phases[phase].le_bail = False``。
    """
    ph = project.phases[phase]
    log: list[str] = []

    def say(msg: str) -> None:
        log.append(msg)
        if progress is not None:
            progress(msg)

    rwp0 = project.rfactors()[project.histograms[hist].name].rwp
    say(f"MPF 開始: Rwp = {100 * rwp0:.3f}%(Rietveld)")
    densities: list[MEMResult] = []
    rwp_hist = [rwp0]
    chi2_hist: list[float] = []
    f_mem: dict[HklKey, complex] = {}
    phases: Mapping[HklKey, complex] | None = None
    saved_vary = {p.name: p.vary for p in project.params}
    for cycle in range(1, n_cycles + 1):
        mem = mem_from_project(project, hist, phase, grid, phases=phases, **mem_kwargs)
        densities.append(mem)
        chi2_hist.append(mem.chi2)
        say(
            f"サイクル {cycle}: MEM C = {mem.chi2:.3f}, R_F(MEM) = {100 * mem.r_factor():.2f}%"
            f"({mem.message})"
        )
        f_mem = {
            (int(a), int(b), int(c)): complex(f)
            for (a, b, c), f in zip(mem.hkl, mem.f_calc, strict=True)
        }
        # |F_MEM|² を固定した構造因子表でパターンフィッティング
        ph.le_bail = True
        ph.f2_update = False
        ph.f2_tables[hist] = {k: abs(v) ** 2 for k, v in f_mem.items()}
        project.fix_all()
        pats = [s.format(i=hist, p=phase) for s in fit_patterns]
        for pat in pats:
            if project.params.select(pat):
                project.vary(pat)
        if project.params.free():
            project.refine(max_cycles=max_cycles_fit)
        rwp = project.rfactors()[project.histograms[hist].name].rwp
        rwp_hist.append(rwp)
        say(f"サイクル {cycle}: パターンフィッティング Rwp = {100 * rwp:.3f}%")
        phases = f_mem
        if abs(rwp_hist[-2] - rwp) < rwp_tol and cycle > 1:
            say("Rwp の変化が小さいので終了")
            break
    for p in project.params:
        if p.name in saved_vary and p.tie is None and not p.locked:
            p.vary = saved_vary[p.name]
    return MPFResult(densities, rwp_hist, chi2_hist, f_mem, log)


def phase_angle_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real))
