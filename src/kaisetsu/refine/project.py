"""精密化プロジェクト: 相・ヒストグラム・パラメータをまとめ、計算と精密化を行う。

使い方(スクリプト):

    proj = Project("PbSO4")
    proj.add_phase(Structure.from_cif("PbSO4.cif"), name="PbSO4")
    proj.add_histogram(read_powder_data("PBSO4.XRA"), CWInstrument(...), SampleGeometry(),
                       n_background=6)
    proj.vary("h0:bkg:*", "h0:PbSO4:scale")
    proj.refine()
    proj.vary("PbSO4:cell:*", "h0:inst:zero")
    proj.refine()
    print(proj.summary())

パラメータ名の規則:
    {相}:cell:{a|b|c|alpha|beta|gamma}      格子定数
    {相}:{サイト}:{x|y|z|occ|uiso|u11..u23}  原子
    h{i}:inst:{zero|u|v|w|p|x|y|z|s_l|h_l|pol}  装置
    h{i}:sample:{displacement|mu_eff|rough_a|rough_b|disp_x|disp_y|mu_r}  試料
    h{i}:bkg:{k}                           バックグラウンド係数
    h{i}:{相}:{scale|md_r|size|strain}     相-ヒストグラム
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import CrystalSystem, Lattice
from kaisetsu.crystal.structure import Structure
from kaisetsu.pattern.calculator import PatternResult, PhaseContribution, calculate_pattern
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.profile.background import ChebyshevBackground
from kaisetsu.refine.constraints import (
    SymmetryConstraint,
    adp_constraint,
    lattice_constraint,
    lattice_parameter_names,
    position_constraint,
)
from kaisetsu.refine.lsq import FitResult, refine_least_squares
from kaisetsu.refine.parameters import Parameter, ParameterSet
from kaisetsu.refine.rfactors import (
    RFactors,
    bragg_rfactor,
    partition_intensities,
    powder_rfactors,
    structure_rfactor,
)
from kaisetsu.reflections.hkl import canonical_hkl
from kaisetsu.reflections.structure_factor import structure_factors

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

log = logging.getLogger(__name__)

_ADP_NAMES = ("u11", "u22", "u33", "u12", "u13", "u23")
_XYZ = ("x", "y", "z")


def _attr_setter(obj: object, attr: str) -> Callable[[float], None]:
    def _set(v: float) -> None:
        setattr(obj, attr, v)

    return _set


def _item_setter(arr: Any, k: int) -> Callable[[float], None]:
    def _set(v: float) -> None:
        arr[k] = v

    return _set


def _cell_setter(ph: PhaseEntry, k: int) -> Callable[[float], None]:
    def _set(v: float) -> None:
        ph.set_cell(k, v)

    return _set


def _size_setter(pc: PhaseContribution) -> Callable[[float], None]:
    def _set(v: float) -> None:
        pc.size_um = v if v > 0 else None

    return _set


@dataclass
class PhaseEntry:
    """プロジェクト内の 1 相。"""

    name: str
    structure: Structure
    le_bail: bool = False
    #: Le Bail 用の抽出強度(ヒストグラム添字 → 代表 hkl → |F|^2)
    f2_tables: dict[int, dict[tuple[int, int, int], float]] = field(default_factory=dict)
    position_constraints: dict[str, SymmetryConstraint] = field(default_factory=dict)
    adp_constraints: dict[str, SymmetryConstraint] = field(default_factory=dict)
    _cell: FloatArray = field(default_factory=lambda: np.zeros(6), repr=False)
    _cell_dirty: bool = field(default=False, repr=False)

    def set_cell(self, k: int, value: float) -> None:
        self._cell[k] = value
        self._cell_dirty = True

    def sync(self) -> None:
        if self._cell_dirty:
            self.structure.lattice = Lattice.from_parameters(self._cell)
            self._cell_dirty = False


@dataclass
class HistogramEntry:
    """プロジェクト内の 1 ヒストグラム(観測データ + 装置 + 試料 + 背景 + 相の寄与)。"""

    name: str
    data: PowderData
    instrument: CWInstrument
    sample: SampleGeometry
    background: ChebyshevBackground
    contributions: dict[str, PhaseContribution]
    x_min: float
    x_max: float
    excluded: list[tuple[float, float]] = field(default_factory=list)
    weight_factor: float = 1.0
    kalpha2_mode: str = "exact"
    fcj_method: str = "accurate"
    d_min: float | None = None
    d_min_margin_deg: float = 2.0
    anomalous: dict[str, complex] | None = None
    scattering_override: dict[str, complex] | None = None
    result: PatternResult | None = None

    def mask(self) -> BoolArray:
        x = self.data.x
        assert self.data.weight is not None
        m = (x >= self.x_min) & (x <= self.x_max) & (self.data.weight > 0)
        for lo, hi in self.excluded:
            m &= ~((x >= lo) & (x <= hi))
        return m

    def effective_d_min(self) -> float:
        if self.d_min is not None:
            return self.d_min
        lam = self.instrument.radiation.wavelength
        tt = min(self.x_max + self.d_min_margin_deg, 179.0)
        return lam / (2.0 * math.sin(math.radians(tt / 2.0)))


@dataclass
class ReflectionReport:
    """反射ごとの計算・観測量。"""

    hkl: npt.NDArray[np.int64]
    d: FloatArray
    two_theta: FloatArray
    multiplicity: npt.NDArray[np.int64]
    f2_calc: FloatArray
    f2_obs: FloatArray
    i_calc: FloatArray
    i_obs: FloatArray
    fwhm: FloatArray
    r_bragg: float
    r_f: float


class Project:
    """Rietveld 精密化プロジェクト。"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.phases: dict[str, PhaseEntry] = {}
        self.histograms: list[HistogramEntry] = []
        self.params = ParameterSet()
        self.last_fit: FitResult | None = None
        self.log: list[str] = []

    # ------------------------------------------------------------------
    # 構築
    # ------------------------------------------------------------------
    def add_phase(
        self, structure: Structure, name: str | None = None, le_bail: bool = False
    ) -> PhaseEntry:
        name = name or structure.name or f"phase{len(self.phases) + 1}"
        if name in self.phases:
            raise ValueError(f"相の名前が重複しています: {name}")
        if ":" in name:
            raise ValueError("相の名前に ':' は使えません")
        entry = PhaseEntry(name=name, structure=structure, le_bail=le_bail)
        entry._cell = structure.lattice.parameters.copy()
        self.phases[name] = entry
        for h in self.histograms:
            h.contributions[name] = PhaseContribution(structure)
        self.build_parameters()
        return entry

    def add_histogram(
        self,
        data: PowderData,
        instrument: CWInstrument,
        sample: SampleGeometry | None = None,
        background: ChebyshevBackground | None = None,
        *,
        n_background: int = 6,
        name: str | None = None,
        phases: Iterable[str] | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        excluded: Sequence[tuple[float, float]] = (),
        kalpha2_mode: str = "exact",
        weight_factor: float = 1.0,
    ) -> HistogramEntry:
        if background is None:
            coeffs = np.zeros(max(n_background, 1))
            coeffs[0] = float(np.median(data.y))
            background = ChebyshevBackground(coeffs, float(data.x[0]), float(data.x[-1]))
        names = list(phases) if phases is not None else list(self.phases)
        for n in names:
            if n not in self.phases:
                raise KeyError(f"相 {n!r} はプロジェクトにありません")
        contributions = {n: PhaseContribution(self.phases[n].structure) for n in names}
        h = HistogramEntry(
            name=name or data.name or f"h{len(self.histograms)}",
            data=data,
            instrument=instrument,
            sample=sample if sample is not None else SampleGeometry(),
            background=background,
            contributions=contributions,
            x_min=float(data.x[0]) if x_min is None else x_min,
            x_max=float(data.x[-1]) if x_max is None else x_max,
            excluded=list(excluded),
            weight_factor=weight_factor,
            kalpha2_mode=kalpha2_mode,
        )
        self.histograms.append(h)
        self.build_parameters()
        return h

    def phase_contribution(self, hist: int, phase: str) -> PhaseContribution:
        return self.histograms[hist].contributions[phase]

    # ------------------------------------------------------------------
    # パラメータ
    # ------------------------------------------------------------------
    def build_parameters(self) -> ParameterSet:
        """モデルからパラメータ集合を(再)構築する。既存の vary フラグは名前が同じなら引き継ぐ。"""
        old = self.params
        ps = ParameterSet()
        for ph in self.phases.values():
            self._build_phase_params(ps, ph)
        for i, h in enumerate(self.histograms):
            self._build_hist_params(ps, i, h)
        for p in ps:
            if p.name in old:
                op = old[p.name]
                if p.tie is None and not p.locked:
                    p.vary = op.vary
                if op.tie is not None and p.tie is None and all(m in ps for m in op.tie.masters):
                    ps.tie(p.name, op.tie.masters, op.tie.coeffs, op.tie.offset)
        self.params = ps
        return ps

    def _build_phase_params(self, ps: ParameterSet, ph: PhaseEntry) -> None:
        st = ph.structure
        pre = ph.name
        # 格子
        system = st.spacegroup.crystal_system
        unique = "b"
        gsg = st.spacegroup.gemmi_spacegroup
        if system is CrystalSystem.MONOCLINIC and gsg is not None:
            try:
                unique = str(gsg.monoclinic_unique_axis()) or "b"
            except Exception:
                unique = "b"
        rhombo = system is CrystalSystem.TRIGONAL and abs(st.lattice.gamma - 120.0) > 1e-3
        lc = lattice_constraint(system, unique, rhombo)
        ph._cell = st.lattice.parameters.copy()
        names = lattice_parameter_names()
        for k, nm in enumerate(names):
            unit = "Å" if k < 3 else "deg"
            p = Parameter(
                f"{pre}:cell:{nm}",
                float(ph._cell[k]),
                step=1e-5 if k < 3 else 1e-4,
                min=0.1 if k < 3 else 10.0,
                max=1000.0 if k < 3 else 170.0,
                setter=_cell_setter(ph, k),
                unit=unit,
                description=f"格子定数 {nm}",
            )
            if k in lc.fixed:
                p.locked = True
            ps.add(p)
        for dep, master in lc.ties.items():
            ps.tie(f"{pre}:cell:{names[dep]}", f"{pre}:cell:{names[master]}")
        # 原子
        for site in st.sites:
            spre = f"{pre}:{site.label}"
            pc = position_constraint(st.spacegroup, site.position)
            if not np.allclose(pc.symmetrized, site.position, atol=1e-6):
                log.info(
                    "%s の座標を特殊位置に乗せ直しました: %s -> %s",
                    spre,
                    site.position,
                    pc.symmetrized,
                )
            site.position[:] = pc.symmetrized
            ph.position_constraints[site.label] = pc
            for k, nm in enumerate(_XYZ):
                p = Parameter(
                    f"{spre}:{nm}",
                    float(site.position[k]),
                    step=5e-5,
                    setter=_item_setter(site.position, k),
                    description=f"{site.label} の分率座標 {nm}",
                )
                if k in pc.fixed:
                    p.locked = True
                ps.add(p)
            for j, rel in pc.dependent.items():
                ps.tie(
                    f"{spre}:{_XYZ[j]}",
                    [f"{spre}:{_XYZ[k]}" for k in rel.pivots],
                    rel.coeffs,
                    rel.offset,
                )
            ps.add(
                Parameter(
                    f"{spre}:occ",
                    site.occupancy,
                    min=0.0,
                    max=1.0,
                    step=1e-3,
                    setter=_attr_setter(site, "occupancy"),
                    description=f"{site.label} の占有率",
                )
            )
            if site.u_aniso is None:
                ps.add(
                    Parameter(
                        f"{spre}:uiso",
                        site.u_iso,
                        min=-0.05,
                        max=2.0,
                        step=2e-4,
                        setter=_attr_setter(site, "u_iso"),
                        unit="Å²",
                        description=f"{site.label} の等方性変位パラメータ U_iso",
                    )
                )
            else:
                ac = adp_constraint(st.spacegroup, site.position, site.u_aniso)
                site.u_aniso[:] = ac.symmetrized
                ph.adp_constraints[site.label] = ac
                for k, nm in enumerate(_ADP_NAMES):
                    p = Parameter(
                        f"{spre}:{nm}",
                        float(site.u_aniso[k]),
                        min=-0.1,
                        max=2.0,
                        step=2e-4,
                        setter=_item_setter(site.u_aniso, k),
                        unit="Å²",
                        description=f"{site.label} の異方性変位パラメータ {nm.upper()}",
                    )
                    if k in ac.fixed:
                        p.locked = True
                    ps.add(p)
                for j, rel in ac.dependent.items():
                    ps.tie(
                        f"{spre}:{_ADP_NAMES[j]}",
                        [f"{spre}:{_ADP_NAMES[k]}" for k in rel.pivots],
                        rel.coeffs,
                        rel.offset,
                    )

    def _build_hist_params(self, ps: ParameterSet, i: int, h: HistogramEntry) -> None:
        pre = f"h{i}"
        inst = h.instrument
        spec: list[tuple[str, str, float, float, float, str, str]] = [
            # (名前, 属性, step, min, max, unit, 説明)
            ("zero", "zero", 1e-4, -5.0, 5.0, "deg", "ゼロ点シフト"),
            ("u", "u", 1e-5, -10.0, 10.0, "deg²", "ガウス幅 U (tan²θ)"),
            ("v", "v", 1e-5, -10.0, 10.0, "deg²", "ガウス幅 V (tanθ)"),
            ("w", "w", 1e-5, -1.0, 10.0, "deg²", "ガウス幅 W"),
            ("p", "p", 1e-5, -1.0, 10.0, "deg²", "ガウス幅 P (1/cos²θ)"),
            ("x", "x", 1e-4, -1.0, 5.0, "deg", "ローレンツ幅 X (tanθ, 歪み)"),
            ("y", "y", 1e-4, -1.0, 5.0, "deg", "ローレンツ幅 Y (1/cosθ, サイズ)"),
            ("z", "z", 1e-4, -1.0, 5.0, "deg", "ローレンツ幅 Z (定数)"),
            ("s_l", "s_l", 1e-4, 0.0, 0.2, "", "軸発散 S/L"),
            ("h_l", "h_l", 1e-4, 0.0, 0.2, "", "軸発散 H/L"),
            ("pol", "polarization", 1e-3, 0.0, 1.0, "", "偏光係数"),
        ]
        for nm, attr, step, lo, hi, unit, desc in spec:
            ps.add(
                Parameter(
                    f"{pre}:inst:{nm}",
                    float(getattr(inst, attr)),
                    step=step,
                    min=lo,
                    max=hi,
                    setter=_attr_setter(inst, attr),
                    unit=unit,
                    description=desc,
                )
            )
        smp = h.sample
        sspec: list[tuple[str, str, float, float, float, str, str]] = [
            ("displacement", "displacement_mm", 1e-4, -5.0, 5.0, "mm", "試料高さずれ (BB)"),
            ("mu_eff", "mu_eff_inv_mm", 1e-3, 0.0, 1e3, "1/mm", "透過補正 μ_eff (BB)"),
            ("rough_a", "roughness_a", 1e-3, 0.0, 1.0, "", "表面粗さ A (Suortti)"),
            ("rough_b", "roughness_b", 1e-3, 0.0, 10.0, "", "表面粗さ B (Suortti)"),
            ("disp_x", "displace_x_mm", 1e-4, -5.0, 5.0, "mm", "試料位置ずれ x (DS)"),
            ("disp_y", "displace_y_mm", 1e-4, -5.0, 5.0, "mm", "試料位置ずれ y (DS)"),
            ("mu_r", "mu_r", 1e-3, 0.0, 20.0, "", "円筒試料の μR (DS)"),
        ]
        for nm, attr, step, lo, hi, unit, desc in sspec:
            ps.add(
                Parameter(
                    f"{pre}:sample:{nm}",
                    float(getattr(smp, attr)),
                    step=step,
                    min=lo,
                    max=hi,
                    setter=_attr_setter(smp, attr),
                    unit=unit,
                    description=desc,
                )
            )
        scale_y = float(np.mean(np.abs(h.data.y))) + 1e-6
        bk = h.background
        for k in range(bk.order):
            ps.add(
                Parameter(
                    f"{pre}:bkg:{k}",
                    float(bk.coefficients[k]),
                    step=1e-3 * scale_y,
                    setter=_item_setter(bk.coefficients, k),
                    description=f"背景 Chebyshev 係数 {k}",
                )
            )
        for pname, pc in h.contributions.items():
            ppre = f"{pre}:{pname}"
            ps.add(
                Parameter(
                    f"{ppre}:scale",
                    pc.scale,
                    min=0.0,
                    step=max(1e-3 * abs(pc.scale), 1e-9),
                    setter=_attr_setter(pc, "scale"),
                    description=f"{pname} のスケール因子",
                )
            )
            ps.add(
                Parameter(
                    f"{ppre}:md_r",
                    pc.march_dollase_r,
                    min=0.1,
                    max=10.0,
                    step=1e-3,
                    setter=_attr_setter(pc, "march_dollase_r"),
                    description=f"{pname} の March-Dollase 配向パラメータ r",
                )
            )
            ps.add(
                Parameter(
                    f"{ppre}:size",
                    pc.size_um if pc.size_um is not None else 0.0,
                    min=0.0,
                    max=100.0,
                    step=1e-3,
                    setter=_size_setter(pc),
                    unit="μm",
                    description=f"{pname} の結晶子サイズ(0 で広がりなし)",
                )
            )
            ps.add(
                Parameter(
                    f"{ppre}:strain",
                    pc.strain_e6,
                    min=0.0,
                    max=1e5,
                    step=1.0,
                    setter=_attr_setter(pc, "strain_e6"),
                    unit="×1e-6",
                    description=f"{pname} の微小歪み",
                )
            )

    def vary(self, *patterns: str) -> list[str]:
        return self.params.vary(*patterns)

    def fix(self, *patterns: str) -> list[str]:
        return self.params.fix(*patterns)

    def fix_all(self) -> None:
        self.params.fix_all()

    def tie(
        self,
        name: str,
        masters: str | Iterable[str],
        coeffs: float | Iterable[float] = 1.0,
        offset: float = 0.0,
    ) -> None:
        self.params.tie(name, masters, coeffs, offset)

    # ------------------------------------------------------------------
    # 計算
    # ------------------------------------------------------------------
    def _contributions_for(self, i: int, h: HistogramEntry) -> list[PhaseContribution]:
        out = []
        for pname, pc in h.contributions.items():
            ph = self.phases[pname]
            ph.sync()
            pc.structure = ph.structure
            if ph.le_bail:
                if i not in ph.f2_tables:
                    self.start_le_bail(pname, i)
                pc.f2_table = ph.f2_tables[i]
            else:
                pc.f2_table = None
            out.append(pc)
        return out

    def calculate(self, hist: int = 0) -> PatternResult:
        h = self.histograms[hist]
        res = calculate_pattern(
            h.data.x,
            self._contributions_for(hist, h),
            h.instrument,
            h.sample,
            h.background,
            d_min=h.effective_d_min(),
            kalpha2_mode=h.kalpha2_mode,
            fcj_method=h.fcj_method,
            anomalous=h.anomalous,
            scattering_override=h.scattering_override,
        )
        h.result = res
        return res

    def calculate_all(self) -> list[PatternResult]:
        return [self.calculate(i) for i in range(len(self.histograms))]

    def residuals(self) -> FloatArray:
        parts = []
        for i, h in enumerate(self.histograms):
            res = self.calculate(i)
            m = h.mask()
            assert h.data.weight is not None
            w = h.data.weight[m] * h.weight_factor
            parts.append(np.sqrt(w) * (h.data.y[m] - res.y_calc[m]))
        if not parts:
            raise ValueError("ヒストグラムがありません")
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # 精密化
    # ------------------------------------------------------------------
    def refine(
        self,
        max_cycles: int = 20,
        *,
        verbose: int = 0,
        progress: Callable[[int, float], None] | None = None,
        le_bail_cycles: int = 0,
        ftol: float = 1e-6,
        xtol: float = 1e-8,
    ) -> FitResult:
        """vary の立ったパラメータを最小二乗で精密化する。

        le_bail_cycles > 0 なら、Le Bail 相の強度を更新しながらその回数だけ繰り返す。
        """
        if not self.histograms:
            raise ValueError("ヒストグラムがありません")
        self.params.evaluate_ties()
        self.params.apply_all()
        n_outer = max(le_bail_cycles, 1) if any(ph.le_bail for ph in self.phases.values()) else 1
        result: FitResult | None = None
        for _ in range(n_outer):
            if self.params.free():
                result = refine_least_squares(
                    self.params,
                    self.residuals,
                    max_cycles=max_cycles,
                    verbose=verbose,
                    progress=progress,
                    ftol=ftol,
                    xtol=xtol,
                )
            if any(ph.le_bail for ph in self.phases.values()):
                self.le_bail_update()
        if result is None:
            raise ValueError("精密化するパラメータがありません(vary を立ててください)")
        self.last_fit = result
        rf = self.rfactors()
        self.log.append(f"精密化: {result.names} → {rf['total']}")
        return result

    # ------------------------------------------------------------------
    # Le Bail
    # ------------------------------------------------------------------
    def start_le_bail(
        self, phase: str, hist: int | None = None, from_structure: bool = True
    ) -> None:
        """相を Le Bail モードにし、抽出強度の初期値を作る。"""
        ph = self.phases[phase]
        ph.le_bail = True
        hists = range(len(self.histograms)) if hist is None else [hist]
        for i in hists:
            h = self.histograms[i]
            pc = h.contributions[phase]
            ph.sync()
            from kaisetsu.reflections.hkl import generate_reflections

            refl = generate_reflections(
                ph.structure.lattice, ph.structure.spacegroup, h.effective_d_min()
            )
            if from_structure and ph.structure.sites:
                f2 = (
                    np.abs(
                        structure_factors(
                            ph.structure,
                            refl.hkl,
                            h.instrument.radiation,
                            anomalous=h.anomalous,
                            override=h.scattering_override,
                        )
                    )
                    ** 2
                )
                f2 = np.maximum(f2, 1e-3 * max(float(f2.max()), 1e-30))
            else:
                f2 = np.ones(len(refl))
            ph.f2_tables[i] = {
                (int(hk[0]), int(hk[1]), int(hk[2])): float(val)
                for hk, val in zip(refl.hkl, f2, strict=True)
            }
            pc.f2_table = ph.f2_tables[i]
            pc.f2_default = float(np.mean(f2))

    def le_bail_update(self) -> None:
        """Rietveld 分配で "観測" 強度を求め、Le Bail 相の |F|^2 を更新する。"""
        for i, h in enumerate(self.histograms):
            res = self.calculate(i)
            for k, (pname, _pc) in enumerate(h.contributions.items()):
                ph = self.phases[pname]
                if not ph.le_bail:
                    continue
                tbl = res.peaks[k]
                i_obs = partition_intensities(
                    h.data.y,
                    res.y_calc,
                    res.y_background,
                    tbl.windows(),
                    tbl.contributions,
                    tbl.intensity_total,
                )
                table = ph.f2_tables.setdefault(i, {})
                for hk, f2c, ic, io in zip(
                    tbl.hkl, tbl.f2, tbl.intensity_total, i_obs, strict=True
                ):
                    key = (int(hk[0]), int(hk[1]), int(hk[2]))
                    if ic > 0 and np.isfinite(io):
                        table[key] = max(float(f2c * io / ic), 0.0)

    # ------------------------------------------------------------------
    # 結果
    # ------------------------------------------------------------------
    def rfactors(self) -> dict[str, RFactors]:
        n_par = len(self.params.free())
        out: dict[str, RFactors] = {}
        all_o: list[FloatArray] = []
        all_c: list[FloatArray] = []
        all_w: list[FloatArray] = []
        all_b: list[FloatArray] = []
        for i, h in enumerate(self.histograms):
            res = h.result if h.result is not None else self.calculate(i)
            m = h.mask()
            assert h.data.weight is not None
            out[h.name] = powder_rfactors(
                h.data.y[m], res.y_calc[m], h.data.weight[m], n_par, res.y_background[m]
            )
            all_o.append(h.data.y[m])
            all_c.append(res.y_calc[m])
            all_w.append(h.data.weight[m] * h.weight_factor)
            all_b.append(res.y_background[m])
        out["total"] = powder_rfactors(
            np.concatenate(all_o),
            np.concatenate(all_c),
            np.concatenate(all_w),
            n_par,
            np.concatenate(all_b),
        )
        return out

    def reflection_report(self, hist: int = 0, phase: str | None = None) -> ReflectionReport:
        h = self.histograms[hist]
        res = self.calculate(hist)
        names = list(h.contributions)
        pname = phase or names[0]
        k = names.index(pname)
        tbl = res.peaks[k]
        i_obs = partition_intensities(
            h.data.y,
            res.y_calc,
            res.y_background,
            tbl.windows(),
            tbl.contributions,
            tbl.intensity_total,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            f2_obs = np.where(tbl.intensity_total > 0, tbl.f2 * i_obs / tbl.intensity_total, 0.0)
        return ReflectionReport(
            hkl=tbl.hkl,
            d=tbl.d,
            two_theta=tbl.two_theta,
            multiplicity=tbl.multiplicity,
            f2_calc=tbl.f2,
            f2_obs=f2_obs,
            i_calc=tbl.intensity_total,
            i_obs=i_obs,
            fwhm=tbl.fwhm,
            r_bragg=bragg_rfactor(i_obs, tbl.intensity_total),
            r_f=structure_rfactor(f2_obs, tbl.f2),
        )

    def canonical(self, phase: str, hkl: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return canonical_hkl(self.phases[phase].structure.spacegroup, hkl)

    def summary(self) -> str:
        lines = [f"プロジェクト {self.name or '(無名)'}"]
        for ph in self.phases.values():
            lines.append(f"相 {ph.name}: {ph.structure.spacegroup.symbol}, {ph.structure.lattice}")
        rf = self.rfactors() if self.histograms else {}
        for i, h in enumerate(self.histograms):
            lines.append(f"ヒストグラム h{i} ({h.name}): {len(h.data)} 点, {rf[h.name]}")
            for pname in h.contributions:
                rep = self.reflection_report(i, pname)
                lines.append(
                    f"  {pname}: R_B = {100 * rep.r_bragg:.2f}%  R_F = {100 * rep.r_f:.2f}%"
                )
        if "total" in rf and len(self.histograms) > 1:
            lines.append(f"全体: {rf['total']}")
        lines.append("パラメータ(* 精密化中, = 拘束):")
        lines.append(self.params.summary())
        return "\n".join(lines)
