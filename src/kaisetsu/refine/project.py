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
    h{i}:inst:{zero|u|v|w|x|y|sh_l|pol|kb_ratio}  装置(詳細: p, z。kb_ratio は Kβ の強度比)
    h{i}:inst:{difc|difa|difb|zero|alpha|beta0|beta1|betaq|sig0|sig1|sig2|sigq|x|y|z}  TOF 装置
    h{i}:sample:{displacement|transparency|rough_a|rough_b|disp_x|disp_y|mu_r}  試料
    h{i}:bkg:{k}                           バックグラウンド係数(Chebyshev または固定点の高さ)
    h{i}:hump{k}:{center|fwhm|height}      非晶質ハンプ(ガウス関数)
    h{i}:{相}:{scale|md_r|size|strain|ext} 相-ヒストグラム(ext は Sabine 消衰係数)
    h{i}:{相}:size_ax                      一軸異方的サイズの軸方向成分(有効にしたとき)
    h{i}:{相}:S_{HKL}                      Stephens 異方的歪み(有効にしたとき。ラウエ群で拘束)
    h{i}:{相}:sh{l}_{j}                    球面調和関数の集合組織の係数(有効にしたとき)
    g{i}:pdf:{scale|qdamp|qbroad}          PDF データセット(G(r))
    g{i}:{相}:{scale|delta1|delta2|sratio|spdiameter}  相-PDF
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import CrystalSystem, Lattice
from kaisetsu.crystal.structure import Structure
from kaisetsu.pattern.calculator import PatternResult, PhaseContribution, calculate_pattern
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.pattern.tof import TOFInstrument, calculate_pattern_tof
from kaisetsu.pdf.calculator import PairList, PDFPhase, PDFResult, calculate_pdf, pair_distribution
from kaisetsu.pdf.data import PDFData
from kaisetsu.profile.background import (
    BackgroundModel,
    ChebyshevBackground,
    GaussianHump,
    InterpolatedBackground,
    estimate_chebyshev_background,
)
from kaisetsu.profile.corrections import brindley_factor
from kaisetsu.profile.stephens import MONOMIALS, monomial_name, stephens_constraint
from kaisetsu.profile.texture import basis_size, coefficient_names, texture_index
from kaisetsu.refine.constraints import (
    SymmetryConstraint,
    adp_constraint,
    lattice_constraint,
    lattice_parameter_names,
    position_constraint,
)
from kaisetsu.refine.lsq import FitResult, refine_least_squares
from kaisetsu.refine.parameters import Parameter, ParameterSet
from kaisetsu.refine.pawley import PawleySolution, solve_pawley
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


def _shared_position_groups(sites: Sequence[Any], tol: float = 1e-4) -> list[list[Any]]:
    """同じ分率座標(格子並進を除く)にあるサイトをまとめる。2 個以上の組だけ返す。"""
    groups: list[list[Any]] = []
    used: set[int] = set()
    for i, a in enumerate(sites):
        if i in used:
            continue
        group = [a]
        for j in range(i + 1, len(sites)):
            if j in used:
                continue
            d = np.asarray(sites[j].position, dtype=float) - np.asarray(a.position, dtype=float)
            d -= np.round(d)
            if np.all(np.abs(d) < tol):
                group.append(sites[j])
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append(group)
    return groups


def _mirror_parameter(ps: ParameterSet, master: str, dependent: str) -> None:
    """dependent を master と常に同じ値に保つ。master が拘束・固定されていればそれを写す。"""
    m = ps[master]
    d = ps[dependent]
    if m.locked:
        d.locked = True
        d.value = m.value
        d.apply()
        return
    if m.tie is not None:
        ps.tie(dependent, m.tie.masters, m.tie.coeffs, m.tie.offset)
    else:
        ps.tie(dependent, [master], [1.0], 0.0)
    d.apply()


def _item_setter(arr: Any, k: int) -> Callable[[float], None]:
    def _set(v: float) -> None:
        arr[k] = v

    return _set


def _cell_setter(ph: PhaseEntry, k: int) -> Callable[[float], None]:
    def _set(v: float) -> None:
        ph.set_cell(k, v)

    return _set


def _kbeta_setter(inst: CWInstrument) -> Callable[[float], None]:
    """Kβ の強度比を書き込む(Radiation は不変なので差し替える)。"""

    def _set(v: float) -> None:
        from dataclasses import replace

        rad = inst.radiation
        if rad.wavelength3 is not None:
            inst.radiation = replace(rad, ratio3=max(v, 0.0))

    return _set


def _size_setter(pc: PhaseContribution) -> Callable[[float], None]:
    def _set(v: float) -> None:
        pc.size_um = v if v > 0 else None

    return _set


def _size_axial_setter(pc: PhaseContribution) -> Callable[[float], None]:
    def _set(v: float) -> None:
        pc.size_axial_um = max(v, 1e-6)

    return _set


@dataclass
class PhaseEntry:
    """プロジェクト内の 1 相。"""

    name: str
    structure: Structure
    le_bail: bool = False
    #: Le Bail 用の抽出強度(ヒストグラム添字 → 代表 hkl → |F|^2)
    f2_tables: dict[int, dict[tuple[int, int, int], float]] = field(default_factory=dict)
    #: 偽なら f2_tables を Le Bail で更新せず固定の |F|² 表として使う(MPF 用)
    f2_update: bool = True
    #: Pawley 法(残差評価のたびに反射強度を線形最小二乗で解く)。le_bail と排他
    pawley: bool = False
    #: Pawley 法の直近の解(ヒストグラム添字 → 解)
    pawley_solutions: dict[int, PawleySolution] = field(default_factory=dict)
    #: Brindley 微小吸収補正の粒子半径 [μm](0 で補正なし)
    brindley_radius_um: float = 0.0

    @property
    def uses_f2_table(self) -> bool:
        return self.le_bail or self.pawley

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


def merge_ranges(ranges: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """区間の一覧を正規化する: 逆順は直し、幅 0 は捨て、重なる・接する区間はまとめて昇順に。"""
    items: list[tuple[float, float]] = []
    for a, b in ranges:
        lo, hi = (float(a), float(b)) if a <= b else (float(b), float(a))
        if hi > lo:
            items.append((lo, hi))
    items.sort()
    out: list[tuple[float, float]] = []
    for lo, hi in items:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


@dataclass
class HistogramEntry:
    """プロジェクト内の 1 ヒストグラム(観測データ + 装置 + 試料 + 背景 + 相の寄与)。"""

    name: str
    data: PowderData
    instrument: CWInstrument | TOFInstrument
    sample: SampleGeometry
    background: BackgroundModel
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
    #: 非晶質ハンプ(背景に加える)
    humps: list[GaussianHump] = field(default_factory=list)

    def background_total(self, x: npt.ArrayLike) -> FloatArray:
        """背景関数 + ハンプ。"""
        y = np.asarray(self.background(x), dtype=np.float64)
        for hmp in self.humps:
            y = y + hmp(x)
        return y

    def set_excluded(self, ranges: Iterable[tuple[float, float]]) -> None:
        """除外領域を設定する(逆順は直し、空・重なり・隣接は 1 つにまとめる)。"""
        self.excluded = merge_ranges(ranges)

    def mask(self) -> BoolArray:
        x = self.data.x
        assert self.data.weight is not None
        m = (x >= self.x_min) & (x <= self.x_max) & (self.data.weight > 0)
        for lo, hi in self.excluded:
            m &= ~((x >= lo) & (x <= hi))
        return m

    @property
    def is_tof(self) -> bool:
        return isinstance(self.instrument, TOFInstrument)

    def effective_d_min(self) -> float:
        if self.d_min is not None:
            return self.d_min
        if isinstance(self.instrument, TOFInstrument):
            return float(self.instrument.d_from_tof(np.array([self.x_min]))[0]) * 0.98
        lam = self.instrument.radiation.wavelength
        tt = min(self.x_max + self.d_min_margin_deg, 179.0)
        return lam / (2.0 * math.sin(math.radians(tt / 2.0)))

    def effective_d_max(self) -> float | None:
        if isinstance(self.instrument, TOFInstrument):
            return float(self.instrument.d_from_tof(np.array([self.x_max]))[0]) * 1.02
        return None


def _structure_key(st: Structure) -> bytes:
    """原子対リストの再利用判定用: 格子・座標・占有率・変位パラメータのバイト列。"""
    parts = [np.ascontiguousarray(st.lattice.parameters, dtype=np.float64).tobytes()]
    for site in st.sites:
        parts.append(np.ascontiguousarray(site.position, dtype=np.float64).tobytes())
        parts.append(np.array([site.occupancy, site.u_iso], dtype=np.float64).tobytes())
        if site.u_aniso is not None:
            parts.append(np.ascontiguousarray(site.u_aniso, dtype=np.float64).tobytes())
    return b"".join(parts)


@dataclass
class PDFEntry:
    """プロジェクト内の 1 つの PDF データセット(観測 G(r) + 変換条件 + 相の寄与)。

    Attributes:
        radiation: 'xray' または 'neutron'
        qmax, qmin: 変換に使った Q の範囲 [Å^-1](終端効果の計算に使う)
        qdamp, qbroad: 装置分解能のパラメータ
        scale: データセットのスケール因子(PDFfit2 の dscale)
        phases: 相の名前 → PDF 用パラメータ
        r_min, r_max: フィットに使う r の範囲 [Å]
    """

    name: str
    data: PDFData
    radiation: str
    qmax: float
    phases: dict[str, PDFPhase]
    r_min: float
    r_max: float
    qmin: float = 0.0
    qdamp: float = 0.0
    qbroad: float = 0.0
    scale: float = 1.0
    weight_factor: float = 1.0
    dr_fine: float = 0.01
    result: PDFResult | None = None
    _pair_cache: dict[str, tuple[bytes, float, PairList]] = field(default_factory=dict, repr=False)

    def mask(self) -> BoolArray:
        r = self.data.r
        return (r >= self.r_min) & (r <= self.r_max) & (r > 0.0) & (self.data.weight > 0.0)

    def pairs_for(self, name: str, structure: Structure, r_max: float) -> PairList:
        """構造が変わっていなければ前回の原子対リストを使い回す。"""
        key = _structure_key(structure)
        hit = self._pair_cache.get(name)
        if hit is not None and hit[0] == key and hit[1] >= r_max:
            return hit[2]
        pl = pair_distribution(structure, r_max, self.radiation)
        self._pair_cache[name] = (key, r_max, pl)
        return pl


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
        self.pdf_datasets: list[PDFEntry] = []
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
        for g in self.pdf_datasets:
            g.phases[name] = PDFPhase(structure)
        self.build_parameters()
        return entry

    def add_histogram(
        self,
        data: PowderData,
        instrument: CWInstrument | TOFInstrument,
        sample: SampleGeometry | None = None,
        background: BackgroundModel | None = None,
        *,
        n_background: int = 8,
        name: str | None = None,
        phases: Iterable[str] | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        excluded: Sequence[tuple[float, float]] = (),
        kalpha2_mode: str = "exact",
        weight_factor: float = 1.0,
    ) -> HistogramEntry:
        """ヒストグラムを追加する。

        background を省略すると、観測パターンからピークを避けて推定した Chebyshev 背景
        (n_background 項)を初期値にする。
        """
        if background is None:
            background = _initial_background(data, max(n_background, 1))
        names = list(phases) if phases is not None else list(self.phases)
        for n in names:
            if n not in self.phases:
                raise KeyError(f"相 {n!r} はプロジェクトにありません")
        contributions = {n: PhaseContribution(self.phases[n].structure) for n in names}
        h = HistogramEntry(
            name=name or data.name or f"h{len(self.histograms)}",
            data=data,
            # 呼び出し側のオブジェクトを使い回しても別のプロジェクトに影響しないようコピーする
            instrument=copy.deepcopy(instrument),
            sample=copy.deepcopy(sample) if sample is not None else SampleGeometry(),
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

    def add_pdf(
        self,
        data: PDFData,
        radiation: str = "xray",
        *,
        qmax: float | None = None,
        qmin: float = 0.0,
        qdamp: float = 0.0,
        qbroad: float = 0.0,
        name: str | None = None,
        phases: Iterable[str] | None = None,
        r_min: float | None = None,
        r_max: float | None = None,
        weight_factor: float = 1.0,
    ) -> PDFEntry:
        """PDF データセット G(r) を追加する。

        qmax を省略するとデータのメタデータ(.gr ヘッダ)から取り、無ければ 0(終端効果なし)。
        r_min の既定は 1.0 Å(それより内側は終端リップルだけで情報が無い)。
        """
        if radiation not in ("xray", "neutron"):
            raise ValueError("radiation は xray か neutron です")
        names = list(phases) if phases is not None else list(self.phases)
        for n in names:
            if n not in self.phases:
                raise KeyError(f"相 {n!r} はプロジェクトにありません")
        if qmax is None:
            q = data.metadata.get("qmax", 0.0)
            qmax = float(q) if isinstance(q, int | float) else 0.0
        g = PDFEntry(
            name=name or data.name or f"g{len(self.pdf_datasets)}",
            data=data,
            radiation=radiation,
            qmax=qmax,
            qmin=qmin,
            qdamp=qdamp,
            qbroad=qbroad,
            phases={n: PDFPhase(self.phases[n].structure) for n in names},
            r_min=max(float(data.r[0]), 1.0) if r_min is None else r_min,
            r_max=float(data.r[-1]) if r_max is None else r_max,
            weight_factor=weight_factor,
        )
        self.pdf_datasets.append(g)
        self.build_parameters()
        return g

    def pdf_phase(self, dataset: int, phase: str) -> PDFPhase:
        return self.pdf_datasets[dataset].phases[phase]

    def reset_background(
        self,
        hist: int,
        n_terms: int | None = None,
        estimate: bool = True,
        kind: str | None = None,
    ) -> None:
        """背景の項数(固定点の数)を変える、種類を変える、または観測から推定し直す。

        kind は 'chebyshev' か 'points'(None なら今のまま)。種類を変えるときは必ず推定し直す。
        """
        h = self.histograms[hist]
        current = getattr(h.background, "kind", "chebyshev")
        kind = kind or current
        n = max(
            n_terms if n_terms is not None else h.background.order, 1 if kind == "chebyshev" else 2
        )
        if estimate or kind != current:
            if kind == "points":
                h.background = InterpolatedBackground.from_pattern(h.data.x, h.data.y, n)
            else:
                h.background = _initial_background(h.data, n)
        elif kind == "points":
            old_b = h.background
            assert isinstance(old_b, InterpolatedBackground)
            px = np.linspace(old_b.x_min, old_b.x_max, n)
            h.background = InterpolatedBackground(px, old_b(px), old_b.x_min, old_b.x_max)
        else:
            old = h.background.coefficients
            coeffs = np.zeros(n)
            coeffs[: min(n, len(old))] = old[: min(n, len(old))]
            h.background = ChebyshevBackground(coeffs, h.background.x_min, h.background.x_max)
        self.build_parameters()

    def enable_stephens(self, hist: int, phase: str, enable: bool = True) -> None:
        """相-ヒストグラムの歪み広がりを Stephens の異方的モデルにする(等方 strain から初期化)。"""
        pc = self.histograms[hist].contributions[phase]
        if enable:
            if pc.stephens is None:
                from kaisetsu.profile.stephens import isotropic_coefficients

                st = self.phases[phase].structure
                pc.stephens = isotropic_coefficients(
                    pc.strain_e6, st.lattice.reciprocal_metric_tensor
                )
        else:
            pc.stephens = None
        self.build_parameters()

    def enable_uniaxial_size(
        self,
        hist: int,
        phase: str,
        axis: tuple[int, int, int] | None = None,
        enable: bool = True,
    ) -> None:
        """相-ヒストグラムの結晶子サイズ広がりを一軸異方的にする(板状・針状)。

        axis は板の法線または針の方向(逆格子の hkl)。軸方向のサイズ h{i}:相:size_ax は
        現在の等方サイズから初期化する。等方サイズが無ければ 1 μm にする。
        """
        pc = self.histograms[hist].contributions[phase]
        if enable:
            if axis is not None:
                pc.size_axis = (int(axis[0]), int(axis[1]), int(axis[2]))
            if pc.size_um is None or pc.size_um <= 0:
                pc.size_um = 1.0
            if pc.size_axial_um is None:
                pc.size_axial_um = pc.size_um
        else:
            pc.size_axial_um = None
        self.build_parameters()

    def enable_texture(self, hist: int, phase: str, order: int = 4, enable: bool = True) -> None:
        """相-ヒストグラムに球面調和関数の集合組織(次数 order、偶数)を入れる。

        係数 h{i}:相:sh{l}_{j} はすべて 0(無配向)から始める。次数を変えると係数は 0 に戻る。
        March-Dollase と併用もできるが、通常は md_r を 1 に固定しておく。
        """
        pc = self.histograms[hist].contributions[phase]
        if enable:
            order = int(order)
            if order < 2 or order % 2:
                raise ValueError("球面調和関数の次数は 2 以上の偶数にしてください")
            st = self.phases[phase].structure
            pc.structure = st
            pc.sh_order = order
            n = basis_size(pc.texture_basis())
            if pc.sh_coefficients is None or len(pc.sh_coefficients) != n:
                pc.sh_coefficients = np.zeros(n)
        else:
            pc.sh_order = 0
            pc.sh_coefficients = None
        self.build_parameters()

    def texture_index(self, hist: int, phase: str) -> float:
        """集合組織指数 J = 1 + Σ C²(球面調和関数を使っていなければ 1)。"""
        pc = self.histograms[hist].contributions[phase]
        if pc.sh_order <= 0 or pc.sh_coefficients is None:
            return 1.0
        return texture_index(pc.sh_coefficients)

    def add_hump(self, hist: int, center: float, fwhm: float, height: float) -> GaussianHump:
        """非晶質ハンプを背景に加える(パラメータ h{i}:hump{k}:center|fwhm|height)。"""
        hmp = GaussianHump(center, fwhm, height)
        self.histograms[hist].humps.append(hmp)
        self.build_parameters()
        return hmp

    def clear_humps(self, hist: int) -> None:
        self.histograms[hist].humps.clear()
        self.build_parameters()

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
        for i, g in enumerate(self.pdf_datasets):
            self._build_pdf_params(ps, i, g)
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
        # 混合サイト(同じ座標に複数の散乱種): 座標と変位パラメータは先頭の種と同じ値に保ち、
        # 占有率は和を保つ(最後の種を他に従属させる)
        for group in _shared_position_groups(st.sites):
            first = group[0]
            for site in group[1:]:
                mirrored: list[str] = [*_XYZ]
                if first.u_aniso is None and site.u_aniso is None:
                    mirrored.append("uiso")
                elif first.u_aniso is not None and site.u_aniso is not None:
                    mirrored.extend(_ADP_NAMES)
                for nm in mirrored:
                    _mirror_parameter(ps, f"{pre}:{first.label}:{nm}", f"{pre}:{site.label}:{nm}")
            total = sum(site.occupancy for site in group)
            ps.tie(
                f"{pre}:{group[-1].label}:occ",
                [f"{pre}:{site.label}:occ" for site in group[:-1]],
                [-1.0] * (len(group) - 1),
                total,
            )

    def _build_tof_inst_params(self, ps: ParameterSet, pre: str, inst: TOFInstrument) -> None:
        spec: list[tuple[str, str, float, float, float, str, str]] = [
            ("difc", "difc", 1e-2, 1.0, 1e6, "μs/Å", "DIFC(d の係数)"),
            ("difa", "difa", 1e-3, -1e4, 1e4, "μs/Å²", "DIFA(d² の係数)"),
            ("difb", "difb", 1e-3, -1e4, 1e4, "μs·Å", "DIFB(1/d の係数)"),
            ("zero", "zero", 1e-2, -1e4, 1e4, "μs", "ゼロ点"),
            ("alpha", "alpha", 1e-4, 1e-4, 10.0, "1/μs·Å", "立ち上がり α = alpha / d"),
            ("beta0", "beta0", 1e-4, 0.0, 10.0, "1/μs", "減衰 β の定数項"),
            ("beta1", "beta1", 1e-5, 0.0, 10.0, "Å⁴/μs", "減衰 β の 1/d⁴ の係数"),
            ("betaq", "betaq", 1e-5, 0.0, 10.0, "Å²/μs", "減衰 β の 1/d² の係数"),
            ("sig0", "sig0", 1e-1, -1e6, 1e6, "μs²", "ガウス分散 σ² の定数項"),
            ("sig1", "sig1", 1e-1, -1e6, 1e6, "μs²/Å²", "σ² の d² の係数"),
            ("sig2", "sig2", 1e-1, -1e6, 1e6, "μs²/Å⁴", "σ² の d⁴ の係数"),
            ("sigq", "sigq", 1e-1, -1e6, 1e6, "μs²/Å", "σ² の d の係数"),
            ("x", "x", 1e-2, 0.0, 1e4, "μs/Å", "ローレンツ幅 γ の d の係数(歪み)"),
            ("y", "y", 1e-2, 0.0, 1e4, "μs/Å²", "γ の d² の係数(サイズ)"),
            ("z", "z", 1e-2, 0.0, 1e4, "μs", "γ の定数項"),
        ]  # fmt: skip
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

    def _build_hist_params(self, ps: ParameterSet, i: int, h: HistogramEntry) -> None:
        pre = f"h{i}"
        inst = h.instrument
        if isinstance(inst, TOFInstrument):
            self._build_tof_inst_params(ps, pre, inst)
            self._build_hist_common_params(ps, i, h)
            return
        spec: list[tuple[str, str, float, float, float, str, str]] = [
            # (名前, 属性, step, min, max, unit, 説明)
            ("zero", "zero", 1e-4, -5.0, 5.0, "deg", "ゼロ点シフト"),
            ("u", "u", 1e-5, -10.0, 10.0, "deg²", "ガウス幅 U (tan²θ)"),
            ("v", "v", 1e-5, -10.0, 10.0, "deg²", "ガウス幅 V (tanθ)"),
            ("w", "w", 1e-5, -1.0, 10.0, "deg²", "ガウス幅 W"),
            ("p", "p", 1e-5, -1.0, 10.0, "deg²", "ガウス幅 P (1/cos²θ)(詳細。通常は 0)"),
            ("x", "x", 1e-4, -1.0, 5.0, "deg", "ローレンツ幅 X (tanθ, 歪み)"),
            ("y", "y", 1e-4, -1.0, 5.0, "deg", "ローレンツ幅 Y (1/cosθ, サイズ)"),
            ("z", "z", 1e-4, -1.0, 5.0, "deg", "ローレンツ幅 Z (定数、詳細。通常は不要)"),
            ("sh_l", "sh_l", 1e-4, 0.0, 0.4, "", "軸発散 SH/L (= S/L + H/L、FCJ 非対称)"),
            ("pol", "polarization", 1e-3, 0.0, 1.0, "", "偏光係数"),
        ]
        advanced = {"p": 0.0, "z": 0.0}  # 詳細扱い(この値なら表に出さない)
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
                    advanced=nm in advanced,
                    neutral_value=advanced.get(nm),
                )
            )
        if inst.radiation.wavelength3 is not None:
            ps.add(
                Parameter(
                    f"{pre}:inst:kb_ratio",
                    float(inst.radiation.ratio3),
                    step=1e-4,
                    min=0.0,
                    max=0.5,
                    setter=_kbeta_setter(inst),
                    description="Kβ 線の強度比 I(Kβ)/I(Kα1)(フィルタ漏れ)",
                )
            )
        self._build_hist_common_params(ps, i, h)

    def _build_hist_common_params(self, ps: ParameterSet, i: int, h: HistogramEntry) -> None:
        """試料・背景・相ごとのパラメータ(CW と TOF で共通)。"""
        pre = f"h{i}"
        smp = h.sample
        sspec: list[tuple[str, str, float, float, float, str, str]] = [
            ("displacement", "displacement_mm", 1e-4, -5.0, 5.0, "mm", "試料高さずれ (BB)"),
            ("transparency", "transparency_mm", 1e-3, 0.0, 10.0, "mm", "透過補正 1/μ_eff (BB)"),
            ("rough_a", "roughness_a", 1e-3, 0.0, 1.0, "", "表面粗さ a (1 = なし、小さいほど粗い)"),
            ("rough_b", "roughness_b", 1e-2, 0.01, 10.0, "", "表面粗さ b (詳細。通常 1 に固定)"),
            ("disp_x", "displace_x_mm", 1e-4, -5.0, 5.0, "mm", "試料位置ずれ x (DS)"),
            ("disp_y", "displace_y_mm", 1e-4, -5.0, 5.0, "mm", "試料位置ずれ y (DS)"),
            (
                "mu_r",
                "mu_r",
                1e-3,
                0.0,
                20.0,
                "",
                "円筒試料の μR(λ = 1 Å での値。λ に比例)" if h.is_tof else "円筒試料の μR (DS)",
            ),
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
                    advanced=nm == "rough_b",
                )
            )
        scale_y = float(np.mean(np.abs(h.data.y))) + 1e-6
        bk = h.background
        is_points = getattr(bk, "kind", "chebyshev") == "points"
        for k in range(bk.order):
            if is_points:
                px = bk.points_x[k]  # type: ignore[attr-defined]
                desc = f"背景 固定点 {k} の高さ(2θ = {px:.2f}°)"
            else:
                desc = f"背景 Chebyshev 係数 {k}"
            ps.add(
                Parameter(
                    f"{pre}:bkg:{k}",
                    float(bk.coefficients[k]),
                    step=1e-3 * scale_y,
                    setter=_item_setter(bk.coefficients, k),
                    description=desc,
                )
            )
        span = float(h.data.x[-1] - h.data.x[0])
        for k, hmp in enumerate(h.humps):
            hpre = f"{pre}:hump{k}"
            ps.add(
                Parameter(
                    f"{hpre}:center",
                    hmp.center,
                    min=float(h.data.x[0]) - span,
                    max=float(h.data.x[-1]) + span,
                    step=1e-2,
                    setter=_attr_setter(hmp, "center"),
                    unit="deg",
                    description=f"非晶質ハンプ {k} の中心",
                )
            )
            ps.add(
                Parameter(
                    f"{hpre}:fwhm",
                    hmp.fwhm,
                    min=0.1,
                    max=4.0 * span,
                    step=1e-2,
                    setter=_attr_setter(hmp, "fwhm"),
                    unit="deg",
                    description=f"非晶質ハンプ {k} の半値全幅",
                )
            )
            ps.add(
                Parameter(
                    f"{hpre}:height",
                    hmp.height,
                    min=0.0,
                    step=1e-3 * scale_y,
                    setter=_attr_setter(hmp, "height"),
                    description=f"非晶質ハンプ {k} の高さ",
                )
            )
        for pname, pc in h.contributions.items():
            ppre = f"{pre}:{pname}"
            phase_entry = self.phases.get(pname)
            extracted = phase_entry is not None and (phase_entry.le_bail or phase_entry.pawley)
            ps.add(
                Parameter(
                    f"{ppre}:scale",
                    pc.scale,
                    min=0.0,
                    step=max(1e-3 * abs(pc.scale), 1e-9),
                    setter=_attr_setter(pc, "scale"),
                    # Le Bail / Pawley では抽出強度がスケールを含むので、動かすと縮退して
                    # スケール → 0、強度 → ∞ に流れて発散する(スクロース 11-BM で確認)
                    locked=extracted,
                    description=f"{pname} のスケール因子"
                    + "(Le Bail / Pawley では抽出強度に含まれるので固定)" * extracted,
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
                    description=f"{pname} の結晶子サイズ(0 で広がりなし)"
                    + ("。一軸モデルでは軸に垂直な方向" if pc.size_axial_um is not None else ""),
                )
            )
            ps.add(
                Parameter(
                    f"{ppre}:ext",
                    pc.extinction,
                    min=0.0,
                    max=1e5,
                    step=1e-2,
                    setter=_attr_setter(pc, "extinction"),
                    description=f"{pname} の Sabine 消衰係数(GSAS-II の Extinction と同じ単位)",
                )
            )
            if pc.size_axial_um is not None:
                ps.add(
                    Parameter(
                        f"{ppre}:size_ax",
                        float(pc.size_axial_um),
                        min=1e-3,
                        max=100.0,
                        step=1e-3,
                        setter=_size_axial_setter(pc),
                        unit="μm",
                        description=f"{pname} の結晶子サイズ(軸 {pc.size_axis} 方向)",
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
                    description=f"{pname} の微小歪み(等方。Stephens を使うときは無視)",
                )
            )
            if pc.stephens is not None:
                sc = stephens_constraint(self.phases[pname].structure.spacegroup, pc.stephens)
                pc.stephens[:] = sc.symmetrized
                for k, mono in enumerate(MONOMIALS):
                    par = Parameter(
                        f"{ppre}:{monomial_name(mono)}",
                        float(pc.stephens[k]),
                        min=-1e12,
                        max=1e12,
                        step=max(1e-3 * abs(float(pc.stephens[k])), 1e-3),
                        setter=_item_setter(pc.stephens, k),
                        description=f"{pname} の Stephens 異方的歪み {monomial_name(mono)}",
                    )
                    if k in sc.fixed:
                        par.locked = True
                    ps.add(par)
                for j, rel in sc.dependent.items():
                    ps.tie(
                        f"{ppre}:{monomial_name(MONOMIALS[j])}",
                        [f"{ppre}:{monomial_name(MONOMIALS[k])}" for k in rel.pivots],
                        rel.coeffs,
                        rel.offset,
                    )
            if pc.sh_order > 0:
                pc.structure = self.phases[pname].structure
                basis = pc.texture_basis()
                n_sh = basis_size(basis)
                if pc.sh_coefficients is None or len(pc.sh_coefficients) != n_sh:
                    pc.sh_coefficients = np.zeros(n_sh)
                for k, nm in enumerate(coefficient_names(basis)):
                    ell = nm[2:].split("_")[0]
                    ps.add(
                        Parameter(
                            f"{ppre}:{nm}",
                            float(pc.sh_coefficients[k]),
                            min=-10.0,
                            max=10.0,
                            step=1e-3,
                            setter=_item_setter(pc.sh_coefficients, k),
                            description=f"{pname} の集合組織(球面調和関数 l = {ell})",
                        )
                    )

    def _build_pdf_params(self, ps: ParameterSet, i: int, g: PDFEntry) -> None:
        pre = f"g{i}"
        ps.add(
            Parameter(
                f"{pre}:pdf:scale",
                g.scale,
                min=0.0,
                step=max(1e-3 * abs(g.scale), 1e-6),
                setter=_attr_setter(g, "scale"),
                description="PDF データセットのスケール因子",
            )
        )
        ps.add(
            Parameter(
                f"{pre}:pdf:qdamp",
                g.qdamp,
                min=0.0,
                max=1.0,
                step=1e-4,
                setter=_attr_setter(g, "qdamp"),
                unit="Å⁻¹",
                description="装置分解能による G(r) の減衰 Q_damp",
            )
        )
        ps.add(
            Parameter(
                f"{pre}:pdf:qbroad",
                g.qbroad,
                min=0.0,
                max=1.0,
                step=1e-4,
                setter=_attr_setter(g, "qbroad"),
                unit="Å⁻¹",
                description="装置分解能によるピーク幅の広がり Q_broad",
            )
        )
        for pname, pp in g.phases.items():
            ppre = f"{pre}:{pname}"
            spec: list[tuple[str, str, float, float, float, str, str]] = [
                ("scale", "scale", 1e-3, 0.0, 1e6, "", f"{pname} の PDF スケール因子(pscale)"),
                ("delta1", "delta1", 1e-3, 0.0, 10.0, "Å", f"{pname} の相関運動補正 δ1 (1/r)"),
                ("delta2", "delta2", 1e-3, 0.0, 50.0, "Å²", f"{pname} の相関運動補正 δ2 (1/r²)"),
                ("sratio", "sratio", 1e-3, 0.1, 2.0, "", f"{pname} の近距離ピーク幅比(r < rcut)"),
                ("spdiameter", "spdiameter", 1e-2, 0.0, 1e4, "Å", f"{pname} の粒子直径(0=バルク)"),
            ]  # fmt: skip
            for nm, attr, step, lo, hi, unit, desc in spec:
                ps.add(
                    Parameter(
                        f"{ppre}:{nm}",
                        float(getattr(pp, attr)),
                        step=step,
                        min=lo,
                        max=hi,
                        setter=_attr_setter(pp, attr),
                        unit=unit,
                        description=desc,
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
            if ph.uses_f2_table:
                if i not in ph.f2_tables:
                    self._init_f2_table(pname, i)
                pc.f2_table = ph.f2_tables[i]
            else:
                pc.f2_table = None
            out.append(pc)
        return out

    def calculate(self, hist: int = 0, *, pawley: bool = True) -> PatternResult:
        """パターンを計算する。Pawley 相があれば反射強度を線形最小二乗で解いて反映する。"""
        h = self.histograms[hist]
        res = self._calculate_raw(hist, h)
        if pawley and any(self.phases[n].pawley for n in h.contributions):
            self._pawley_solve(hist, h, res)
        h.result = res
        return res

    def _calculate_raw(self, hist: int, h: HistogramEntry) -> PatternResult:
        """Pawley の線形解を含まない素のパターン計算(CW / TOF で分岐)。"""
        if isinstance(h.instrument, TOFInstrument):
            return calculate_pattern_tof(
                h.data.x,
                self._contributions_for(hist, h),
                h.instrument,
                h.background_total,
                d_min=h.effective_d_min(),
                d_max=h.effective_d_max(),
                anomalous=h.anomalous,
                scattering_override=h.scattering_override,
                sample=h.sample,
            )
        return calculate_pattern(
            h.data.x,
            self._contributions_for(hist, h),
            h.instrument,
            h.sample,
            h.background_total,
            d_min=h.effective_d_min(),
            kalpha2_mode=h.kalpha2_mode,
            fcj_method=h.fcj_method,
            anomalous=h.anomalous,
            scattering_override=h.scattering_override,
        )

    def _pawley_solve(self, i: int, h: HistogramEntry, res: PatternResult) -> None:
        """Pawley 相の反射強度を解き、f2_tables と計算パターンを更新する。"""
        names = list(h.contributions)
        assert h.data.weight is not None
        mask = h.mask()
        for k, pname in enumerate(names):
            ph = self.phases[pname]
            if not ph.pawley:
                continue
            tbl = res.peaks[k]
            y_fixed = res.y_calc - res.y_phases[k]
            profiles: list[FloatArray] = []
            for c, it in zip(tbl.contributions, tbl.intensity_total, strict=True):
                profiles.append(c / it if it > 0 and len(c) else np.zeros(0))
            sol = solve_pawley(
                h.data.y, h.data.weight, y_fixed, tbl.windows(), profiles, mask=mask,
                compute_sigma=False, split_weights=tbl.intensity_total,
            )  # fmt: skip
            table = ph.f2_tables.setdefault(i, {})
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(tbl.intensity_total > 0, sol.intensity / tbl.intensity_total, 1.0)
            new_f2 = tbl.f2 * ratio
            for hk, f2 in zip(tbl.hkl, new_f2, strict=True):
                table[(int(hk[0]), int(hk[1]), int(hk[2]))] = max(float(f2), 0.0)
            # 計算パターンを解いた強度で置き換える(表はこの比で更新済み)
            res.y_calc = y_fixed + sol.y_bragg
            res.y_phases[k] = sol.y_bragg
            tbl.intensity_total = sol.intensity
            tbl.f2 = new_f2
            tbl.contributions = [
                p * a if len(p) else p for p, a in zip(profiles, sol.intensity, strict=True)
            ]
            ph.pawley_solutions[i] = sol

    def pawley_report(self, hist: int, phase: str) -> PawleySolution:
        """Pawley 法で解いた強度と σ(相関込み)。f2_tables を |F|² の単位に直す前の積分強度。"""
        h = self.histograms[hist]
        res = self._calculate_raw(hist, h)
        names = list(h.contributions)
        k = names.index(phase)
        tbl = res.peaks[k]
        assert h.data.weight is not None
        profiles = [
            c / it if it > 0 and len(c) else np.zeros(0)
            for c, it in zip(tbl.contributions, tbl.intensity_total, strict=True)
        ]
        return solve_pawley(
            h.data.y, h.data.weight, res.y_calc - res.y_phases[k], tbl.windows(), profiles,
            mask=h.mask(), compute_sigma=True, split_weights=tbl.intensity_total,
        )  # fmt: skip

    def pawley_structure_factors(
        self, hist: int, phase: str
    ) -> tuple[npt.NDArray[np.int64], FloatArray, FloatArray]:
        """Pawley 法の結果を (hkl, |F|², σ(|F|²)) で返す(MEM の入力用)。"""
        sol = self.pawley_report(hist, phase)
        h = self.histograms[hist]
        res = h.result if h.result is not None else self.calculate(hist)
        k = list(h.contributions).index(phase)
        tbl = res.peaks[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(tbl.intensity_total > 0, tbl.f2 / tbl.intensity_total, 0.0)
            sig = np.where(np.isfinite(sol.sigma), sol.sigma * scale, np.inf)
        return tbl.hkl.copy(), sol.intensity * scale, sig

    def calculate_all(self) -> list[PatternResult]:
        return [self.calculate(i) for i in range(len(self.histograms))]

    def calculate_pdf(self, dataset: int = 0) -> PDFResult:
        """PDF データセットの G(r) を現在のモデルで計算する。"""
        g = self.pdf_datasets[dataset]
        phases: list[PDFPhase] = []
        pairs: list[PairList] = []
        r_pair = float(g.data.r[-1]) + 6.0
        for pname, pp in g.phases.items():
            ph = self.phases[pname]
            ph.sync()
            pp.structure = ph.structure
            phases.append(pp)
            pairs.append(g.pairs_for(pname, ph.structure, r_pair))
        res = calculate_pdf(
            g.data.r,
            phases,
            g.radiation,
            qmax=g.qmax,
            qmin=g.qmin,
            qdamp=g.qdamp,
            qbroad=g.qbroad,
            scale=g.scale,
            dr_fine=g.dr_fine,
            pairs=pairs,
        )
        g.result = res
        return res

    #: 真なら精密化でピーク幅(Γ_G², Γ_L, TOF σ²)が負にならないよう罰則で拘束する
    #: (`kaisetsu.refine.width_check`)。既定は偽: GSAS-II と同じく負の範囲はその成分を 0 と
    #: して計算し、diagnostics で知らせる。拘束すると Rwp が悪くなることがある(PbSO4 で 9.9→14%)
    penalize_negative_widths: bool = False

    def jacobian(self, base: FloatArray | None = None, *, with_penalty: bool = False) -> FloatArray:
        """現在のパラメータでの残差ヤコビアン(半解析的)。base は直近の残差。

        with_penalty が真なら `residuals_with_penalty` に対するヤコビアン(罰則行を含む)。
        """
        from kaisetsu.refine.jacobian import project_jacobian

        fn = self.residuals_with_penalty if with_penalty else self.residuals
        return project_jacobian(self, base, residual_fn=fn)

    def n_fit_points(self) -> int:
        """フィットに使われるデータ点の総数(範囲内・除外領域外・重み > 0、PDF も含む)。"""
        n = sum(int(np.count_nonzero(h.mask())) for h in self.histograms)
        n += sum(int(np.count_nonzero(g.mask())) for g in self.pdf_datasets)
        return n

    def width_penalty(self) -> FloatArray:
        """ピーク幅(Γ_G², Γ_L, TOF σ²)が負になった分の罰則残差。可行なら全部 0。"""
        from kaisetsu.refine.width_check import width_penalty

        return width_penalty(self)

    def n_penalty_rows(self) -> int:
        """`width_penalty` の長さ(罰則が無効なら 0)。"""
        from kaisetsu.refine.width_check import n_penalty_rows

        return n_penalty_rows(self) if self.penalize_negative_widths else 0

    def residuals_with_penalty(self) -> FloatArray:
        """残差の末尾に幅の罰則行を付けたもの(精密化の目的関数)。"""
        r = self.residuals()
        if not self.penalize_negative_widths:
            return r
        pen = self.width_penalty()
        return np.concatenate([r, pen]) if pen.size else r

    def residuals(self) -> FloatArray:
        parts = []
        for i, h in enumerate(self.histograms):
            res = self.calculate(i)
            m = h.mask()
            assert h.data.weight is not None
            w = h.data.weight[m] * h.weight_factor
            parts.append(np.sqrt(w) * (h.data.y[m] - res.y_calc[m]))
        for i, g in enumerate(self.pdf_datasets):
            pres = self.calculate_pdf(i)
            m = g.mask()
            w = g.data.weight[m] * g.weight_factor
            parts.append(np.sqrt(w) * (g.data.g[m] - pres.g_calc[m]))
        if not parts:
            raise ValueError("ヒストグラムも PDF データもありません")
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
        analytic_jacobian: bool = True,
    ) -> FitResult:
        """vary の立ったパラメータを最小二乗で精密化する。

        le_bail_cycles > 0 なら、Le Bail 相の強度を更新しながらその回数だけ繰り返す。
        analytic_jacobian が真なら、背景・スケール・構造パラメータの微分を半解析的に求める
        (`kaisetsu.refine.jacobian`)。Pawley 相があるときは数値微分に戻す。
        """
        if not self.histograms and not self.pdf_datasets:
            raise ValueError("ヒストグラムも PDF データもありません")
        self.params.evaluate_ties()
        self.params.apply_all()
        n_free = len(self.params.free())
        n_data = self.n_fit_points()
        if n_free and n_data == 0:
            raise ValueError(
                "フィットに使えるデータ点がありません。データ範囲(x_min, x_max)と除外領域を"
                "確認してください"
            )
        if n_free > n_data:
            raise ValueError(
                f"動かすパラメータ({n_free} 個)がデータ点({n_data} 点)より多く、"
                "最小二乗が決まりません。データ範囲・除外領域を広げるか、動かす量を減らしてください"
            )
        updating = any(ph.le_bail and ph.f2_update for ph in self.phases.values())
        n_outer = max(le_bail_cycles, 1) if updating else 1
        result: FitResult | None = None
        for _ in range(n_outer):
            if self.params.free():
                use_analytic = analytic_jacobian and not any(
                    ph.pawley for ph in self.phases.values()
                )
                n_pen = self.n_penalty_rows()
                result = refine_least_squares(
                    self.params,
                    self.residuals_with_penalty,
                    max_cycles=max_cycles,
                    verbose=verbose,
                    progress=progress,
                    ftol=ftol,
                    xtol=xtol,
                    jacobian_fn=(
                        partial(self.jacobian, with_penalty=n_pen > 0) if use_analytic else None
                    ),
                    n_penalty=n_pen,
                )
            if updating:
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
    def start_pawley(
        self, phase: str, hist: int | None = None, from_structure: bool = True
    ) -> None:
        """相を Pawley モードにする(反射強度は残差評価のたびに線形最小二乗で決まる)。"""
        ph = self.phases[phase]
        ph.pawley = True
        ph.le_bail = False
        self._init_f2_table(phase, hist, from_structure)
        self._lock_extracted_scale(phase)

    def start_le_bail(
        self, phase: str, hist: int | None = None, from_structure: bool = True
    ) -> None:
        """相を Le Bail モードにし、抽出強度の初期値を作る。"""
        ph = self.phases[phase]
        ph.le_bail = True
        ph.pawley = False
        self._init_f2_table(phase, hist, from_structure)
        self._lock_extracted_scale(phase)

    def stop_extraction(self, phase: str) -> None:
        """Le Bail / Pawley をやめて構造からの強度計算に戻す(スケールの固定も解く)。"""
        ph = self.phases[phase]
        ph.le_bail = False
        ph.pawley = False
        ph.f2_update = True
        self._lock_extracted_scale(phase)

    def _lock_extracted_scale(self, phase: str) -> None:
        """相のスケール因子を、Le Bail / Pawley なら固定、そうでなければ解除する。"""
        ph = self.phases[phase]
        extracted = ph.le_bail or ph.pawley
        for p in self.params:
            parts = p.name.split(":")
            if len(parts) == 3 and parts[1] == phase and parts[2] == "scale":
                p.locked = extracted
                if extracted:
                    p.vary = False
                base = f"{phase} のスケール因子"
                p.description = base + (
                    "(Le Bail / Pawley では抽出強度に含まれるので固定)" if extracted else ""
                )

    def _init_f2_table(
        self, phase: str, hist: int | None = None, from_structure: bool = True
    ) -> None:
        """f2_tables の初期値(構造から、または一様)を作る。"""
        ph = self.phases[phase]
        hists = range(len(self.histograms)) if hist is None else [hist]
        for i in hists:
            h = self.histograms[i]
            pc = h.contributions[phase]
            ph.sync()
            from kaisetsu.reflections.hkl import generate_reflections

            refl = generate_reflections(
                ph.structure.lattice,
                ph.structure.spacegroup,
                h.effective_d_min(),
                h.effective_d_max() or math.inf,
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
                if not (ph.le_bail and ph.f2_update):
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
        for i, g in enumerate(self.pdf_datasets):
            pres = g.result if g.result is not None else self.calculate_pdf(i)
            m = g.mask()
            out[f"g{i}"] = powder_rfactors(g.data.g[m], pres.g_calc[m], g.data.weight[m], n_par)
            all_o.append(g.data.g[m])
            all_c.append(pres.g_calc[m])
            all_w.append(g.data.weight[m] * g.weight_factor)
            all_b.append(np.zeros(int(np.count_nonzero(m))))
        if not all_o:
            return out
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

    def diagnostics(self) -> list[str]:
        """精密化結果の簡単な点検。初心者向けの注意点を日本語で返す(問題が無ければ空)。"""
        notes: list[str] = []
        ps = self.params
        adp_tags = ("uiso", "u11", "u22", "u33")
        for p in ps:
            tail = p.name.rsplit(":", 1)[-1]
            if tail in adp_tags and p.value < -1e-6:
                notes.append(
                    f"{p.name} が負({p.value:.4f} Å²)です。平板試料(Bragg-Brentano)なら"
                    "表面粗さ(sample:rough_a)を、キャピラリなら吸収 μR を動かして"
                    "みてください。背景が高すぎる、偏光係数が合っていないときにも起こります。"
                )
            elif tail in adp_tags and p.value > 0.1:
                notes.append(
                    f"{p.name} が大きすぎます({p.value:.3f} Å²)。"
                    "そのサイトの元素や占有率を疑ってください。"
                )
        for p in ps.free():
            if math.isfinite(p.min) and abs(p.value - p.min) <= 1e-9 * max(1.0, abs(p.min)):
                notes.append(
                    f"{p.name} が下限 {p.min:g} に張り付いています。初期値を見直してください"
                )
            elif math.isfinite(p.max) and abs(p.value - p.max) <= 1e-9 * max(1.0, abs(p.max)):
                notes.append(
                    f"{p.name} が上限 {p.max:g} に張り付いています。初期値を見直してください"
                )
            elif p.esd is not None and p.esd > max(abs(p.value), 10 * p.step):
                notes.append(
                    f"{p.name} は決まっていません(esd {p.esd:.3g} が値 {p.value:.3g} より大きい)。"
                    "他のパラメータと強く相関しているか、データに感度がありません。固定を検討してください。"
                )
        from kaisetsu.refine.width_check import width_violations

        for v in width_violations(self):
            notes.append(v.message())
        if self.histograms and self.phases:
            rf = self.rfactors()["total"]
            if rf.chi2 > 10.0 and not self.pdf_datasets:
                notes.append(
                    f"χ² = {rf.chi2:.1f} と大きめです。計数が多いデータでは普通ですが、"
                    "差のプロットに系統的な残差があれば、ピーク形状(U, V, W, X, Y)・背景の項数・"
                    "未同定のピーク(別の相)を確認してください。"
                )
        return notes

    def phase_fractions(self, hist: int = 0) -> dict[str, tuple[float, float]]:
        """ヒストグラム hist での各相の重量分率と esd(相の名前 → (W, σ_W))。

        W_p = S_p M_p / Σ_j S_j M_j(Hill & Howard, J. Appl. Cryst. 20, 467 (1987) の
        W_p ∝ S_p (ZMV)_p を、スケール因子が 1/V を陽に含むこのプログラムの定義に直したもの。
        M は単位胞の質量)。σ_W はスケール因子の esd から相関を無視して伝播させた値。
        Le Bail / Pawley 相はスケールに意味が無いので除外する。

        相に Brindley の粒子半径が設定されていれば(X 線のみ)、微小吸収因子 τ_p で
        W_p ∝ S_p M_p / τ_p と補正する。τ_p は混合物の平均線吸収係数 μ̄ = Σ W_p μ_p に
        依存するので数回反復する。
        """
        h = self.histograms[hist]
        items: list[tuple[str, float, float, float]] = []
        for pname in h.contributions:
            ph = self.phases[pname]
            if ph.uses_f2_table:
                continue
            p = self.params[f"h{hist}:{pname}:scale"]
            items.append((pname, p.value, p.esd or 0.0, ph.structure.cell_mass()))
        rad = h.instrument.radiation
        use_brindley = rad.is_xray and any(
            self.phases[n].brindley_radius_um > 0 for n, _, _, _ in items
        )
        if use_brindley and len(items) > 1:
            mu = {}
            dens = {}
            for n, _, _, _ in items:
                st = self.phases[n].structure
                mu[n] = st.linear_absorption_coefficient(rad.wavelength)
                dens[n] = st.density()
            tau = dict.fromkeys(mu, 1.0)
            for _ in range(6):
                raw = {n: s * m / tau[n] for n, s, _, m in items}
                tot = sum(raw.values())
                w = {n: v / tot for n, v in raw.items()} if tot > 0 else dict.fromkeys(raw, 0.0)
                # 体積分率で平均した線吸収係数
                vol = {n: w[n] / dens[n] if dens[n] > 0 else 0.0 for n in w}
                vtot = sum(vol.values())
                mu_bar = sum(vol[n] / vtot * mu[n] for n in vol) if vtot > 0 else 0.0
                for n in tau:
                    r_cm = self.phases[n].brindley_radius_um * 1e-4
                    tau[n] = float(brindley_factor((mu[n] - mu_bar) * r_cm))
            items = [(n, s / tau[n], e / tau[n], m) for n, s, e, m in items]
        total = sum(s * m for _, s, _, m in items)
        if total <= 0:
            return {name: (0.0, 0.0) for name, _, _, _ in items}
        out: dict[str, tuple[float, float]] = {}
        for name, s, _, m in items:
            var = 0.0
            for name2, _, es2, m2 in items:
                if name2 == name:
                    d = m * (total - s * m) / total**2
                else:
                    d = -(m * s * m2) / total**2
                var += (d * es2) ** 2
            out[name] = (s * m / total, math.sqrt(var))
        return out

    def summary(self) -> str:
        lines = [f"プロジェクト {self.name or '(無名)'}"]
        for ph in self.phases.values():
            lines.append(f"相 {ph.name}: {ph.structure.spacegroup.symbol}, {ph.structure.lattice}")
        rf = self.rfactors() if (self.histograms or self.pdf_datasets) else {}
        for i, h in enumerate(self.histograms):
            lines.append(f"ヒストグラム h{i} ({h.name}): {len(h.data)} 点, {rf[h.name]}")
            for pname, pc in h.contributions.items():
                rep = self.reflection_report(i, pname)
                extra = ""
                if pc.sh_order > 0:
                    extra = f"  集合組織指数 J = {self.texture_index(i, pname):.3f}"
                lines.append(
                    f"  {pname}: R_B = {100 * rep.r_bragg:.2f}%  R_F = {100 * rep.r_f:.2f}%{extra}"
                )
            fractions = self.phase_fractions(i)
            if len(fractions) > 1:
                lines.append(
                    "  重量分率: "
                    + ", ".join(
                        f"{n} {100 * w:.2f} ± {100 * e:.2f} wt%" for n, (w, e) in fractions.items()
                    )
                )
        for i, g in enumerate(self.pdf_datasets):
            r = rf[f"g{i}"]
            lines.append(
                f"PDF g{i} ({g.name}): {int(np.count_nonzero(g.mask()))} 点 "
                f"(r = {g.r_min:.2f}〜{g.r_max:.2f} Å), Rw = {100 * r.rwp:.3f}%  χ² = {r.chi2:.3f}"
            )
        if "total" in rf and len(self.histograms) + len(self.pdf_datasets) > 1:
            lines.append(f"全体: {rf['total']}")
        lines.append("パラメータ(* 精密化中, = 拘束):")
        lines.append(self.params.summary())
        return "\n".join(lines)


def _initial_background(data: PowderData, n_terms: int) -> ChebyshevBackground:
    """観測からの背景推定。失敗したら中央値の定数にする。"""
    try:
        return estimate_chebyshev_background(data.x, data.y, n_terms)
    except (ValueError, np.linalg.LinAlgError):
        coeffs = np.zeros(n_terms)
        coeffs[0] = float(np.median(data.y))
        return ChebyshevBackground(coeffs, float(data.x[0]), float(data.x[-1]))
