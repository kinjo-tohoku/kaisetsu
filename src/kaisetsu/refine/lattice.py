"""Le Bail 法で格子定数だけを速く出す(放射光マイクロ XRD、相同定後の格子定数チェック用)。

強度を構造から計算しない Le Bail 法は、μm ビームで照射体積の結晶粒が少なく強度が当てにならない
データでも、ピーク位置だけから格子定数を決められる。CIF からは空間群と格子定数の初期値だけを
使う(原子座標は無くてもよい)。

系統誤差の扱い(放射光の 2 次元検出器を積分したデータ):
    試料–検出器距離と波長の誤差はどちらも 2θ を比例的にずらし、格子定数とほぼ完全に相関する
    (距離 300 mm で試料位置が 30 μm ずれると Δa/a = 1e-4)。標準試料で校正した装置テンプレート
    (波長・幅・距離)を使って格子定数だけを動かすのが基本で、絶対値が要るときは標準試料を試料と
    同じ位置に置くか内部標準を混ぜる。相対値(マッピング、その場測定の変化)は 1e-5 まで信用できる。
    ビーム中心の誤差は環を一周積分すると平均されてほぼ消える。

位置の補正(position):
    - "none": 動かさない(校正済みのテンプレートを使うとき)
    - "zero": ゼロ点(2θ の定数ずれ)
    - "distance": 試料–検出器距離(ビーム方向の試料位置)のずれ。Debye-Scherrer では
      sample:disp_y(Δ2θ ∝ sin 2θ)、Bragg-Brentano では試料高さずれ、TOF ではゼロ点
ピーク幅(widths):
    - "sample": 試料由来の広がり(ローレンツ成分 X, Y。歪みと結晶子サイズに相当)だけ動かす。
      装置の U, V, W はテンプレートのまま。X, Y はテンプレートの値より小さくならない
    - "instrument": 装置の幅 U, V, W, X, Y も動かす(校正していない装置で最初に使う)
    - "none": 動かさない

出典: A. Le Bail, H. Duroy & J. L. Fourquet, Mater. Res. Bull. 23, 447-452 (1988);
    位置の補正は GSAS-II の Sample Parameters(DisplaceY)と同じ形
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from kaisetsu.crystal import Lattice, Structure
from kaisetsu.crystal.cif import read_structure
from kaisetsu.io.readers import read_powder_data
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.refine.project import Project
from kaisetsu.refine.recipe import Recipe

PositionMode = Literal["none", "zero", "distance"]
WidthMode = Literal["none", "sample", "instrument"]

CELL_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


def format_with_esd(value: float, esd: float | None, digits: int = 5) -> str:
    """5.41134(12) の形。esd が無ければ小数 digits 桁。"""
    if esd is None or not math.isfinite(esd) or esd <= 0.0:
        return f"{value:.{digits}f}"
    # esd は有効数字 2 桁。2 桁目が 0 なら 1 桁にする(0.0009 → (9)、0.00012 → (12))
    n = max(-math.floor(math.log10(esd)) + 1, 0)
    e = round(esd * 10**n)
    if e >= 100 or (e >= 20 and e % 10 == 0):
        n = max(n - 1, 0)
        e = round(esd * 10**n)
    return f"{value:.{n}f}({e})"


@dataclass
class CellResult:
    """1 相の格子定数。"""

    phase: str
    values: dict[str, float]  # a, b, c, alpha, beta, gamma
    esds: dict[str, float | None]
    volume: float
    volume_esd: float | None
    n_reflections: int

    def text(self) -> str:
        parts = []
        for nm in CELL_NAMES:
            unit = " Å" if nm in ("a", "b", "c") else "°"
            parts.append(f"{nm} = {format_with_esd(self.values[nm], self.esds.get(nm))}{unit}")
        return (
            f"{self.phase}: "
            + ", ".join(parts)
            + f", V = {format_with_esd(self.volume, self.volume_esd, 3)} Å³"
            + f"(反射 {self.n_reflections} 本)"
        )


@dataclass
class LatticeResult:
    """1 つのデータの結果。"""

    name: str
    path: str
    ok: bool
    message: str = ""
    rwp: float = float("nan")
    chi2: float = float("nan")
    n_points: int = 0
    cells: list[CellResult] = field(default_factory=list)
    #: 位置の補正(パラメータ名 → (値, esd))
    position: dict[str, tuple[float, float | None]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    project: Project | None = field(default=None, repr=False)

    def summary(self) -> str:
        if not self.ok:
            return f"{self.name}: 失敗({self.message})"
        lines = [
            f"{self.name}: Rwp = {100 * self.rwp:.3f}%  χ² = {self.chi2:.3f}  {self.n_points} 点"
        ]
        lines += ["  " + c.text() for c in self.cells]
        for pname, (v, e) in self.position.items():
            lines.append(f"  {pname} = {format_with_esd(v, e, 4)}")
        lines += ["  注意: " + w for w in self.warnings]
        return "\n".join(lines)


@dataclass
class LatticeBatch:
    """複数データの結果。CSV に書ける。"""

    results: list[LatticeResult] = field(default_factory=list)

    @property
    def phase_names(self) -> list[str]:
        for r in self.results:
            if r.ok:
                return [c.phase for c in r.cells]
        return []

    @property
    def position_names(self) -> list[str]:
        names: list[str] = []
        for r in self.results:
            for n in r.position:
                if n not in names:
                    names.append(n)
        return names

    def to_csv(self, path: str | Path) -> Path:
        import csv

        p = Path(path)
        phases = self.phase_names
        pos = self.position_names
        header = ["name", "ok", "Rwp%", "chi2", "n_points"]
        for ph in phases:
            for nm in CELL_NAMES:
                header += [f"{ph}:{nm}", f"esd({ph}:{nm})"]
            header += [f"{ph}:V", f"esd({ph}:V)", f"{ph}:n_refl"]
        for n in pos:
            header += [n, f"esd({n})"]
        header += ["warnings"]
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in self.results:
                row: list[object] = [
                    r.name,
                    int(r.ok),
                    f"{100 * r.rwp:.4f}",
                    f"{r.chi2:.4f}",
                    r.n_points,
                ]
                by_phase = {c.phase: c for c in r.cells}
                for ph in phases:
                    c = by_phase.get(ph)
                    for nm in CELL_NAMES:
                        v = None if c is None else c.values.get(nm)
                        e = None if c is None else c.esds.get(nm)
                        row += ["" if v is None else f"{v:.8g}", "" if e is None else f"{e:.3g}"]
                    row += [
                        "" if c is None else f"{c.volume:.8g}",
                        "" if c is None or c.volume_esd is None else f"{c.volume_esd:.3g}",
                        "" if c is None else c.n_reflections,
                    ]
                for n in pos:
                    v_e = r.position.get(n)
                    row += (
                        ["", ""]
                        if v_e is None
                        else [f"{v_e[0]:.8g}", "" if v_e[1] is None else f"{v_e[1]:.3g}"]
                    )
                row.append(" | ".join(r.warnings) if r.ok else r.message)
                w.writerow(row)
        return p

    def summary(self) -> str:
        return "\n".join(r.summary() for r in self.results)


def le_bail_recipe(
    hist: str = "h0",
    *,
    position: PositionMode = "zero",
    widths: WidthMode = "sample",
    geometry: GeometryKind | None = None,
    is_tof: bool = False,
    cell_scan: bool = True,
) -> Recipe:
    """Le Bail で格子定数を出すための段階的レシピ。

    背景 → 格子定数と位置の補正 → ピーク幅 → 最終。Le Bail 相の抽出強度は `Recipe.run` が
    各段階で 3 回ずつ更新する。
    """
    r = Recipe("Le Bail 格子定数")
    if cell_scan:
        # 初期値が谷の外だと最小二乗は格子定数を動かせない(kaisetsu.refine.cell_scan)
        r.add("格子定数の初期値を振る", [], 1, kind="cell_scan")
    r.add("背景", [f"{hist}:bkg:*"], 8)
    pos: list[str] = []
    if position == "zero" or (position == "distance" and is_tof):
        pos = [f"{hist}:inst:zero"]
    elif position == "distance":
        pos = [
            f"{hist}:sample:disp_y"
            if geometry is GeometryKind.DEBYE_SCHERRER
            else f"{hist}:sample:displacement"
        ]
    r.add("格子定数とピーク位置", ["*:cell:*", *pos], 10)
    if widths == "sample":
        r.add("ピーク幅(試料の広がり X, Y)", [f"{hist}:inst:x", f"{hist}:inst:y"], 10)
    elif widths == "instrument":
        if is_tof:
            r.add("ピーク幅 sig1", [f"{hist}:inst:sig1"], 10)
            r.add(
                "ピーク幅 sig0, sig2, X, Y",
                [f"{hist}:inst:sig0", f"{hist}:inst:sig2", f"{hist}:inst:x", f"{hist}:inst:y"],
                10,
            )
        else:
            r.add("ピーク幅 W", [f"{hist}:inst:w"], 10)
            r.add(
                "ピーク幅 U, V, X, Y",
                [f"{hist}:inst:u", f"{hist}:inst:v", f"{hist}:inst:x", f"{hist}:inst:y"],
                10,
            )
    r.add("最終(全パラメータ)", [], 20)
    return r


def _load_structures(
    structures: Sequence[Structure | str | Path], names: Sequence[str] | None
) -> list[tuple[str, Structure]]:
    out: list[tuple[str, Structure]] = []
    for i, s in enumerate(structures):
        st = s if isinstance(s, Structure) else read_structure(s)
        if names is not None and i < len(names) and names[i]:
            name = names[i]
        else:
            name = st.name or (Path(s).stem if not isinstance(s, Structure) else f"phase{i + 1}")
        name = name.replace(":", "_")
        base, k = name, 2
        while any(name == n for n, _ in out):
            name = f"{base}_{k}"
            k += 1
        out.append((name, st))
    return out


def _esd_of(project: Project, name: str) -> float | None:
    p = project.params[name]
    if p.tie is None:
        return p.esd
    if len(p.tie.masters) == 1:
        m = project.params[p.tie.masters[0]]
        return None if m.esd is None else abs(float(p.tie.coeffs[0])) * m.esd
    return None


def _cell_result(project: Project, phase: str, hist: int = 0) -> CellResult:
    ph = project.phases[phase]
    ph.sync()
    values = {nm: project.params[f"{phase}:cell:{nm}"].value for nm in CELL_NAMES}
    esds = {nm: _esd_of(project, f"{phase}:cell:{nm}") for nm in CELL_NAMES}
    lat = Lattice.from_parameters([values[nm] for nm in CELL_NAMES])
    volume = lat.volume
    # 体積の esd: 動かした格子定数の勾配 g = ∂V/∂p と直近の精密化の共分散 C から gᵀ C g。
    # 共分散が無い(直近の精密化に無い)成分は相関を無視して (∂V/∂p σ)² を足す
    # (斜方晶では a, b, c の相関が +0.4〜0.8 あり、二乗和だと esd が 0.6〜0.7 倍に出る。
    # レビュー 2026-10-17 #9)
    fit = project.last_fit
    grads: dict[str, float] = {}
    var_indep = 0.0
    any_esd = False
    for k, nm in enumerate(CELL_NAMES):
        p = project.params[f"{phase}:cell:{nm}"]
        if not p.is_free or p.esd is None:
            continue
        any_esd = True
        step = max(1e-6 * abs(p.value), 1e-7)

        def shifted(sign: float, k: int = k, name: str = p.name, h: float = step) -> np.ndarray:
            arr = np.array([values[n] for n in CELL_NAMES], dtype=float)
            arr[k] += sign * h
            # 拘束で連動する成分(例: 立方晶の b, c)も一緒に動かす
            for j, n2 in enumerate(CELL_NAMES):
                q = project.params[f"{phase}:cell:{n2}"]
                if q.tie is not None and q.tie.masters == (name,):
                    arr[j] += float(q.tie.coeffs[0]) * sign * h
            return arr

        v_plus = Lattice.from_parameters(shifted(1.0)).volume
        v_minus = Lattice.from_parameters(shifted(-1.0)).volume
        dv = (v_plus - v_minus) / (2.0 * step)
        if fit is not None and p.name in fit.names:
            grads[p.name] = dv
        else:
            var_indep += (dv * p.esd) ** 2
    var = var_indep
    if grads and fit is not None:
        idx = [fit.names.index(n) for n in grads]
        g = np.array(list(grads.values()), dtype=float)
        cov = np.asarray(fit.covariance, dtype=float)[np.ix_(idx, idx)]
        var += max(float(g @ cov @ g), 0.0)
    h = project.histograms[hist]
    k_phase = list(h.contributions).index(phase)
    n_refl = len(project.calculate(hist).peaks[k_phase].hkl)
    return CellResult(
        phase=phase,
        values=values,
        esds=esds,
        volume=volume,
        volume_esd=math.sqrt(var) if any_esd else None,
        n_reflections=n_refl,
    )


def _correlation_warnings(project: Project, position_names: Iterable[str]) -> list[str]:
    fit = project.last_fit
    if fit is None:
        return []
    out: list[str] = []
    names = fit.names
    for pos in position_names:
        if pos not in names:
            continue
        i = names.index(pos)
        for j, n in enumerate(names):
            if ":cell:" in n and abs(float(fit.correlation[i, j])) >= 0.95:
                out.append(
                    f"{n} と {pos} の相関が {fit.correlation[i, j]:+.3f} です。反射が少ないか "
                    "2θ 範囲が狭いので、位置の補正は標準試料で校正した値に固定するのが安全です"
                )
    return out


def _diagnostics_for_lattice(project: Project, widths: WidthMode) -> list[str]:
    """診断文。widths='sample' で X, Y が下限(テンプレートの値)なのは正常なので言い換える。"""
    notes: list[str] = []
    floor_hit = False
    for note in project.diagnostics():
        if widths == "sample" and any(note.startswith(f"h0:inst:{nm} が下限") for nm in ("x", "y")):
            floor_hit = True
            continue
        notes.append(note)
    if floor_hit:
        notes.append("試料由来の広がりは検出されませんでした(X, Y はテンプレートの値のまま)")
    return notes


def refine_lattice(
    data: PowderData | str | Path,
    structures: Sequence[Structure | str | Path],
    instrument: CWInstrument | TOFInstrument,
    sample: SampleGeometry | None = None,
    *,
    names: Sequence[str] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    n_background: int = 6,
    position: PositionMode = "zero",
    widths: WidthMode = "sample",
    progress: Callable[[str], None] | None = None,
    keep_project: bool = True,
) -> LatticeResult:
    """1 つのデータを Le Bail で精密化して格子定数を返す。

    Args:
        data: 回折データ(PowderData かファイル)
        structures: 相の CIF(複数可)。空間群と格子定数の初期値だけ使う
        instrument, sample: 装置と試料形状(校正したテンプレートを渡す)
        names: 相の名前(省略すると CIF の名前かファイル名)
        x_min, x_max: フィット範囲(省略で全体)
        n_background: Chebyshev 背景の項数
        position, widths: モジュール docstring を参照
        progress: 段階ごとの 1 行を受け取る関数
        keep_project: 真なら結果に Project を残す(図を見る、続きを手で精密化するため)
    """
    t0 = time.perf_counter()
    path = "" if isinstance(data, PowderData) else str(data)
    label = (data.name or "data") if isinstance(data, PowderData) else Path(data).stem
    try:
        pd = data if isinstance(data, PowderData) else read_powder_data(data)
        phases = _load_structures(structures, names)
        if not phases:
            raise ValueError("相(CIF)がありません")
        proj = Project(f"Le Bail {label}")
        smp = sample if sample is not None else SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER)
        h = proj.add_histogram(
            pd, instrument, smp, n_background=n_background, x_min=x_min, x_max=x_max
        )
        for name, st in phases:
            proj.add_phase(st.copy(), name=name, le_bail=True)
            proj.start_le_bail(name)
        proj.build_parameters()
        if widths == "sample" and not h.is_tof:
            # 試料の広がりは装置の幅に足す一方なので、テンプレートの X, Y を下限にする
            for nm in ("x", "y"):
                p = proj.params[f"h0:inst:{nm}"]
                p.min = max(p.value, 0.0)
        recipe = le_bail_recipe(
            "h0", position=position, widths=widths, geometry=smp.kind, is_tof=h.is_tof
        )

        def _step_done(step: object, _fit: object) -> None:
            if progress is not None:
                rf = proj.rfactors()["total"]
                progress(
                    f"[{getattr(step, 'name', '')}] Rwp = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}"
                )

        recipe.run(proj, progress=_step_done)
        rf = proj.rfactors()["total"]
        pos_step = next(s for s in recipe.steps if s.name == "格子定数とピーク位置")
        pos_names = [s for s in pos_step.patterns if ":cell:" not in s]
        result = LatticeResult(
            name=label,
            path=path,
            ok=True,
            rwp=rf.rwp,
            chi2=rf.chi2,
            n_points=proj.n_fit_points(),
            cells=[_cell_result(proj, name) for name, _ in phases],
            position={
                n: (proj.params[n].value, proj.params[n].esd) for n in pos_names if n in proj.params
            },
            warnings=_correlation_warnings(proj, pos_names)
            + _diagnostics_for_lattice(proj, widths),
            elapsed=time.perf_counter() - t0,
            project=proj if keep_project else None,
        )
    except Exception as exc:
        result = LatticeResult(
            name=label, path=path, ok=False, message=str(exc), elapsed=time.perf_counter() - t0
        )
    if progress is not None:
        progress(result.summary())
    return result


def refine_lattice_batch(
    paths: Iterable[str | Path],
    structures: Sequence[Structure | str | Path],
    instrument: CWInstrument | TOFInstrument,
    sample: SampleGeometry | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    keep_projects: bool = False,
    **kwargs: object,
) -> LatticeBatch:
    """複数のデータ(マッピング、その場測定)を同じ条件で順に処理する。

    kwargs は `refine_lattice` に渡す(x_min, x_max, n_background, position, widths, names)。
    """
    phases = _load_structures(structures, kwargs.pop("names", None))  # type: ignore[arg-type]
    batch = LatticeBatch()
    for p in paths:
        res = refine_lattice(
            p,
            [st for _, st in phases],
            instrument,
            sample,
            names=[n for n, _ in phases],
            progress=progress,
            keep_project=keep_projects,
            **kwargs,  # type: ignore[arg-type]
        )
        batch.results.append(res)
    return batch


__all__ = [
    "CELL_NAMES",
    "CellResult",
    "LatticeBatch",
    "LatticeResult",
    "PositionMode",
    "WidthMode",
    "format_with_esd",
    "le_bail_recipe",
    "refine_lattice",
    "refine_lattice_batch",
]
