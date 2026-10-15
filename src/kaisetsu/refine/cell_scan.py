"""格子定数の初期値を走査して、計算ピークが観測ピークに重なる谷を見つけてから精密化を始める。

最小二乗は、計算ピークと観測ピークが重なっていないと格子定数を正しい方向に動かせない
(残差の微分が 0 に近い)。合成 LaB6(FWHM 0.03°、放射光級)で初期値が +0.6% ずれると
標準手順の「格子定数とピーク位置」の段階は a を逆方向に動かし(Rwp 71% のまま)、あとの
「ピーク幅 W」の段階で幅が広がった拍子にたまたま戻っていた。±3% では a = 7.1 Å や 2.1 Å に
発散する。相見(電気化学 94, 264 (2026))の AutoRietveld は初期値を大量に振って 1 回ずつ
精密化し最良を採る。ここでは格子定数を等方的に ±span の範囲で振り、スケール因子だけを
解析的に最適化した Rwp が最小の点から始める(1 点 = パターン計算 1 回。合成 LaB6 1 万点で
81 点 0.4 秒)。刻みは高角端のピーク幅(Δd/d)の 0.4 倍で、どの刻みでも谷を踏むようにする。
格子定数が独立に 2 つ以上ある系(正方・六方・斜方…)では、等方のあとで各軸を ±span/2 で
個別に走査し、改善が止まるまで最大 CELL_SCAN_AXIS_ROUNDS 周繰り返す(1 周だけだと、他の軸が
ずれたままなので途中までしか寄らない。合成 PbSO4 で a +0.5%、c −0.5% から 1 周で残り 0.1%、
3 周で 0.02%)。角度(単斜の β など)は振らない。

採用は Rwp が CELL_SCAN_GAIN(0.2 ポイント)以上下がるときだけ。初期値が既に谷の中なら
何も変えない。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from kaisetsu.pattern.instrument import CWInstrument
from kaisetsu.pattern.tof import TOFInstrument

if TYPE_CHECKING:
    from kaisetsu.refine.parameters import Parameter
    from kaisetsu.refine.project import Project

#: 走査の範囲(±)。文献値や前の温度の値からのずれはこの範囲に収まる(熱膨張 1% は 1000 K 級)
CELL_SCAN_SPAN = 0.01
#: 採用の条件: Rwp がこれだけ(0.2 ポイント)下がる
CELL_SCAN_GAIN = 0.002
#: ±span で谷が見つからず、スケールだけ合わせた Rwp がこれより悪ければ範囲を
#: CELL_SCAN_WIDE_SPAN に広げる(文献と組成が違う、相転移で 1% を超えて跳んだ、など)
CELL_SCAN_WIDE_RWP = 0.25
CELL_SCAN_WIDE_SPAN = 0.03
#: 刻み = 高角端の Δd/d × この係数(谷の幅はピーク幅程度なので、半分以下の刻みなら必ず踏む)
CELL_SCAN_STEP_FACTOR = 0.4
#: 刻みの下限と点数の上限(放射光の極端に鋭いピークで走査が長くならないように)
CELL_SCAN_MIN_STEP = 2e-5
CELL_SCAN_MAX_POINTS = 801
#: TOF で幅を出せないときの Δd/d
CELL_SCAN_TOF_DEFAULT = 5e-4
#: 各軸の走査を繰り返す上限
CELL_SCAN_AXIS_ROUNDS = 3


@dataclass
class CellScanResult:
    """1 つの相の走査結果。"""

    phase: str
    names: list[str]
    rwp_before: float
    rwp_after: float
    step: float
    n_points: int
    #: 採用した倍率(名前 → 倍率)。採用しなければ空
    factors: dict[str, float] = field(default_factory=dict)
    message: str = ""

    @property
    def adopted(self) -> bool:
        return bool(self.factors)


def _tch_fwhm(gauss: np.ndarray, lorentz: np.ndarray) -> np.ndarray:
    """Thompson-Cox-Hastings の擬 Voigt 全幅(Γ_G と Γ_L から)。

    出典: P. Thompson, D. E. Cox, J. B. Hastings, J. Appl. Cryst. 20, 79 (1987), 式 (3)
    """
    g, lo = np.asarray(gauss), np.asarray(lorentz)
    return np.asarray(
        (
            g**5
            + 2.69269 * g**4 * lo
            + 2.42843 * g**3 * lo**2
            + 4.47163 * g**2 * lo**3
            + 0.07842 * g * lo**4
            + lo**5
        )
        ** 0.2
    )


def resolution_dd(project: Project, hist: int) -> float:
    """ヒストグラムのフィット範囲の高角端(TOF では d_min)でのピーク幅を Δd/d で返す。"""
    h = project.histograms[hist]
    inst = h.instrument
    if isinstance(inst, CWInstrument):
        tt = float(min(h.x_max, float(h.data.x[-1])))
        tt = min(max(tt, 1.0), 179.0)
        fwhm_deg = float(_tch_fwhm(inst.fwhm_gauss(tt), inst.fwhm_lorentz(tt)))
        theta = math.radians(tt / 2.0)
        return math.radians(fwhm_deg) / (2.0 * math.tan(theta))
    if isinstance(inst, TOFInstrument):
        try:
            d = float(h.effective_d_min())
            sigma2 = float(inst.sigma2_of(np.array([d]))[0])
            t = inst.difc * d
            if sigma2 > 0.0 and t > 0.0:
                return 2.3548 * math.sqrt(sigma2) / t
        except (ValueError, AttributeError, TypeError):
            pass
    return CELL_SCAN_TOF_DEFAULT


def scan_step(project: Project, span: float = CELL_SCAN_SPAN) -> tuple[float, int]:
    """走査の刻み(倍率の差)と片側の点数。全ヒストグラムの中で一番鋭い幅に合わせる。"""
    dd = min(resolution_dd(project, i) for i in range(len(project.histograms)))
    step = max(CELL_SCAN_STEP_FACTOR * dd, CELL_SCAN_MIN_STEP)
    half = math.ceil(span / step)
    if 2 * half + 1 > CELL_SCAN_MAX_POINTS:
        half = (CELL_SCAN_MAX_POINTS - 1) // 2
        step = span / half
    return step, half


def analytic_rwp(project: Project) -> float:
    """各ヒストグラムのスケール因子だけを解析的に最適化した Rwp(全ヒストグラム合計)。

    背景と相の相対スケールは今の値のまま。Bragg 強度全体に掛ける 1 つの係数 s を
    s = Σw(y_obs − bkg)·y_B / Σw·y_B² で決める(負なら 0)。
    """
    num = 0.0
    den = 0.0
    for i, h in enumerate(project.histograms):
        res = project.calculate(i)
        m = h.mask()
        assert h.data.weight is not None
        w = h.data.weight[m] * h.weight_factor
        yo = h.data.y[m]
        bk = res.y_background[m]
        yb = res.y_bragg[m]
        s = float(np.sum(w * (yo - bk) * yb)) / max(float(np.sum(w * yb * yb)), 1e-300)
        s = max(s, 0.0)
        num += float(np.sum(w * (yo - bk - s * yb) ** 2))
        den += float(np.sum(w * yo * yo))
    return math.sqrt(num / den) if den > 0.0 else float("nan")


def independent_cell_lengths(project: Project, phase: str) -> list[Parameter]:
    """相の格子定数 a, b, c のうち、拘束も固定もされていないもの。"""
    out: list[Parameter] = []
    for nm in ("a", "b", "c"):
        key = f"{phase}:cell:{nm}"
        if key in project.params:
            p = project.params[key]
            if p.tie is None and not p.locked:
                out.append(p)
    return out


def _set_scaled(
    project: Project, params: Sequence[Parameter], base: Sequence[float], factor: float
) -> None:
    for p, v0 in zip(params, base, strict=True):
        p.value = float(min(max(v0 * factor, p.min), p.max))
    project.params.evaluate_ties()
    project.params.apply_all()


def _scan_1d(
    project: Project,
    params: Sequence[Parameter],
    step: float,
    half: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """params をまとめて倍率 1 ± k·step(k = −half…half)で振った Rwp の列。"""
    base = [p.value for p in params]
    factors = 1.0 + step * np.arange(-half, half + 1)
    rwp = np.empty(len(factors))
    for k, f in enumerate(factors):
        _set_scaled(project, params, base, float(f))
        rwp[k] = analytic_rwp(project)
        if progress is not None:
            progress(k + 1, len(factors))
    _set_scaled(project, params, base, 1.0)
    return factors, rwp


def scan_cell(
    project: Project,
    phase: str,
    *,
    span: float = CELL_SCAN_SPAN,
    min_gain: float = CELL_SCAN_GAIN,
    progress: Callable[[int, int], None] | None = None,
) -> CellScanResult:
    """1 つの相の格子定数を振って、Rwp が最小の初期値に置き換える(改善が小さければ何もしない)。

    Returns:
        CellScanResult。`factors` に採用した倍率(名前 → 倍率)。
    """
    params = independent_cell_lengths(project, phase)
    if not project.histograms or not params:
        return CellScanResult(phase, [], float("nan"), float("nan"), 0.0, 0, message="対象なし")
    step, _ = scan_step(project, span)
    n_points = 0
    rwp0 = analytic_rwp(project)
    rwp_now = rwp0
    factors_adopted: dict[str, float] = {}
    notes: list[str] = []
    span_used = span
    # 等方。±span で谷が無く Rwp も悪ければ ±CELL_SCAN_WIDE_SPAN に広げてもう一度
    for sp in (span, max(CELL_SCAN_WIDE_SPAN, span)):
        step_sp, half_sp = scan_step(project, sp)
        f_iso, rwp_iso = _scan_1d(project, params, step_sp, half_sp, progress)
        n_points += len(f_iso)
        k = int(np.argmin(rwp_iso))
        if rwp_now - rwp_iso[k] >= min_gain and abs(f_iso[k] - 1.0) > 0.5 * step_sp:
            base = [p.value for p in params]
            _set_scaled(project, params, base, float(f_iso[k]))
            for p in params:
                factors_adopted[p.name] = factors_adopted.get(p.name, 1.0) * float(f_iso[k])
            notes.append(
                f"等方 {100 * (f_iso[k] - 1.0):+.3f}%"
                f"(Rwp {100 * rwp_now:.1f} → {100 * rwp_iso[k]:.1f}%)"
            )
            rwp_now = float(rwp_iso[k])
            span_used = sp
            break
        if rwp0 <= CELL_SCAN_WIDE_RWP or sp > span:
            break
        span_used = max(CELL_SCAN_WIDE_SPAN, span)
    # 各軸(独立な格子定数が 2 つ以上のとき)。改善が止まるまで繰り返す
    if len(params) > 1:
        half_ax = max(math.ceil(0.5 * span / step), 1)
        for _round in range(CELL_SCAN_AXIS_ROUNDS):
            improved = False
            for p in params:
                f_ax, rwp_ax = _scan_1d(project, [p], step, half_ax, progress)
                n_points += len(f_ax)
                k = int(np.argmin(rwp_ax))
                if rwp_now - rwp_ax[k] >= min_gain and abs(f_ax[k] - 1.0) > 0.5 * step:
                    _set_scaled(project, [p], [p.value], float(f_ax[k]))
                    factors_adopted[p.name] = factors_adopted.get(p.name, 1.0) * float(f_ax[k])
                    tag = p.name.rsplit(":", 1)[-1]
                    notes.append(
                        f"{tag} {100 * (f_ax[k] - 1.0):+.3f}%"
                        f"(Rwp {100 * rwp_now:.1f} → {100 * rwp_ax[k]:.1f}%)"
                    )
                    rwp_now = float(rwp_ax[k])
                    improved = True
            if not improved:
                break
    for h in project.histograms:
        h.result = None
    project.last_fit = None
    if factors_adopted:
        cells = "、".join(f"{p.name.rsplit(':', 1)[-1]} = {p.value:.5f}" for p in params)
        msg = (
            f"格子定数の初期値を ±{100 * span_used:.1f}% の範囲で {n_points} 点"
            f"(刻み {100 * step:.3f}%)振り、" + "、".join(notes) + f" を採りました。{cells}"
        )
    else:
        msg = (
            f"格子定数の初期値を ±{100 * span_used:.1f}% の範囲で {n_points} 点"
            f"(刻み {100 * step:.3f}%)振りましたが、初期値のままが最良です"
            f"(Rwp {100 * rwp0:.1f}%)"
        )
    return CellScanResult(
        phase, [p.name for p in params], rwp0, rwp_now, step, n_points, factors_adopted, msg
    )


def scan_all_cells(
    project: Project,
    *,
    span: float = CELL_SCAN_SPAN,
    min_gain: float = CELL_SCAN_GAIN,
    phases: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[CellScanResult]:
    """全相(または phases)の格子定数を順に走査する。各相の結果を返す。"""
    names = list(phases) if phases is not None else list(project.phases)
    return [
        scan_cell(project, name, span=span, min_gain=min_gain, progress=progress)
        for name in names
        if name in project.phases
    ]
