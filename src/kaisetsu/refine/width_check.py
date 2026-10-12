"""ピーク幅のパラメータが負の幅を与えていないかの点検と、最小二乗への罰則。

Caglioti 式 Γ_G² = U tan²θ + V tanθ + W (+ P/cos²θ) は、U, V, W の組み合わせによっては
データ範囲の端で負になる。`CWInstrument.fwhm_gauss` はそこを下限(FWHM 0.0007°)で黙って
止めるので精密化は「通って」しまい、しかも止めた範囲では U, V, W の微分が 0 になって
ますます負の側に行きやすい(付属の FAP を標準手順で回すと 2θ ≥ 125° が負になっていた)。

- `width_violations`: 負(またはほぼ 0)になっている範囲を列挙する(精密化後の診断用)
- `width_penalty`: 負になった分を残差に加える罰則。可行領域(Γ² ≥ 0)の中では 0 なので
  正常な精密化には影響せず、解は境界(Γ² = 0)より外には出ない

ローレンツ幅 Γ_L = X tanθ + Y/cosθ + Z と、TOF のガウス分散 σ² = sig0 + sig1 d² + sig2 d⁴ + sigq d
も同じ扱い。

出典: G. Caglioti, A. Paoletti & F. P. Ricci, Nucl. Instrum. 3, 223-228 (1958);
    罰則法は J. Nocedal & S. J. Wright, Numerical Optimization, 2nd ed. (2006) Ch. 17.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from kaisetsu.pattern.instrument import CWInstrument
from kaisetsu.pattern.tof import TOFInstrument

if TYPE_CHECKING:
    from kaisetsu.refine.project import HistogramEntry, Project

FloatArray = npt.NDArray[np.float64]

#: 罰則の基準。この大きさの違反で残差 1 点あたり sqrt(N) 相当の重さになる
G2_REF = 1.0e-4  # deg²: (0.01°)²
L_REF = 1.0e-2  # deg
S2_REF = 1.0  # μs²
#: 「ほぼ 0(境界に張り付いている)」とみなす閾値(診断用)
G2_TOL = 0.002**2  # deg²
L_TOL = 0.002  # deg
S2_TOL = 1.0e-3  # μs²
#: 罰則をかける点の数(ヒストグラムのデータ範囲を等分)
N_PENALTY_POINTS = 32


@dataclass(frozen=True)
class WidthViolation:
    """負(またはほぼ 0)になっているピーク幅の情報。"""

    hist: int
    kind: str  # "gauss" | "lorentz" | "tof_sigma2"
    lo: float  # 負になっている範囲(2θ [deg] または TOF [μs])
    hi: float
    min_value: float
    at: float

    def message(self) -> str:
        """初心者向けの注意文(日本語)。"""
        h = f"h{self.hist}"
        if self.kind == "gauss":
            return (
                f"{h}:inst:u, v, w の組み合わせでガウス幅 Γ_G² が "
                f"2θ {self.lo:.1f}〜{self.hi:.1f}° で負または 0 になっています"
                f"(最小 {self.min_value:.1e} deg² at {self.at:.1f}°)。"
                "この範囲はピーク幅を下限で止めて計算しているので、W か U を固定して精密化し直して"
                "ください(標準手順は自動で U を固定してやり直します)。"
            )
        if self.kind == "lorentz":
            return (
                f"{h}:inst:x, y, z の組み合わせでローレンツ幅 Γ_L が "
                f"2θ {self.lo:.1f}〜{self.hi:.1f}° で負または 0 になっています"
                f"(最小 {self.min_value:.1e}° at {self.at:.1f}°)。"
                "X か Y を固定するか、Z を 0 に戻してください。"
            )
        return (
            f"{h}:inst:sig0, sig1, sig2 の組み合わせでガウス分散 σ² が "
            f"TOF {self.lo:.0f}〜{self.hi:.0f} μs で負または 0 になっています"
            f"(最小 {self.min_value:.1e} μs² at {self.at:.0f} μs)。"
            "sig0 か sig2 を固定してください。"
        )


def _grid(h: HistogramEntry, n: int) -> FloatArray | None:
    m = h.mask()
    if not m.any():
        return None
    x = h.data.x[m]
    return np.linspace(float(x[0]), float(x[-1]), n)


def _raw_widths(h: HistogramEntry, xs: FloatArray) -> list[tuple[str, FloatArray, float, float]]:
    """(種類, 生の幅(負になりうる), 診断の閾値, 罰則の基準) の一覧。"""
    inst = h.instrument
    if isinstance(inst, TOFInstrument):
        d = inst.d_from_tof(xs)
        return [("tof_sigma2", inst.sigma2_raw(d), S2_TOL, S2_REF)]
    assert isinstance(inst, CWInstrument)
    return [
        ("gauss", inst.gauss_fwhm2(xs), G2_TOL, G2_REF),
        ("lorentz", inst.lorentz_fwhm_raw(xs), L_TOL, L_REF),
    ]


def width_violations(project: Project, n_points: int = 512) -> list[WidthViolation]:
    """各ヒストグラムのデータ範囲で、ピーク幅が負(またはほぼ 0)になっている範囲を返す。"""
    out: list[WidthViolation] = []
    for i, h in enumerate(project.histograms):
        xs = _grid(h, n_points)
        if xs is None:
            continue
        for kind, vals, tol, _ref in _raw_widths(h, xs):
            bad = vals < tol
            if not bad.any():
                continue
            k = int(np.argmin(vals))
            out.append(
                WidthViolation(
                    hist=i,
                    kind=kind,
                    lo=float(xs[bad][0]),
                    hi=float(xs[bad][-1]),
                    min_value=float(vals[k]),
                    at=float(xs[k]),
                )
            )
    return out


def n_penalty_rows(project: Project, n_points: int = N_PENALTY_POINTS) -> int:
    """`width_penalty` が返す配列の長さ(常に一定)。"""
    n = 0
    for h in project.histograms:
        n += n_points * (1 if h.is_tof else 2)
    return n


def width_penalty(project: Project, n_points: int = N_PENALTY_POINTS) -> FloatArray:
    """負になったピーク幅への罰則残差(可行なら全部 0)。長さは `n_penalty_rows` と同じ。

    重みは sqrt(データ点数)。基準 G2_REF(0.01°)² の違反で、データ 1 点の残差が全点で
    1 ずれたのと同じ二乗和になる。負の側で滑らかでない(max(0, ·))が、最小二乗は境界に
    止まるので実用上は問題ない。
    """
    parts: list[FloatArray] = []
    n_data = sum(int(np.count_nonzero(h.mask())) for h in project.histograms)
    weight = float(np.sqrt(max(n_data, 1)))
    for h in project.histograms:
        xs = _grid(h, n_points)
        n_kinds = 1 if h.is_tof else 2
        if xs is None:
            parts.append(np.zeros(n_points * n_kinds))
            continue
        for _kind, vals, _tol, ref in _raw_widths(h, xs):
            parts.append(weight * np.maximum(0.0, -vals) / ref)
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)
