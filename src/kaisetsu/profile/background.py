"""バックグラウンド関数。

- Chebyshev 多項式(既定)
- 固定点の線形補間(ユーザーが置いた点の高さを係数にする。急な立ち上がりや複雑な形に向く)
- 非晶質ハンプ(ガウス関数)の追加項: 位置・幅・高さの 3 パラメータ

どのモデルも `coefficients`(線形係数)、`order`、`basis(x)`(係数ごとの基底関数)、
`__call__(x)` を持ち、精密化では係数が `h{i}:bkg:{k}` になる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class BackgroundModel(Protocol):
    """線形係数で表せる背景モデル。"""

    coefficients: FloatArray
    x_min: float
    x_max: float

    @property
    def order(self) -> int: ...

    def __call__(self, x: npt.ArrayLike) -> FloatArray: ...

    def basis(self, x: npt.ArrayLike) -> FloatArray: ...


@dataclass
class ChebyshevBackground:
    """第一種 Chebyshev 多項式によるバックグラウンド。

    y_b(x) = Σ_n c_n T_n(t),  t = -1 + 2 (x - x_min) / (x_max - x_min)

    x_min, x_max はデータの両端(GSAS-II の 'chebyschev-1' と同じ)。
    """

    coefficients: FloatArray = field(default_factory=lambda: np.zeros(1))
    x_min: float = 0.0
    x_max: float = 1.0

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64).ravel()
        if self.x_max <= self.x_min:
            raise ValueError("x_max は x_min より大きくなければなりません")

    @property
    def order(self) -> int:
        return len(self.coefficients)

    @property
    def kind(self) -> str:
        return "chebyshev"

    def __call__(self, x: npt.ArrayLike) -> FloatArray:
        xa = np.asarray(x, dtype=np.float64)
        t = -1.0 + 2.0 * (xa - self.x_min) / (self.x_max - self.x_min)
        return np.asarray(np.polynomial.chebyshev.chebval(t, self.coefficients))

    def basis(self, x: npt.ArrayLike) -> FloatArray:
        """各係数に対する基底関数の値 (N, order)。線形最小二乗に使う。"""
        xa = np.asarray(x, dtype=np.float64)
        t = -1.0 + 2.0 * (xa - self.x_min) / (self.x_max - self.x_min)
        return np.asarray(np.polynomial.chebyshev.chebvander(t, self.order - 1))


@dataclass
class InterpolatedBackground:
    """固定点 (x_k, c_k) の線形補間によるバックグラウンド。係数 c_k が各点の高さ。

    端の外側は端の値で延長する。基底関数は帽子関数(三角形)。
    """

    points_x: FloatArray = field(default_factory=lambda: np.array([0.0, 1.0]))
    coefficients: FloatArray = field(default_factory=lambda: np.zeros(2))
    x_min: float = 0.0
    x_max: float = 1.0

    def __post_init__(self) -> None:
        self.points_x = np.asarray(self.points_x, dtype=np.float64).ravel()
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64).ravel()
        if len(self.points_x) < 2 or np.any(np.diff(self.points_x) <= 0):
            raise ValueError("固定点は昇順に 2 点以上必要です")
        if len(self.coefficients) != len(self.points_x):
            raise ValueError("固定点の数と係数の数が一致しません")
        if self.x_max <= self.x_min:
            self.x_min, self.x_max = float(self.points_x[0]), float(self.points_x[-1])

    @property
    def order(self) -> int:
        return len(self.coefficients)

    @property
    def kind(self) -> str:
        return "points"

    def __call__(self, x: npt.ArrayLike) -> FloatArray:
        xa = np.asarray(x, dtype=np.float64)
        return np.asarray(np.interp(xa, self.points_x, self.coefficients))

    def basis(self, x: npt.ArrayLike) -> FloatArray:
        xa = np.asarray(x, dtype=np.float64).ravel()
        n = self.order
        out = np.zeros((len(xa), n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1.0
            out[:, k] = np.interp(xa, self.points_x, e)
        return out

    @classmethod
    def from_pattern(
        cls, x: npt.ArrayLike, y: npt.ArrayLike, n_points: int = 10, window_deg: float = 2.0
    ) -> InterpolatedBackground:
        """観測から等間隔の固定点を置き、SNIP 包絡線の値を初期値にする。"""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        n_points = max(int(n_points), 2)
        step = float(np.median(np.diff(xa)))
        window = int(max(3, min(round(window_deg / step), (len(xa) - 1) // 2)))
        env = snip_background(_boxcar(ya, max(round(0.25 / step), 1)), window)
        px = np.linspace(float(xa[0]), float(xa[-1]), n_points)
        return cls(px, np.interp(px, xa, env), float(xa[0]), float(xa[-1]))


@dataclass
class GaussianHump:
    """非晶質ハンプ: 高さ h、中心 c [deg]、半値全幅 w [deg] のガウス関数。"""

    center: float
    fwhm: float
    height: float

    def __call__(self, x: npt.ArrayLike) -> FloatArray:
        xa = np.asarray(x, dtype=np.float64)
        sigma = max(self.fwhm, 1e-6) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return np.asarray(self.height * np.exp(-0.5 * ((xa - self.center) / sigma) ** 2))


def snip_background(y: npt.ArrayLike, window: int, decreasing: bool = True) -> FloatArray:
    """SNIP 法(Ryan et al. 1988)による下側包絡線。

    対数変換(LLS)した上で、半幅 m を window から 1 まで減らしながら
    y_i ← min(y_i, (y_{i-m} + y_{i+m}) / 2) を繰り返す。
    """
    ya = np.asarray(y, dtype=np.float64)
    v = np.log(np.log(np.sqrt(np.maximum(ya, 0.0) + 1.0) + 1.0) + 1.0)
    n = len(v)
    widths = range(window, 0, -1) if decreasing else range(1, window + 1)
    for m in widths:
        if 2 * m >= n:
            continue
        avg = 0.5 * (v[: n - 2 * m] + v[2 * m :])
        mid = v[m : n - m]
        v[m : n - m] = np.minimum(mid, avg)
    return np.asarray((np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0)


def _boxcar(a: FloatArray, n_points: int) -> FloatArray:
    n = max(int(n_points), 1)
    if n <= 1:
        return a
    kernel = np.ones(n) / n
    pad = n // 2
    padded = np.pad(a, (pad, n - 1 - pad), mode="edge")
    return np.asarray(np.convolve(padded, kernel, mode="valid"))


def estimate_chebyshev_background(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n_terms: int = 8,
    window_deg: float = 2.0,
    smooth_deg: float = 0.25,
    margin_sigma: float = 1.0,
    n_iter: int = 40,
    weight: npt.ArrayLike | None = None,
) -> ChebyshevBackground:
    """観測パターンから背景の初期値を推定し、Chebyshev 多項式で返す。

    1. 軽く平滑化した観測に SNIP 法を当て、ピークを削った点ごとの下側包絡線を作る
       (端の立ち上がりは保たれる)。
    2. ピーク除去法(Sonneveld & Visser, J. Appl. Cryst. 8, 1 (1975)): n_terms 項の
       Chebyshev を重み付き最小二乗で当て、「当てはめ曲線と SNIP 包絡線の大きい方 + margin σ」
       より上の点を切り下げて当て直す、を収束まで繰り返す。
       SNIP 包絡線を下限に使うことで、低角の立ち上がりのような端の背景が切り落とされない。

    精密化の初期値として使う。項数が少ないと急な立ち上がりには追従しきれないので、
    低角の背景が合わないときは項数を増やす。

    Args:
        x, y: 観測パターン
        n_terms: Chebyshev の項数
        window_deg: SNIP の最大半幅 [deg]。ピークの束の半幅より大きく、背景のうねりより小さく。
        smooth_deg: SNIP の前に掛ける移動平均の幅 [deg](ノイズ由来の下方バイアスを抑える)
        margin_sigma: 切り下げの余裕(σ = sqrt(当てはめ値) の倍数)
        n_iter: 反復回数の上限
        weight: 最小二乗の重み(None なら 1/max(y, 1))
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) < max(n_terms, 2) * 3:
        raise ValueError("背景を推定するにはデータ点が少なすぎます")
    step = float(np.median(np.diff(xa)))
    window = int(max(3, min(round(window_deg / step), (len(xa) - 1) // 2)))
    env = snip_background(_boxcar(ya, round(smooth_deg / step)), window)
    bkg = ChebyshevBackground(np.zeros(max(n_terms, 1)), float(xa[0]), float(xa[-1]))
    basis = bkg.basis(xa)
    w = (
        np.asarray(weight, dtype=np.float64)
        if weight is not None
        else 1.0 / np.maximum(np.abs(ya), 1.0)
    )
    sw = np.sqrt(np.maximum(w, 0.0))
    y_work = ya.copy()
    coef = np.zeros(bkg.order)
    for _ in range(n_iter):
        coef, *_ = np.linalg.lstsq(basis * sw[:, None], y_work * sw, rcond=None)
        fit = basis @ coef
        ceiling = np.maximum(fit, env) + margin_sigma * np.sqrt(np.maximum(fit, 1.0))
        new = np.minimum(y_work, ceiling)
        if np.allclose(new, y_work, rtol=0, atol=1e-9):
            break
        y_work = new
    bkg.coefficients = coef
    return bkg
