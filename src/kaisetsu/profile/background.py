"""バックグラウンド関数。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


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

    def __call__(self, x: npt.ArrayLike) -> FloatArray:
        xa = np.asarray(x, dtype=np.float64)
        t = -1.0 + 2.0 * (xa - self.x_min) / (self.x_max - self.x_min)
        return np.asarray(np.polynomial.chebyshev.chebval(t, self.coefficients))

    def basis(self, x: npt.ArrayLike) -> FloatArray:
        """各係数に対する基底関数の値 (N, order)。線形最小二乗に使う。"""
        xa = np.asarray(x, dtype=np.float64)
        t = -1.0 + 2.0 * (xa - self.x_min) / (self.x_max - self.x_min)
        return np.asarray(np.polynomial.chebyshev.chebvander(t, self.order - 1))


def snip_background(y: npt.ArrayLike, window: int, decreasing: bool = True) -> FloatArray:
    """SNIP 法(Statistics-sensitive Non-linear Iterative Peak-clipping)による点ごとの背景。

    LLS 変換 v = log(log(sqrt(y + 1) + 1) + 1) の上で、幅 m を 1 から window まで(または逆順に)
    変えながら v_i ← min(v_i, (v_{i-m} + v_{i+m}) / 2) を繰り返し、逆変換して返す。
    端から m 点以内はそのまま残すので、低角の立ち上がりのような端の構造を潰さない。

    出典: C. G. Ryan et al., Nucl. Instrum. Methods B 34, 396-402 (1988);
          M. Morháč et al., Nucl. Instrum. Methods A 401, 113-132 (1997)(減少窓の変種)

    Args:
        y: 観測強度(非負)
        window: 最大の半幅(点数)。ピーク(または重なった束)の半幅より大きく取る。
        decreasing: 真なら幅を window から 1 に減らしていく(端や谷の歪みが小さい)
    """
    ya = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    v = np.log(np.log(np.sqrt(ya + 1.0) + 1.0) + 1.0)
    n = len(v)
    widths = range(window, 0, -1) if decreasing else range(1, window + 1)
    for m in widths:
        if 2 * m >= n:
            continue
        mid = v[m:-m]
        avg = 0.5 * (v[: n - 2 * m] + v[2 * m :])
        v[m:-m] = np.minimum(mid, avg)
    return np.asarray((np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0)


def _boxcar(a: FloatArray, n_points: int) -> FloatArray:
    k = max(int(n_points) | 1, 1)
    if k <= 1:
        return a
    ker = np.ones(k) / k
    return np.asarray(np.convolve(np.pad(a, k // 2, mode="edge"), ker, mode="valid"))


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
