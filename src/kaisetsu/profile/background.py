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
