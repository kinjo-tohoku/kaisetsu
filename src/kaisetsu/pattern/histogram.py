"""粉末回折データ(1 ヒストグラム)。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class PowderData:
    """観測パターン。

    Attributes:
        x: 2θ [deg](昇順)
        y: 強度(カウント)
        weight: 最小二乗の重み 1/σ^2。σ 不明なら Poisson 統計 (1/y) を仮定する。
        name: 表示名
        metadata: 読み込み元の情報
    """

    x: FloatArray
    y: FloatArray
    weight: FloatArray | None = None
    name: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64).ravel()
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        if self.x.shape != self.y.shape:
            raise ValueError("x と y の長さが一致しません")
        if len(self.x) < 2:
            raise ValueError("データ点が少なすぎます")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x(2θ)は単調増加でなければなりません")
        if self.weight is None:
            with np.errstate(divide="ignore"):
                w = np.where(self.y > 0, 1.0 / np.maximum(self.y, 1e-300), 0.0)
            self.weight = np.asarray(w, dtype=np.float64)
        else:
            self.weight = np.asarray(self.weight, dtype=np.float64).ravel()
            if self.weight.shape != self.x.shape:
                raise ValueError("weight の長さが x と一致しません")

    def __len__(self) -> int:
        return len(self.x)

    @property
    def sigma(self) -> FloatArray:
        """標準偏差 σ(重み 0 の点は inf)。"""
        assert self.weight is not None
        with np.errstate(divide="ignore"):
            return np.asarray(np.where(self.weight > 0, 1.0 / np.sqrt(self.weight), np.inf))

    @property
    def step(self) -> float:
        """平均ステップ幅 [deg]。"""
        return float((self.x[-1] - self.x[0]) / (len(self.x) - 1))

    def mask_range(self, x_min: float | None = None, x_max: float | None = None) -> BoolArray:
        m = np.ones(len(self.x), dtype=bool)
        if x_min is not None:
            m &= self.x >= x_min
        if x_max is not None:
            m &= self.x <= x_max
        return m

    def cropped(self, x_min: float | None = None, x_max: float | None = None) -> PowderData:
        m = self.mask_range(x_min, x_max)
        assert self.weight is not None
        return PowderData(self.x[m], self.y[m], self.weight[m], self.name, dict(self.metadata))
