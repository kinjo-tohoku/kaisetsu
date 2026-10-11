"""パターン図(pyqtgraph)。観測・計算・差・背景・Bragg 位置のティック。"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from kaisetsu.pattern.calculator import PatternResult
from kaisetsu.pattern.histogram import PowderData

pg.setConfigOptions(antialias=False, background="w", foreground="k")

_TICK_COLORS = ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


class PatternPlot(QWidget):
    """観測/計算/差/ティックを描くウィジェット。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.plot.setLabel("bottom", "2θ (deg)")
        self.plot.setLabel("left", "強度 (counts)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend(offset=(-10, 10))
        self._obs = self.plot.plot(
            [], [], pen=None, symbol="+", symbolSize=5, symbolPen=pg.mkPen("k"), name="観測"
        )
        self._calc = self.plot.plot([], [], pen=pg.mkPen("r", width=1.2), name="計算")
        self._bkg = self.plot.plot(
            [], [], pen=pg.mkPen("g", width=1, style=Qt.PenStyle.DashLine), name="背景"
        )
        self._diff = self.plot.plot([], [], pen=pg.mkPen("b", width=1), name="差")
        self._ticks: list[pg.PlotDataItem] = []
        self._wavelength: float | None = None
        self._label = pg.TextItem(anchor=(0, 1), color="k")
        self.plot.addItem(self._label, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse
        )
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._ycalc: np.ndarray | None = None

    # ------------------------------------------------------------------
    def set_data(self, data: PowderData | None, wavelength: float | None = None) -> None:
        self._wavelength = wavelength
        if data is None:
            self._x = np.zeros(0)
            self._y = np.zeros(0)
            self._obs.setData([], [])
            self.clear_calc()
            return
        self._x = data.x
        self._y = data.y
        self._obs.setData(data.x, data.y)
        self.clear_calc()
        self.plot.autoRange()

    def clear_calc(self) -> None:
        self._ycalc = None
        self._calc.setData([], [])
        self._bkg.setData([], [])
        self._diff.setData([], [])
        for t in self._ticks:
            self.plot.removeItem(t)
        self._ticks.clear()

    def set_result(
        self,
        result: PatternResult,
        tick_positions: dict[str, np.ndarray],
        mask: np.ndarray | None = None,
    ) -> None:
        """計算結果を描く。tick_positions は相名 → 2θ 配列。"""
        for t in self._ticks:
            self.plot.removeItem(t)
        self._ticks.clear()
        self._ycalc = result.y_calc
        self._calc.setData(result.x, result.y_calc)
        self._bkg.setData(result.x, result.y_background)
        ymax = float(np.nanmax(self._y)) if self._y.size else float(np.nanmax(result.y_calc))
        offset = -0.12 * ymax
        if self._y.size == result.x.size:
            diff = self._y - result.y_calc
            if mask is not None:
                diff = np.where(mask, diff, np.nan)
            self._diff.setData(result.x, diff + offset)
        tick_y = offset * 0.55
        for i, (name, pos) in enumerate(tick_positions.items()):
            pos = pos[np.isfinite(pos)]
            color = _TICK_COLORS[i % len(_TICK_COLORS)]
            item = self.plot.plot(
                pos, np.full(len(pos), tick_y - i * 0.03 * ymax), pen=None, symbol="|", symbolSize=9,
                symbolPen=pg.mkPen(color, width=1), symbolBrush=None, name=f"{name} の反射",
            )  # fmt: skip
            self._ticks.append(item)

    # ------------------------------------------------------------------
    def _on_mouse(self, evt: tuple) -> None:  # type: ignore[type-arg]
        pos = evt[0]
        vb = self.plot.getPlotItem().vb
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        x = float(p.x())
        text = f"2θ = {x:.3f}°"
        if self._wavelength and 0 < x < 180:
            d = self._wavelength / (2.0 * math.sin(math.radians(x / 2.0)))
            text += f"  d = {d:.4f} Å"
        if self._x.size:
            i = int(np.clip(np.searchsorted(self._x, x), 0, len(self._x) - 1))
            text += f"  観測 = {self._y[i]:.0f}"
            if self._ycalc is not None and i < len(self._ycalc):
                text += f"  計算 = {self._ycalc[i]:.0f}"
        self._label.setText(text)
        rng = vb.viewRange()
        self._label.setPos(rng[0][0], rng[1][1])
