"""パターン図(pyqtgraph)。観測・計算・差・背景・Bragg 位置のティック。PDF の G(r) 図。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from kaisetsu.pattern.calculator import PatternResult
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pdf.calculator import PDFResult
from kaisetsu.pdf.data import PDFData

pg.setConfigOptions(antialias=False, background="w", foreground="k")

_TICK_COLORS = ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def export_plot_image(plot: pg.PlotWidget, path: str | Path, width: int = 1600) -> Path:
    """図を PNG / SVG / PDF に書き出す(拡張子で判定)。

    PNG は pyqtgraph の ImageExporter(画面と同じ見た目)。SVG / PDF は曲線データを matplotlib で
    描き直す(pyqtgraph の SVGExporter は記号付きの曲線で失敗することがあるため)。
    """
    p = Path(path)
    item = plot.getPlotItem()
    if p.suffix.lower() in (".svg", ".pdf"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 日本語の軸ラベル用に、入っている日本語フォントを優先する
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"):
            if name in available:
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 5))
        for curve in item.listDataItems():
            x, y = curve.getData()
            if x is None or len(x) == 0:
                continue
            opts = curve.opts
            name = opts.get("name") or None
            pen = opts.get("pen")
            symbol = opts.get("symbol")
            color = "k"
            if pen is not None:
                qc = pg.mkPen(pen).color()
                color = (qc.redF(), qc.greenF(), qc.blueF())
            elif opts.get("symbolPen") is not None:
                qc = pg.mkPen(opts["symbolPen"]).color()
                color = (qc.redF(), qc.greenF(), qc.blueF())
            if pen is None and symbol is not None:
                marker = {"+": "+", "o": "o", "|": "|", "x": "x"}.get(str(symbol), ".")
                ax.plot(
                    x, y, linestyle="none", marker=marker, markersize=3, color=color, label=name
                )
            else:
                ax.plot(x, y, linewidth=0.8, color=color, label=name)
        ax.set_xlabel(item.getAxis("bottom").labelText or "")
        ax.set_ylabel(item.getAxis("left").labelText or "")
        (x0, x1), (y0, y1) = item.vb.viewRange()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        if any(c.opts.get("name") for c in item.listDataItems()):
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(str(p))
        plt.close(fig)
        return p
    from pyqtgraph.exporters import ImageExporter

    exporter = ImageExporter(item)
    exporter.parameters()["width"] = width
    exporter.export(str(p))
    return p


class _PatternViewBox(pg.ViewBox):
    """Shift + 左ドラッグで除外領域を指定できる ViewBox。

    通常のドラッグ(移動・拡大)はそのまま。Shift を押しながらドラッグした範囲を
    sigExcludeRequested(開始, 終了) で知らせる。
    """

    sigExcludeRequested = Signal(float, float)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._drag_region: Any = None

    def mouseDragEvent(self, ev: Any, axis: int | None = None) -> None:
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if ev.button() != Qt.MouseButton.LeftButton or not shift:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        x0 = float(self.mapToView(ev.buttonDownPos()).x())
        x1 = float(self.mapToView(ev.pos()).x())
        lo, hi = min(x0, x1), max(x0, x1)
        if self._drag_region is None:
            self._drag_region = pg.LinearRegionItem(
                values=(lo, hi), movable=False, brush=pg.mkBrush(255, 0, 0, 40), pen=pg.mkPen(None)
            )
            self.addItem(self._drag_region, ignoreBounds=True)
        self._drag_region.setRegion((lo, hi))
        if ev.isFinish():
            self.removeItem(self._drag_region)
            self._drag_region = None
            if hi > lo:
                self.sigExcludeRequested.emit(lo, hi)


class _ExcludeRegion(pg.LinearRegionItem):
    """除外領域の帯。ドラッグで動かせ、右クリックで削除を要求する。"""

    sigRemoveRequested = Signal(object)

    def __init__(self, lo: float, hi: float) -> None:
        super().__init__(
            values=(lo, hi),
            movable=True,
            brush=pg.mkBrush(255, 0, 0, 40),
            hoverBrush=pg.mkBrush(255, 0, 0, 70),
            pen=pg.mkPen((200, 0, 0, 120)),
        )
        self.setZValue(-10)
        self.setToolTip("除外領域: ドラッグで移動、端をドラッグで幅を変更、右クリックで削除")

    def mouseClickEvent(self, ev: Any) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.sigRemoveRequested.emit(self)
            return
        super().mouseClickEvent(ev)


class PatternPlot(QWidget):
    """観測/計算/差/ティックを描くウィジェット。除外領域とフィット範囲も帯で示す。"""

    #: Shift + ドラッグで新しい除外領域が指定された(開始, 終了)
    excludeRequested = Signal(float, float)
    #: 帯の移動・削除で除外領域の一覧が変わった(新しい一覧)
    excludedChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        vb = _PatternViewBox()
        self.plot = pg.PlotWidget(viewBox=vb)
        vb.sigExcludeRequested.connect(self.excludeRequested)
        self._regions: list[_ExcludeRegion] = []
        self._range_shades: list[Any] = []
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
        self._difc: float | None = None

    def set_tof(self, difc: float | None) -> None:
        """TOF データの表示にする(difc は d 値表示用)。None で 2θ に戻す。"""
        self._difc = difc
        if difc is not None:
            self.plot.setLabel("bottom", "TOF (μs)")
            self.plot.setLabel("left", "強度 (counts/μs)")
        else:
            self.plot.setLabel("bottom", "2θ (deg)")
            self.plot.setLabel("left", "強度 (counts)")

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
        if self._difc:
            text = f"TOF = {x:.1f} μs  d ≈ {x / self._difc:.4f} Å"
        else:
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

    # ------------------------------------------------------------------
    # 除外領域とフィット範囲
    # ------------------------------------------------------------------
    def set_excluded(self, ranges: list[tuple[float, float]]) -> None:
        """除外領域の帯を描き直す。"""
        for item in self._regions:
            self.plot.removeItem(item)
        self._regions = []
        for lo, hi in ranges:
            item = _ExcludeRegion(float(lo), float(hi))
            item.sigRegionChangeFinished.connect(self._on_region_changed)
            item.sigRemoveRequested.connect(self._on_region_removed)
            self.plot.addItem(item, ignoreBounds=True)
            self._regions.append(item)

    def excluded(self) -> list[tuple[float, float]]:
        """帯の現在位置から除外領域の一覧を返す。"""
        out: list[tuple[float, float]] = []
        for item in self._regions:
            lo, hi = item.getRegion()
            out.append((float(lo), float(hi)))
        return out

    def set_fit_range(self, x_min: float, x_max: float) -> None:
        """フィットに使わない両端(x_min より前、x_max より後)を灰色で示す。"""
        for item in self._range_shades:
            self.plot.removeItem(item)
        self._range_shades = []
        if not self._x.size:
            return
        x0, x1 = float(self._x[0]), float(self._x[-1])
        for lo, hi in ((x0, x_min), (x_max, x1)):
            if hi > lo:
                shade = pg.LinearRegionItem(
                    values=(lo, hi),
                    movable=False,
                    brush=pg.mkBrush(0, 0, 0, 25),
                    pen=pg.mkPen(None),
                )
                shade.setZValue(-20)
                self.plot.addItem(shade, ignoreBounds=True)
                self._range_shades.append(shade)

    def _on_region_changed(self, _item: Any = None) -> None:
        self.excludedChanged.emit(self.excluded())

    def _on_region_removed(self, item: Any) -> None:
        if item in self._regions:
            self.plot.removeItem(item)
            self._regions.remove(item)
        self.excludedChanged.emit(self.excluded())

    def update_curves(self, x: np.ndarray, y_calc: np.ndarray, y_bkg: np.ndarray) -> None:
        """精密化の途中経過: 計算・背景・差の曲線だけを差し替える(ティックはそのまま)。"""
        self._ycalc = y_calc
        self._calc.setData(x, y_calc)
        self._bkg.setData(x, y_bkg)
        if self._y.size == x.size:
            ymax = float(np.nanmax(self._y)) if self._y.size else 1.0
            self._diff.setData(x, self._y - y_calc - 0.12 * ymax)

    def export_image(self, path: str | Path) -> Path:
        return export_plot_image(self.plot, path)


class PDFPlot(QWidget):
    """G(r) の観測/計算/差を描くウィジェット。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.plot.setLabel("bottom", "r (Å)")
        self.plot.setLabel("left", "G(r) (Å⁻²)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend(offset=(-10, 10))
        self._obs = self.plot.plot(
            [], [], pen=None, symbol="o", symbolSize=4, symbolPen=pg.mkPen("k"),
            symbolBrush=None, name="観測 G(r)",
        )  # fmt: skip
        self._calc = self.plot.plot([], [], pen=pg.mkPen("r", width=1.3), name="計算")
        self._diff = self.plot.plot([], [], pen=pg.mkPen("b", width=1), name="差")
        self._range = pg.LinearRegionItem(brush=pg.mkBrush(0, 120, 255, 20), movable=False)
        self._range.hide()
        self.plot.addItem(self._range)
        self._label = pg.TextItem(anchor=(0, 1), color="k")
        self.plot.addItem(self._label, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse
        )
        self._r = np.zeros(0)
        self._g = np.zeros(0)
        self._gcalc: np.ndarray | None = None

    def set_data(self, data: PDFData | None, r_range: tuple[float, float] | None = None) -> None:
        if data is None:
            self._r = np.zeros(0)
            self._g = np.zeros(0)
            self._obs.setData([], [])
            self.clear_calc()
            self._range.hide()
            return
        self._r = data.r
        self._g = data.g
        self._obs.setData(data.r, data.g)
        self.clear_calc()
        if r_range is not None:
            self._range.setRegion(r_range)
            self._range.show()
        self.plot.autoRange()

    def clear_calc(self) -> None:
        self._gcalc = None
        self._calc.setData([], [])
        self._diff.setData([], [])

    def set_result(self, result: PDFResult, mask: np.ndarray | None = None) -> None:
        self._gcalc = result.g_calc
        self._calc.setData(result.r, result.g_calc)
        if self._g.size == result.r.size:
            span = float(np.nanmax(self._g) - np.nanmin(self._g)) if self._g.size else 1.0
            diff = self._g - result.g_calc
            if mask is not None:
                diff = np.where(mask, diff, np.nan)
            self._diff.setData(result.r, diff - 0.6 * span)

    def update_curves(self, r: np.ndarray, g_calc: np.ndarray) -> None:
        self._gcalc = g_calc
        self._calc.setData(r, g_calc)
        if self._g.size == r.size:
            span = float(np.nanmax(self._g) - np.nanmin(self._g)) if self._g.size else 1.0
            self._diff.setData(r, self._g - g_calc - 0.6 * span)

    def _on_mouse(self, evt: tuple) -> None:  # type: ignore[type-arg]
        pos = evt[0]
        vb = self.plot.getPlotItem().vb
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        x = float(p.x())
        text = f"r = {x:.3f} Å"
        if self._r.size:
            i = int(np.clip(np.searchsorted(self._r, x), 0, len(self._r) - 1))
            text += f"  観測 = {self._g[i]:.3f}"
            if self._gcalc is not None and i < len(self._gcalc):
                text += f"  計算 = {self._gcalc[i]:.3f}"
        self._label.setText(text)
        rng = vb.viewRange()
        self._label.setPos(rng[0][0], rng[1][1])

    def export_image(self, path: str | Path) -> Path:
        return export_plot_image(self.plot, path)


class HistoryPlot(QWidget):
    """精密化の χ² の推移(反復ごと)。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.plot.setLabel("bottom", "反復")
        self.plot.setLabel("left", "χ²")
        self.plot.setLogMode(y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self._curve = self.plot.plot(
            [], [], pen=pg.mkPen("#1f77b4", width=1.5), symbol="o", symbolSize=5,
            symbolBrush="#1f77b4",
        )  # fmt: skip
        self._marks: list[pg.InfiniteLine] = []
        self._values: list[float] = []

    def clear(self) -> None:
        self._values = []
        self._curve.setData([], [])
        for m in self._marks:
            self.plot.removeItem(m)
        self._marks.clear()

    def append(self, chi2: float) -> None:
        if chi2 > 0 and np.isfinite(chi2):
            self._values.append(float(chi2))
            self._curve.setData(np.arange(1, len(self._values) + 1), np.array(self._values))

    def mark_step(self, label: str) -> None:
        """レシピの段階の切れ目に縦線を入れる。"""
        x = len(self._values) + 0.5
        line = pg.InfiniteLine(
            pos=x, angle=90, pen=pg.mkPen("#888", style=Qt.PenStyle.DashLine),
            label=label, labelOpts={"position": 0.9, "color": "#444", "rotateAxis": (1, 0)},
        )  # fmt: skip
        self.plot.addItem(line)
        self._marks.append(line)

    @property
    def values(self) -> list[float]:
        return list(self._values)
