"""図の上のマウス操作(本物のイベント): Shift+ドラッグで除外、普通のドラッグはパン、帯の移動と右クリック削除。

pyqtgraph は移動イベントの間隔を制限し、押してから 0.5 秒未満の短い移動はクリック扱いにするので、
QTest.qWait で間隔を空ける。ドラッグの修飾キーは最初の移動イベントから取られる。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kaisetsu.gui.plot_widget import PatternPlot
from kaisetsu.io import read_powder_data

pytestmark = pytest.mark.slow


class _Driver:
    def __init__(self, plot: PatternPlot) -> None:
        self.plot = plot
        self.view = plot.plot
        self.vb = self.view.getPlotItem().vb
        self.vp = self.view.viewport()

    def pt(self, x: float) -> QPoint:
        rng = self.vb.viewRange()
        ymid = 0.5 * (rng[1][0] + rng[1][1])
        return self.view.mapFromScene(self.vb.mapViewToScene(QPointF(x, ymid)))

    def drag(self, x0: float, x1: float, mod: Qt.KeyboardModifier, steps: int = 5) -> None:
        a, b = self.pt(x0), self.pt(x1)
        QTest.mousePress(self.vp, Qt.MouseButton.LeftButton, mod, a)
        QTest.qWait(30)
        for k in range(1, steps + 1):
            # QTest.mouseMove は修飾キーを付けられないので自前で送る(Shift を押したままの移動)
            pos = QPoint(a.x() + (b.x() - a.x()) * k // steps, a.y())
            ev = QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(pos),
                QPointF(self.vp.mapToGlobal(pos)),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                mod,
            )
            QApplication.sendEvent(self.vp, ev)
            QTest.qWait(60)
        QTest.mouseRelease(self.vp, Qt.MouseButton.LeftButton, mod, b)
        QTest.qWait(30)

    def xrange(self) -> tuple[float, float]:
        lo, hi = self.vb.viewRange()[0]
        return float(lo), float(hi)


@pytest.fixture
def driver(qtbot, data_dir: Path) -> _Driver:  # type: ignore[no-untyped-def]
    plot = PatternPlot()
    qtbot.addWidget(plot)
    plot.resize(900, 500)
    plot.show()
    plot.set_data(read_powder_data(data_dir / "fap" / "FAP.XRA"), 1.5406)
    d = _Driver(plot)
    d.vb.updateAutoRange()
    d.vb.disableAutoRange()
    plot.plot.grab()  # 一度描いて座標変換を確定させる
    QTest.qWait(30)
    return d


def test_shift_drag_requests_exclusion_and_plain_drag_pans(driver: _Driver) -> None:
    got: list[tuple[float, float]] = []
    driver.plot.excludeRequested.connect(lambda lo, hi: got.append((lo, hi)))
    r0 = driver.xrange()
    driver.drag(40.0, 50.0, Qt.KeyboardModifier.ShiftModifier)
    assert len(got) == 1
    lo, hi = got[0]
    assert lo == pytest.approx(40.0, abs=0.5) and hi == pytest.approx(50.0, abs=0.5)
    assert driver.xrange() == pytest.approx(r0, abs=1e-6)  # Shift のときはパンしない
    driver.drag(90.0, 100.0, Qt.KeyboardModifier.NoModifier)
    assert len(got) == 1  # 普通のドラッグでは除外しない
    r1 = driver.xrange()
    assert r1[0] == pytest.approx(r0[0] - 10.0, abs=0.5)  # 右にドラッグ → 表示が左へ 10°


def test_band_drag_moves_region_and_right_click_removes(driver: _Driver) -> None:
    changed: list[list[tuple[float, float]]] = []
    driver.plot.excludedChanged.connect(lambda r: changed.append(list(r)))
    driver.plot.set_excluded([(70.0, 72.0)])
    QTest.qWait(30)
    r0 = driver.xrange()
    driver.drag(71.0, 81.0, Qt.KeyboardModifier.NoModifier)
    assert driver.xrange() == pytest.approx(r0, abs=1e-6)  # 帯の上ではパンしない
    assert len(changed) >= 1
    lo, hi = changed[-1][0]
    assert lo == pytest.approx(80.0, abs=0.5) and hi == pytest.approx(82.0, abs=0.5)
    assert len(driver.plot._regions) == 1
    QTest.mouseClick(
        driver.vp, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, driver.pt(81.0)
    )
    QTest.qWait(30)
    assert changed[-1] == []
    assert driver.plot._regions == []
