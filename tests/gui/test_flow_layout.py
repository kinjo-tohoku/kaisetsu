"""折り返すレイアウト(FlowLayout)。"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QPushButton, QWidget

from kaisetsu.gui.flow_layout import FlowLayout


def _panel(n: int = 6) -> tuple[QWidget, FlowLayout, list[QPushButton]]:
    w = QWidget()
    lay = FlowLayout(w, h_spacing=4, v_spacing=4)
    buttons = [QPushButton(f"ボタン {i}") for i in range(n)]
    for b in buttons:
        lay.addWidget(b)
    return w, lay, buttons


def test_minimum_width_is_widest_child(qtbot) -> None:
    w, lay, buttons = _panel()
    qtbot.addWidget(w)
    widest = max(b.minimumSizeHint().width() for b in buttons)
    assert lay.minimumSize().width() == widest
    assert lay.count() == len(buttons)
    assert lay.itemAt(0) is not None and lay.itemAt(99) is None


def test_wraps_when_narrow(qtbot) -> None:
    w, lay, buttons = _panel()
    qtbot.addWidget(w)
    one_w = buttons[0].sizeHint().width()
    one_h = buttons[0].sizeHint().height()
    assert lay.hasHeightForWidth()
    # 全部が 1 行に入る幅なら 1 行分、2 つしか入らない幅なら 3 行分の高さ
    assert lay.heightForWidth((one_w + 4) * len(buttons) + 10) == one_h
    assert lay.heightForWidth(2 * one_w + 4 + 1) == 3 * one_h + 2 * 4
    # 実際に配置したとき、2 つ目までが 1 行目、3 つ目は 2 行目に来る
    w.resize(2 * one_w + 4 + 1, 200)
    w.show()
    assert buttons[0].y() == buttons[1].y() < buttons[2].y()
    assert buttons[2].x() == buttons[0].x()


def test_take_at_removes(qtbot) -> None:
    w, lay, buttons = _panel(3)
    qtbot.addWidget(w)
    item = lay.takeAt(1)
    assert item is not None and item.widget() is buttons[1]
    assert lay.count() == 2 and lay.takeAt(5) is None
