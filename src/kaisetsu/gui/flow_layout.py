"""横幅が足りないときに次の行へ折り返すレイアウト。

QHBoxLayout は子の最小幅の合計より狭くできないので、ボタンを横に並べた行がパネル全体の
最小幅を決めてしまう(ノート PC でパラメータ欄を狭くできない原因)。このレイアウトの最小幅は
一番幅の広い子 1 つ分で、幅が足りなければ子を次の行に送る。Qt の flowlayout の例と同じ考え方。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QStyle, QWidget


class FlowLayout(QLayout):
    """子を左から右に並べ、幅が足りなければ折り返す。stretch は無い(左詰め)。"""

    def __init__(
        self, parent: QWidget | None = None, h_spacing: int = -1, v_spacing: int = -1
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    # --- QLayout の必須メソッド -----------------------------------------
    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    # --- 配置 ------------------------------------------------------------
    def horizontal_spacing(self) -> int:
        if self._h_spacing >= 0:
            return self._h_spacing
        return self._smart_spacing(QStyle.PixelMetric.PM_LayoutHorizontalSpacing)

    def vertical_spacing(self) -> int:
        if self._v_spacing >= 0:
            return self._v_spacing
        return self._smart_spacing(QStyle.PixelMetric.PM_LayoutVerticalSpacing)

    def _smart_spacing(self, metric: QStyle.PixelMetric) -> int:
        """親がウィジェットならそのスタイルの間隔、親がレイアウトならその間隔。無ければ 6。"""
        parent = self.parent()
        if isinstance(parent, QWidget):
            return parent.style().pixelMetric(metric, None, parent)
        if isinstance(parent, QLayout):
            return parent.spacing()
        return 6

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        """子を rect に並べて、必要な高さを返す。test_only なら高さの計算だけ。"""
        m = self.contentsMargins()
        inner = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = inner.x()
        y = inner.y()
        line_height = 0
        h_space = self.horizontal_spacing()
        v_space = self.vertical_spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + h_space
            if next_x - h_space > inner.right() + 1 and line_height > 0:
                x = inner.x()
                y = y + line_height + v_space
                next_x = x + hint.width() + h_space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()
