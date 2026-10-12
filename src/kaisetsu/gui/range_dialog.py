"""データ範囲(x_min, x_max)と除外領域を数値で編集するダイアログ。

図の上で Shift + ドラッグしても除外領域は作れる(plot_widget)。ここは数値で正確に
指定したい人と、一覧で確認・削除したい人のため。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.dialog_utils import fit_dialog_to_screen
from kaisetsu.refine.project import HistogramEntry


class RangeDialog(QDialog):
    """1 つのヒストグラムのデータ範囲と除外領域を編集する。"""

    def __init__(self, hist: HistogramEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("データ範囲と除外領域")
        x = hist.data.x
        self._x0, self._x1 = float(x[0]), float(x[-1])
        self._unit = "μs" if hist.is_tof else "deg"
        decimals = 1 if hist.is_tof else 3
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.x_min = QDoubleSpinBox()
        self.x_max = QDoubleSpinBox()
        for s in (self.x_min, self.x_max):
            s.setRange(self._x0, self._x1)
            s.setDecimals(decimals)
            s.setSingleStep(1.0 if hist.is_tof else 0.1)
        self.x_min.setValue(max(self._x0, float(hist.x_min)))
        self.x_max.setValue(min(self._x1, float(hist.x_max)))
        row = QHBoxLayout()
        row.addWidget(self.x_min)
        row.addWidget(QLabel("〜"))
        row.addWidget(self.x_max)
        row.addWidget(QLabel(self._unit))
        row.addStretch(1)
        form.addRow("フィットに使う範囲", row)
        layout.addLayout(form)

        layout.addWidget(
            QLabel(
                "除外領域(この範囲の点は精密化と R 因子から外します。試料ホルダや不純物のピークなどに)"
            )
        )
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([f"開始 [{self._unit}]", f"終了 [{self._unit}]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        for lo, hi in hist.excluded:
            self._append_row(lo, hi)
        layout.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.btn_add = QPushButton("追加")
        self.btn_remove = QPushButton("選択した行を削除")
        self.btn_clear = QPushButton("全部削除")
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(lambda: self.table.setRowCount(0))
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_remove)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        layout.addLayout(btns)
        hint = QLabel(
            "図の上で Shift を押しながらドラッグしても除外領域を作れます。"
            "帯はドラッグで動かせ、右クリックで削除できます。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444;")
        layout.addWidget(hint)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.resize(460, 420)
        fit_dialog_to_screen(self, self.buttons)

    # ------------------------------------------------------------------
    def _append_row(self, lo: float, hi: float) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        fmt = "{:.1f}" if self._unit == "μs" else "{:.3f}"
        self.table.setItem(r, 0, QTableWidgetItem(fmt.format(lo)))
        self.table.setItem(r, 1, QTableWidgetItem(fmt.format(hi)))

    def _add_row(self) -> None:
        # 既定は範囲の真ん中に幅 1% の領域(そのまま書き換えてもらう)
        span = self._x1 - self._x0
        mid = 0.5 * (self._x0 + self._x1)
        self._append_row(mid - 0.005 * span, mid + 0.005 * span)
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    # ------------------------------------------------------------------
    def excluded(self) -> list[tuple[float, float]]:
        """表の内容を (開始, 終了) の一覧として返す(数値でない行は無視、逆順は直す)。"""
        out: list[tuple[float, float]] = []
        for r in range(self.table.rowCount()):
            try:
                a = float(self.table.item(r, 0).text())
                b = float(self.table.item(r, 1).text())
            except (AttributeError, ValueError):
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            if hi > lo:
                out.append((lo, hi))
        return out

    def values(self) -> tuple[float, float, list[tuple[float, float]]]:
        """(x_min, x_max, 除外領域)。x_min > x_max なら入れ替える。"""
        a, b = self.x_min.value(), self.x_max.value()
        if a > b:
            a, b = b, a
        return a, b, self.excluded()
