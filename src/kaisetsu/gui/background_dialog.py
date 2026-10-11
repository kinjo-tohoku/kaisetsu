"""背景(バックグラウンド)の設定ダイアログ。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)


class BackgroundDialog(QDialog):
    """Chebyshev 多項式の項数と、観測からの再推定を選ぶ。"""

    def __init__(self, n_terms: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("背景の設定")
        form = QFormLayout(self)
        note = QLabel(
            "背景は Chebyshev 多項式で表します。項数の目安は 6〜12。\n"
            "低角の立ち上がりや緩やかなうねりが合わないときは増やし、\n"
            "ピークが密で背景がピークを吸い始めたら減らします。"
        )
        form.addRow(note)
        self.n_terms = QSpinBox()
        self.n_terms.setRange(1, 36)
        self.n_terms.setValue(n_terms)
        form.addRow("項数", self.n_terms)
        self.estimate = QCheckBox(
            "観測パターンから背景を推定し直す(ピークを避けて滑らかな曲線を当てる)"
        )
        self.estimate.setChecked(True)
        form.addRow(self.estimate)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[int, bool]:
        return int(self.n_terms.value()), bool(self.estimate.isChecked())
