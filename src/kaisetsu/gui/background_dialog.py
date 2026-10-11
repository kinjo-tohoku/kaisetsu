"""背景(バックグラウンド)の設定ダイアログ。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)


class BackgroundDialog(QDialog):
    """背景の種類(Chebyshev / 固定点補間)、項数、観測からの再推定、非晶質ハンプを選ぶ。"""

    def __init__(
        self,
        n_terms: int,
        kind: str = "chebyshev",
        n_humps: int = 0,
        x_range: tuple[float, float] = (10.0, 90.0),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("背景の設定")
        form = QFormLayout(self)
        note = QLabel(
            "Chebyshev 多項式は滑らかな背景向き(項数の目安 6〜12)。\n"
            "固定点補間は、低角の急な立ち上がりや複雑な形で多項式が追従しないときに\n"
            "(点の数 8〜20。各点の高さが精密化パラメータ)。\n"
            "非晶質・ガラス相のハンプ(幅の広い山)は「ハンプ」で足します。"
        )
        form.addRow(note)
        self.kind = QComboBox()
        self.kind.addItem("Chebyshev 多項式", "chebyshev")
        self.kind.addItem("固定点の線形補間", "points")
        self.kind.setCurrentIndex(1 if kind == "points" else 0)
        form.addRow("種類", self.kind)
        self.n_terms = QSpinBox()
        self.n_terms.setRange(1, 60)
        self.n_terms.setValue(n_terms)
        form.addRow("項数(固定点の数)", self.n_terms)
        self.estimate = QCheckBox(
            "観測パターンから背景を推定し直す(ピークを避けて滑らかな曲線を当てる)"
        )
        self.estimate.setChecked(True)
        form.addRow(self.estimate)

        hump = QGroupBox("非晶質ハンプ(ガウス関数)")
        hf = QFormLayout(hump)
        self.n_humps = QSpinBox()
        self.n_humps.setRange(0, 5)
        self.n_humps.setValue(n_humps)
        hf.addRow("個数", self.n_humps)
        row = QHBoxLayout()
        self.hump_center = QDoubleSpinBox()
        self.hump_center.setRange(0.0, 180.0)
        self.hump_center.setDecimals(1)
        self.hump_center.setValue(0.5 * (x_range[0] + x_range[1]))
        self.hump_fwhm = QDoubleSpinBox()
        self.hump_fwhm.setRange(0.5, 90.0)
        self.hump_fwhm.setDecimals(1)
        self.hump_fwhm.setValue(10.0)
        row.addWidget(QLabel("中心 [deg]"))
        row.addWidget(self.hump_center)
        row.addWidget(QLabel("半値全幅 [deg]"))
        row.addWidget(self.hump_fwhm)
        hf.addRow("新しいハンプの初期値", row)
        hf.addRow(QLabel("高さは観測から自動で決めます。複数個のときは中心を等間隔にずらします。"))
        form.addRow(hump)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[int, bool]:
        return int(self.n_terms.value()), bool(self.estimate.isChecked())

    def settings(self) -> dict[str, object]:
        return {
            "kind": str(self.kind.currentData()),
            "n_terms": int(self.n_terms.value()),
            "estimate": bool(self.estimate.isChecked()),
            "n_humps": int(self.n_humps.value()),
            "hump_center": float(self.hump_center.value()),
            "hump_fwhm": float(self.hump_fwhm.value()),
        }
