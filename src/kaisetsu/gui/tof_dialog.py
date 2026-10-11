"""TOF 中性子回折計の装置パラメータダイアログ。GSAS-II の .instprm を読める。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.pattern.instrument import GeometryKind, SampleGeometry
from kaisetsu.pattern.tof import TOFInstrument

_FIELDS: tuple[tuple[str, str, int, float, float, str], ...] = (
    # (属性, 表示名, 小数桁, 下限, 上限, 説明)
    (
        "difc",
        "DIFC [μs/Å]",
        4,
        1.0,
        1e6,
        "TOF = DIFC d + DIFA d² + DIFB/d + ZERO。Mantid の出力ヘッダや .instprm にある",
    ),
    ("difa", "DIFA [μs/Å²]", 5, -1e4, 1e4, "d² の係数(通常は小さい)"),
    ("difb", "DIFB [μs·Å]", 5, -1e4, 1e4, "1/d の係数(通常は小さい)"),
    ("zero", "ZERO [μs]", 4, -1e4, 1e4, "ゼロ点"),
    ("two_theta", "検出器バンクの 2θ [deg]", 2, 1.0, 179.0, "Lorentz 因子 d⁴ sinθ と d 範囲に使う"),
    ("alpha", "alpha", 6, 1e-4, 10.0, "立ち上がり α = alpha / d [1/μs]"),
    ("beta0", "beta-0", 6, 0.0, 10.0, "減衰 β = beta-0 + beta-1/d⁴ + beta-q/d²"),
    ("beta1", "beta-1", 6, 0.0, 10.0, ""),
    ("betaq", "beta-q", 6, 0.0, 10.0, ""),
    ("sig0", "sig-0 [μs²]", 3, -1e6, 1e6, "ガウス分散 σ² = sig-0 + sig-1 d² + sig-2 d⁴ + sig-q d"),
    ("sig1", "sig-1 [μs²/Å²]", 3, -1e6, 1e6, ""),
    ("sig2", "sig-2 [μs²/Å⁴]", 3, -1e6, 1e6, ""),
    ("sigq", "sig-q [μs²/Å]", 3, -1e6, 1e6, ""),
    ("x", "X [μs/Å]", 4, 0.0, 1e4, "ローレンツ幅 γ = X d + Y d² + Z"),
    ("y", "Y [μs/Å²]", 4, 0.0, 1e4, ""),
    ("z", "Z [μs]", 4, 0.0, 1e4, ""),
)


class TOFInstrumentDialog(QDialog):
    """TOF の装置パラメータを編集する。"""

    def __init__(
        self,
        inst: TOFInstrument | None,
        parent: QWidget | None = None,
        sample: SampleGeometry | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("TOF 装置パラメータ")
        inst = inst or TOFInstrument(difc=20000.0)
        self._sample = sample or SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "パルス中性子(J-PARC、SNS、ISIS)の飛行時間データ。装置の .instprm(GSAS-II 形式)があれば"
            "読み込んでください。\n幅のパラメータは GSAS-II の TOF プロファイル(type 3)と同じ定義です。"
        ))  # fmt: skip
        row = QHBoxLayout()
        b = QPushButton("GSAS-II の .instprm を読む…")
        b.clicked.connect(self.load_instprm)
        row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)
        box = QGroupBox("パラメータ")
        form = QFormLayout(box)
        self.spins: dict[str, QDoubleSpinBox] = {}
        for attr, label, decimals, lo, hi, tip in _FIELDS:
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(decimals)
            sp.setValue(float(getattr(inst, attr)))
            if tip:
                sp.setToolTip(tip)
            form.addRow(label, sp)
            self.spins[attr] = sp
        layout.addWidget(box)
        smp = QGroupBox("試料")
        sf = QFormLayout(smp)
        self.mu_r = QDoubleSpinBox()
        self.mu_r.setRange(0.0, 20.0)
        self.mu_r.setDecimals(4)
        self.mu_r.setSingleStep(0.01)
        self.mu_r.setValue(float(self._sample.mu_r))
        self.mu_r.setToolTip(
            "円筒試料(バナジウム缶)の μR を λ = 1 Å で与える。反射ごとに λ = 2 d sinθ に比例させて"
            "透過率を補正する。0 で補正なし。精密化は h0:sample:mu_r"
        )
        sf.addRow("吸収 μR(λ = 1 Å)", self.mu_r)
        layout.addWidget(smp)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def load_instprm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "GSAS-II の装置パラメータ", last_dir(), "instprm (*.instprm);;すべて (*)"
        )
        if not path:
            return
        try:
            inst = TOFInstrument.from_instprm(path)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みエラー", str(exc))
            return
        remember_dir(path)
        self.set_instrument(inst)

    def set_instrument(self, inst: TOFInstrument) -> None:
        for attr, sp in self.spins.items():
            sp.setValue(float(getattr(inst, attr)))

    def result_instrument(self) -> TOFInstrument:
        return TOFInstrument(**{attr: float(sp.value()) for attr, sp in self.spins.items()})

    def result_sample(self) -> SampleGeometry:
        """試料形状(円筒、μR を更新したもの)。"""
        from dataclasses import replace

        return replace(
            self._sample, kind=GeometryKind.DEBYE_SCHERRER, mu_r=float(self.mu_r.value())
        )


__all__ = ["TOFInstrumentDialog"]
