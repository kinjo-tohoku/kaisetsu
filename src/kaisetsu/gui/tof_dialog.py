"""TOF 中性子回折計の装置パラメータダイアログ。GSAS-II の .instprm を読める。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
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

from kaisetsu.gui.dialog_utils import fit_dialog_to_screen
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
    ("z", "Z [μs]", 4, -1e4, 1e4, ""),
)
#: GSAS type 3 だけで使う(Type0m では無効にする)
_GSAS3_ONLY = ("alpha", "beta0", "beta1", "betaq")
_ZR_FIELDS: tuple[tuple[str, str, int, float, float, str], ...] = (
    ("zr_alpha1", "Alpha1", 5, 1e-3, 100.0, "立ち上がり 1/α' = Alpha1 + Alpha2 d"),
    ("zr_alpha2", "Alpha2", 5, -50.0, 50.0, ""),
    ("zr_beta1", "Beta1_0", 5, 1e-3, 100.0, "短い裾の減衰 β₁'(定数)。時定数 = 係数 × ピークの FWHM"),
    ("zr_beta0_0", "Beta0_0", 5, 1e-3, 100.0, "長い裾の減衰 β₀' = Beta0_0 + Beta0_1 / d"),
    ("zr_beta0_1", "Beta0_1", 5, -100.0, 100.0, ""),
    ("zr_r01", "r01", 5, 0.0, 1.0, "長い裾の割合 R = r01 exp(−r02 / d^r03)"),
    ("zr_r02", "r02", 5, 0.0, 100.0, ""),
    ("zr_r03", "r03", 5, 0.0, 20.0, ""),
)  # fmt: skip


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
            "パルス中性子(J-PARC、SNS、ISIS)の飛行時間データ。装置の .instprm(GSAS-II)か"
            " .zDiffractometer(Z-Rietveld)があれば読み込んでください。\n"
            "幅のパラメータは GSAS-II の TOF プロファイル(type 3)と同じ定義です。Z-Rietveld の"
            "Type0m(J-PARC)は換算係数・σ²・γ・α・β・R をすべて取り込み、同じ関数で計算します。"
        ))  # fmt: skip
        row = QHBoxLayout()
        b = QPushButton("装置ファイルを読む(.instprm / .zDiffractometer)…")
        b.clicked.connect(self.load_instprm)
        row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)
        krow = QHBoxLayout()
        krow.addWidget(QLabel("プロファイル関数"))
        self.kind = QComboBox()
        self.kind.addItem("GSAS type 3(両側指数 α, β ⊗ 擬 Voigt)", "gsas3")
        self.kind.addItem("Z-Rietveld Type0m(J-PARC の装置ファイル)", "zrietveld")
        self.kind.currentIndexChanged.connect(self._update_enabled)
        krow.addWidget(self.kind, 1)
        layout.addLayout(krow)
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
        self.zr_box = QGroupBox("Z-Rietveld Type0m のパラメータ(装置ファイルの値)")
        zf = QFormLayout(self.zr_box)
        self.zr_spins: dict[str, QDoubleSpinBox] = {}
        for attr, label, decimals, lo, hi, tip in _ZR_FIELDS:
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(decimals)
            sp.setValue(float(getattr(inst, attr)))
            if tip:
                sp.setToolTip(tip)
            zf.addRow(label, sp)
            self.zr_spins[attr] = sp
        layout.addWidget(self.zr_box)
        self.kind.setCurrentIndex(max(self.kind.findData(inst.profile_kind), 0))
        self._update_enabled()
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
        fit_dialog_to_screen(self, self.buttons)

    def load_instprm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "装置パラメータ(GSAS-II .instprm / Z-Rietveld .zDiffractometer)",
            last_dir(),
            "装置パラメータ (*.instprm *.zDiffractometer);;すべて (*)",
        )
        if not path:
            return
        try:
            inst = TOFInstrument.from_instrument_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みエラー", str(exc))
            return
        remember_dir(path)
        self.set_instrument(inst)

    def _update_enabled(self) -> None:
        zr = self.kind.currentData() == "zrietveld"
        self.zr_box.setEnabled(zr)
        for attr in _GSAS3_ONLY:
            self.spins[attr].setEnabled(not zr)

    def set_instrument(self, inst: TOFInstrument) -> None:
        for attr, sp in self.spins.items():
            sp.setValue(float(getattr(inst, attr)))
        for attr, sp in self.zr_spins.items():
            sp.setValue(float(getattr(inst, attr)))
        self.kind.setCurrentIndex(max(self.kind.findData(inst.profile_kind), 0))
        self._update_enabled()

    def result_instrument(self) -> TOFInstrument:
        kw: dict[str, object] = {attr: float(sp.value()) for attr, sp in self.spins.items()}
        kw.update({attr: float(sp.value()) for attr, sp in self.zr_spins.items()})
        kw["profile_kind"] = str(self.kind.currentData())
        return TOFInstrument(**kw)

    def result_sample(self) -> SampleGeometry:
        """試料形状(円筒、μR を更新したもの)。"""
        from dataclasses import replace

        return replace(
            self._sample, kind=GeometryKind.DEBYE_SCHERRER, mu_r=float(self.mu_r.value())
        )


__all__ = ["TOFInstrumentDialog"]
