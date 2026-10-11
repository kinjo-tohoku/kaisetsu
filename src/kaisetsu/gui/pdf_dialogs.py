"""PDF 関連のダイアログ: G(r) の読み込み条件、I(Q) → G(r) の変換、PDF の標準手順。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.io.readers import read_powder_data
from kaisetsu.pdf.data import PDFData
from kaisetsu.pdf.transform import (
    TransformResult,
    TransformSettings,
    powder_to_q,
    transform,
)
from kaisetsu.refine.recipe import Recipe, pdf_recipe


def _spin(value: float, lo: float, hi: float, decimals: int, step: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    return s


class PDFDataDialog(QDialog):
    """G(r) を読み込んだ後の条件: 放射線、Q_max、Q_damp/Q_broad の初期値、フィット範囲。"""

    def __init__(self, data: PDFData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PDF データの条件")
        form = QFormLayout(self)
        meta = data.metadata
        qmax = (
            float(meta.get("qmax", 0.0) or 0.0)
            if isinstance(meta.get("qmax"), int | float)
            else 0.0
        )
        form.addRow(
            QLabel(
                f"{data.name}: r = {data.r[0]:.2f}〜{data.r[-1]:.2f} Å、{len(data)} 点"
                + (f"、ヘッダの Q_max = {qmax:g} Å⁻¹" if qmax > 0 else "")
            )
        )
        self.radiation = QComboBox()
        self.radiation.addItem("X 線", "xray")
        self.radiation.addItem("中性子", "neutron")
        form.addRow("放射線", self.radiation)
        self.qmax = _spin(qmax if qmax > 0 else 25.0, 0.0, 200.0, 2, 1.0)
        self.qmax.setToolTip(
            "G(r) を作ったときの Q_max。終端効果(リップル)の計算に使う。0 なら無効"
        )
        form.addRow("Q_max [Å⁻¹]", self.qmax)
        self.qdamp = _spin(0.04, 0.0, 1.0, 4, 0.005)
        self.qdamp.setToolTip(
            "装置分解能による G(r) の減衰。放射光 X 線 0.03〜0.07、中性子 0.01〜0.03 程度"
        )
        form.addRow("Q_damp [Å⁻¹]", self.qdamp)
        self.qbroad = _spin(0.0, 0.0, 1.0, 4, 0.005)
        self.qbroad.setToolTip("分解能によるピーク幅の広がり。標準試料で決める。通常は 0〜0.03")
        form.addRow("Q_broad [Å⁻¹]", self.qbroad)
        self.r_min = _spin(max(float(data.r[0]), 1.0), 0.0, 1000.0, 2, 0.5)
        self.r_max = _spin(float(data.r[-1]), 0.0, 1000.0, 2, 0.5)
        row = QHBoxLayout()
        row.addWidget(self.r_min)
        row.addWidget(QLabel("〜"))
        row.addWidget(self.r_max)
        form.addRow("フィット範囲 r [Å]", row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict[str, object]:
        return {
            "radiation": str(self.radiation.currentData()),
            "qmax": float(self.qmax.value()),
            "qdamp": float(self.qdamp.value()),
            "qbroad": float(self.qbroad.value()),
            "r_min": float(self.r_min.value()),
            "r_max": float(self.r_max.value()),
        }


class TransformDialog(QDialog):
    """I(Q)(または 2θ の全散乱データ)から G(r) を作る。プレビュー付き。"""

    def __init__(self, parent: QWidget | None = None, composition: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("I(Q) から G(r) を作る")
        self.resize(980, 640)
        self.result: TransformResult | None = None
        self._q: np.ndarray | None = None
        self._i: np.ndarray | None = None
        self._bg: np.ndarray | None = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 0)

        files = QGroupBox("データ")
        ff = QFormLayout(files)
        self.sample_edit = QLineEdit()
        self.sample_edit.setReadOnly(True)
        b1 = QPushButton("選ぶ…")
        b1.clicked.connect(lambda: self._browse(self.sample_edit))
        r1 = QHBoxLayout()
        r1.addWidget(self.sample_edit, 1)
        r1.addWidget(b1)
        ff.addRow("試料 I(Q)", r1)
        self.bg_edit = QLineEdit()
        self.bg_edit.setReadOnly(True)
        b2 = QPushButton("選ぶ…")
        b2.clicked.connect(lambda: self._browse(self.bg_edit))
        b3 = QPushButton("なし")
        b3.clicked.connect(lambda: self.bg_edit.setText(""))
        r2 = QHBoxLayout()
        r2.addWidget(self.bg_edit, 1)
        r2.addWidget(b2)
        r2.addWidget(b3)
        ff.addRow("容器・空セル(任意)", r2)
        self.x_is_q = QComboBox()
        self.x_is_q.addItem("横軸は Q [Å⁻¹]", True)
        self.x_is_q.addItem("横軸は 2θ [deg](波長で Q に直す)", False)
        ff.addRow("横軸", self.x_is_q)
        self.wavelength = _spin(0.2, 0.01, 10.0, 6, 0.01)
        ff.addRow("波長 [Å]", self.wavelength)
        left.addWidget(files)

        cond = QGroupBox("変換条件")
        cf = QFormLayout(cond)
        self.composition = QLineEdit(composition or "Ni")
        self.composition.setToolTip(
            "組成式(例 Ni, CaF2, Na2Ca3Al2F14)。散乱因子の平均 <f²>, <f>² に使う"
        )
        cf.addRow("組成", self.composition)
        self.radiation = QComboBox()
        self.radiation.addItem("X 線", "xray")
        self.radiation.addItem("中性子", "neutron")
        cf.addRow("放射線", self.radiation)
        self.qmin = _spin(0.5, 0.0, 100.0, 2, 0.1)
        self.qmax = _spin(25.0, 1.0, 200.0, 2, 0.5)
        self.qmax.setToolTip(
            "フーリエ変換の上限。ノイズが大きくなる手前で切る。放射光 20〜30、実験室 Mo で 15 程度"
        )
        self.qmaxinst = _spin(0.0, 0.0, 200.0, 2, 0.5)
        self.qmaxinst.setToolTip(
            "データとして信頼できる Q の上限(多項式補正はここまで当てる)。0 なら Q_max と同じ"
        )
        cf.addRow("Q_min [Å⁻¹]", self.qmin)
        cf.addRow("Q_max [Å⁻¹]", self.qmax)
        cf.addRow("Q_maxinst [Å⁻¹]", self.qmaxinst)
        self.rpoly = _spin(0.9, 0.0, 5.0, 2, 0.1)
        self.rpoly.setToolTip(
            "アドホック多項式補正の強さ。r < r_poly の構造は使えなくなる。0.9 Å が標準"
        )
        cf.addRow("r_poly [Å]", self.rpoly)
        self.bgscale = _spin(1.0, 0.0, 10.0, 3, 0.05)
        cf.addRow("容器の係数", self.bgscale)
        self.polarization = _spin(0.0, 0.0, 1.0, 3, 0.05)
        self.polarization.setToolTip("X 線の偏光係数で補正する(0 なら補正なし)。放射光は 0.9〜1")
        cf.addRow("偏光補正 P", self.polarization)
        self.mu_r = _spin(0.0, 0.0, 20.0, 3, 0.1)
        self.mu_r.setToolTip("円筒試料の μR(0 なら吸収補正なし)")
        cf.addRow("吸収 μR", self.mu_r)
        self.lorch = QCheckBox("Lorch 窓を掛ける(リップルを抑えるがピークが鈍る)")
        cf.addRow(self.lorch)
        self.rmax = _spin(30.0, 5.0, 500.0, 1, 5.0)
        self.rstep = _spin(0.01, 0.001, 0.5, 3, 0.005)
        cf.addRow("r_max [Å]", self.rmax)
        cf.addRow("r 刻み [Å]", self.rstep)
        left.addWidget(cond)
        run = QPushButton("変換してプレビュー")
        run.clicked.connect(self.run_transform)
        left.addWidget(run)
        self.log = QLabel("")
        self.log.setWordWrap(True)
        left.addWidget(self.log)
        left.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("この G(r) をプロジェクトに追加")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        left.addWidget(buttons)

        right = QVBoxLayout()
        root.addLayout(right, 1)
        self.plot_fq = pg.PlotWidget(title="F(Q) = Q[S(Q) − 1]")
        self.plot_fq.setLabel("bottom", "Q (Å⁻¹)")
        self.plot_gr = pg.PlotWidget(title="G(r)")
        self.plot_gr.setLabel("bottom", "r (Å)")
        right.addWidget(self.plot_fq)
        right.addWidget(self.plot_gr)
        self._fq_raw = self.plot_fq.plot([], [], pen=pg.mkPen("#999"), name="補正前")
        self._fq = self.plot_fq.plot([], [], pen=pg.mkPen("r"), name="補正後")
        self._poly = self.plot_fq.plot([], [], pen=pg.mkPen("g", width=1), name="多項式")
        self._gr = self.plot_gr.plot([], [], pen=pg.mkPen("k"))

    def _browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "全散乱データ",
            last_dir(),
            "データ (*.xy *.xye *.dat *.txt *.csv *.chi *.iq *.qye);;すべて (*)",
        )
        if path:
            edit.setText(path)
            remember_dir(path)

    def settings(self) -> TransformSettings:
        return TransformSettings(
            composition=self.composition.text().strip() or "Ni",
            radiation=str(self.radiation.currentData()),
            qmin=float(self.qmin.value()),
            qmax=float(self.qmax.value()),
            qmaxinst=float(self.qmaxinst.value()),
            rpoly=float(self.rpoly.value()),
            bgscale=float(self.bgscale.value()),
            polarization=float(self.polarization.value()) or None,
            wavelength=float(self.wavelength.value()),
            mu_r=float(self.mu_r.value()),
            lorch=self.lorch.isChecked(),
            rmin=0.0,
            rmax=float(self.rmax.value()),
            rstep=float(self.rstep.value()),
        )

    def _load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        data = read_powder_data(path)
        if bool(self.x_is_q.currentData()):
            return data.x.copy(), data.y.copy()
        return powder_to_q(data, float(self.wavelength.value()))

    def run_transform(self) -> None:
        if not self.sample_edit.text():
            QMessageBox.information(self, "変換", "試料の I(Q) ファイルを選んでください")
            return
        try:
            q, i = self._load(self.sample_edit.text())
            bg = None
            if self.bg_edit.text():
                qb, ib = self._load(self.bg_edit.text())
                bg = np.interp(q, qb, ib)
            self.result = transform(q, i, self.settings(), background=bg)
        except Exception as exc:
            QMessageBox.critical(self, "変換エラー", str(exc))
            return
        res = self.result
        self._fq_raw.setData(res.q, np.nan_to_num(res.fq_raw))
        self._fq.setData(res.q, np.nan_to_num(res.fq))
        self._poly.setData(res.q, res.polynomial)
        self._gr.setData(res.r, res.gr)
        self.log.setText("\n".join(res.log))

    def _accept(self) -> None:
        if self.result is None:
            self.run_transform()
        if self.result is None:
            return
        self.accept()

    def pdf_data(self) -> PDFData | None:
        if self.result is None:
            return None
        name = Path(self.sample_edit.text()).stem if self.sample_edit.text() else "transformed"
        return self.result.to_pdf_data(name)


class PDFRecipeDialog(QDialog):
    """PDF の標準手順のオプション。"""

    def __init__(self, phase_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PDF の標準手順で精密化")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "PDFgui と同じ順序で精密化します:\n"
            "1. スケール → 2. 格子定数 → 3. 変位パラメータと δ2 → 4. Q_damp → 5. 原子座標 → 6. 最終"
        ))  # fmt: skip
        form = QFormLayout()
        self.phase = QComboBox()
        self.phase.addItem("すべての相", "*")
        for n in phase_names:
            self.phase.addItem(n, n)
        form.addRow("対象の相", self.phase)
        self.coords = QCheckBox("原子座標を動かす")
        self.coords.setChecked(True)
        self.adp = QCheckBox("変位パラメータ(U)を動かす")
        self.adp.setChecked(True)
        self.delta2 = QCheckBox("相関運動補正 δ2 を動かす(低温で支配的な 1/r² 項)")
        self.delta2.setChecked(True)
        self.delta1 = QCheckBox("相関運動補正 δ1 を動かす(高温で支配的な 1/r 項)")
        self.qdamp = QCheckBox("Q_damp を動かす(標準試料で決めた値があれば外す)")
        self.qdamp.setChecked(True)
        self.qbroad = QCheckBox("Q_broad を動かす")
        self.spd = QCheckBox("球状粒子の直径を動かす(ナノ粒子)")
        self.occ = QCheckBox("占有率を動かす(混合サイト・部分占有のサイトだけ)")
        for w in (
            self.coords,
            self.adp,
            self.delta2,
            self.delta1,
            self.qdamp,
            self.qbroad,
            self.spd,
            self.occ,
        ):
            form.addRow(w)
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 200)
        self.cycles.setValue(15)
        form.addRow("各段階の最大反復回数", self.cycles)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def recipe(self) -> Recipe:
        r = pdf_recipe(
            str(self.phase.currentData()),
            "g*",
            coordinates=self.coords.isChecked(),
            adp=self.adp.isChecked(),
            delta2=self.delta2.isChecked(),
            delta1=self.delta1.isChecked(),
            qdamp=self.qdamp.isChecked(),
            qbroad=self.qbroad.isChecked(),
            spdiameter=self.spd.isChecked(),
            occupancy=self.occ.isChecked(),
        )
        for s in r.steps:
            s.max_cycles = (
                max(self.cycles.value(), 5)
                if s.name != "最終(全パラメータ)"
                else self.cycles.value() * 2
            )
        return r


__all__ = ["PDFDataDialog", "PDFRecipeDialog", "TransformDialog"]
