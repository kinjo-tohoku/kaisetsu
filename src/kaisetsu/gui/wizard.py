"""はじめての精密化ウィザード: データ → 相 → 装置 → 実行。"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from kaisetsu.crystal.cif import read_structure
from kaisetsu.gui.dialog_utils import available_height
from kaisetsu.gui.dialogs import InstrumentDialog
from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.io.readers import read_powder_data
from kaisetsu.refine.recipe import standard_recipe

if TYPE_CHECKING:
    from kaisetsu.gui.main_window import MainWindow


class _FilePage(QWizardPage):
    def __init__(self, title: str, subtitle: str, filt: str, parent: QWizard | None = None) -> None:
        super().__init__(parent)
        self.setTitle(title)
        self.setSubTitle(subtitle)
        self._filter = filt
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        btn = QPushButton("ファイルを選ぶ…")
        btn.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(btn)
        layout.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        layout.addStretch(1)
        self.registerField(title + "*", self.edit)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.title(), last_dir(), self._filter)
        if path:
            remember_dir(path)
        if path:
            self.edit.setText(path)
            self.info.setText(self.describe(path))
            self.completeChanged.emit()

    def describe(self, path: str) -> str:
        return path

    def isComplete(self) -> bool:
        return bool(self.edit.text())


class DataPage(_FilePage):
    def __init__(self, parent: QWizard | None = None) -> None:
        super().__init__(
            "1. 回折データ",
            "測定した粉末回折パターンのファイルを選びます(xy / xye / GSAS 形式など)。",
            "回折データ (*.xy *.xye *.chi *.dat *.txt *.gsa *.gsas *.fxye *.raw *.xra *.cwn *.cif);;"
            "すべて (*)",
            parent,
        )

    def describe(self, path: str) -> str:
        try:
            d = read_powder_data(path)
        except Exception as exc:
            return f"読み込めません: {exc}"
        return f"{len(d)} 点、2θ = {d.x[0]:.3f}〜{d.x[-1]:.3f}°、最大強度 {d.y.max():.0f}"


class PhasePage(_FilePage):
    def __init__(self, parent: QWizard | None = None) -> None:
        super().__init__(
            "2. 結晶構造(CIF)",
            "試料に含まれる相の結晶構造を CIF ファイルで指定します。",
            "CIF (*.cif);;すべて (*)",
            parent,
        )

    def describe(self, path: str) -> str:
        try:
            st = read_structure(path)
        except Exception as exc:
            return f"読み込めません: {exc}"
        return f"{st.name}: 空間群 {st.spacegroup.hm}, {st.lattice}, {st.n_sites} サイト"


class InstrumentPage(QWizardPage):
    def __init__(self, parent: QWizard | None = None) -> None:
        super().__init__(parent)
        self.setTitle("3. 装置")
        self.setSubTitle("使った装置に近いプリセットを選び、必要なら波長や光学系を直します。")
        layout = QVBoxLayout(self)
        # ダイアログの中身だけ埋め込む(OK/Cancel ボタンは隠す)
        self.dialog = InstrumentDialog(None, None, self)
        self.dialog.buttons.hide()
        layout.addWidget(self.dialog)
        self.dialog.preset.setCurrentIndex(1)


class RunPage(QWizardPage):
    def __init__(self, parent: QWizard | None = None) -> None:
        super().__init__(parent)
        self.setTitle("4. 精密化")
        self.setSubTitle(
            "「完了」を押すと、推奨の順序で精密化を実行します(途中経過はログに出ます)。"
        )
        layout = QVBoxLayout(self)
        self.cell_scan = QCheckBox("格子定数の初期値を ±1% 振って、ピークが重なる谷から始める")
        self.cell_scan.setChecked(True)
        self.cell_scan.setToolTip(
            "CIF の格子定数が試料と少し違っても(温度・組成)、最初に ±1% の範囲で振って"
            "計算ピークが観測ピークに重なる初期値を見つけます。数秒で済みます"
        )
        self.coords = QCheckBox("原子座標も動かす")
        self.coords.setChecked(True)
        self.adp = QCheckBox("変位パラメータ(U)も動かす")
        self.adp.setChecked(True)
        self.disp = QCheckBox(
            "ゼロ点ではなく試料変位を動かす(平板: 高さずれ、キャピラリ: x, y ずれ)"
        )
        self.rough = QCheckBox("表面粗さも動かす(重い元素など吸収の強い試料。Bragg-Brentano)")
        self.ka2 = QCheckBox("Kα2 なしも試して良い方を採る(単色化されたデータかどうか)")
        self.ka2.setChecked(True)
        self.light = QCheckBox("軽原子の U_iso を共通にする(重原子が散乱を支配するとき)")
        self.light.setChecked(True)
        self.humps = QCheckBox("残差の広いこぶを背景のハンプにして試す(ガラスキャピラリ・非晶質)")
        self.humps.setChecked(True)
        self.humps.setToolTip(
            "ピーク幅のあとで残差に幅の広い山が残っていれば、背景にガウスのハンプを足して精密化し、"
            "Rwp が 0.5 ポイント以上下がれば採用します(ログに中心と幅が出ます)"
        )
        self.lebail = QCheckBox(
            "Le Bail(構造を使わない)も走らせて、プロファイルと背景だけの Rwp の下限を出す"
        )
        self.lebail.setChecked(True)
        self.lebail.setToolTip(
            "Rietveld の Rwp が Le Bail より大きく悪ければ、残りの差は構造モデルの問題"
            "(占有率・選択配向・第二相)です。差が小さければピーク形状や背景の限界です"
        )
        for w in (
            self.cell_scan,
            self.coords,
            self.adp,
            self.disp,
            self.rough,
            self.ka2,
            self.light,
            self.humps,
            self.lebail,
        ):
            layout.addWidget(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("背景(Chebyshev)の項数:"))
        self.n_background = QSpinBox()
        self.n_background.setRange(1, 36)
        self.n_background.setValue(12)
        self.n_background.setToolTip(
            "目安 8〜16。低角の立ち上がりやキャピラリのこぶが合わなければ増やす"
            "(公開データ 13 本では 8 → 16 項で 8 本が改善)"
        )
        row.addWidget(self.n_background)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(QLabel(
            "手順: スケールと背景 → 格子定数とピーク位置 → ピーク幅 → (Kα2 なしを試す) → "
            "原子座標 → 変位パラメータ\n"
            "Rwp が大きく変わらなくなれば収束です。結果はパラメータ表と「結果の要約」で確認できます。"
        ))  # fmt: skip
        layout.addStretch(1)


class StartWizard(QWizard):
    def __init__(self, main: MainWindow) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("はじめての精密化")
        # ノート PC(有効 500〜700 px)でも「次へ」が見えるように画面の高さに合わせる
        self.resize(760, max(420, min(640, available_height(self) - 80)))
        self.page_data = DataPage(self)
        self.page_phase = PhasePage(self)
        self.page_inst = InstrumentPage(self)
        self.page_run = RunPage(self)
        for p in (self.page_data, self.page_phase, self.page_inst, self.page_run):
            self.addPage(p)
        self.setButtonText(QWizard.WizardButton.NextButton, "次へ >")
        self.setButtonText(QWizard.WizardButton.BackButton, "< 戻る")
        self.setButtonText(QWizard.WizardButton.FinishButton, "完了(精密化を開始)")
        self.setButtonText(QWizard.WizardButton.CancelButton, "キャンセル")
        self.page_data.edit.textChanged.connect(self._apply_data_hints)

    def _apply_data_hints(self, path: str) -> None:
        """データファイルにターゲット・波長があれば装置ページに反映する。"""
        if not path:
            return
        try:
            meta = read_powder_data(path).metadata
        except Exception:
            return
        self.page_inst.dialog.apply_hints(meta)

    def accept(self) -> None:
        inst, smp = self.page_inst.dialog.result_instrument()
        # Le Bail 並走は精密化前の装置から始める(本体の精密化が inst を書き換えるので複製)
        inst_copy, smp_copy = copy.deepcopy(inst), copy.deepcopy(smp)
        main = self.main
        main.load_data_file(
            self.page_data.edit.text(), inst, smp, n_background=self.page_run.n_background.value()
        )
        main.add_phase_file(
            self.page_phase.edit.text(), name=_phase_name(self.page_phase.edit.text())
        )
        recipe = standard_recipe(
            "*",
            "h*",
            coordinates=self.page_run.coords.isChecked(),
            adp=self.page_run.adp.isChecked(),
            sample_displacement=self.page_run.disp.isChecked(),
            roughness=self.page_run.rough.isChecked(),
            try_no_kalpha2=self.page_run.ka2.isChecked(),
            light_atom_uiso=self.page_run.light.isChecked(),
            humps=self.page_run.humps.isChecked(),
            cell_scan=self.page_run.cell_scan.isChecked(),
        )
        companion = None
        if self.page_run.lebail.isChecked():
            companion = le_bail_companion(
                main,
                self.page_data.edit.text(),
                self.page_phase.edit.text(),
                inst_copy,
                smp_copy,
                n_background=self.page_run.n_background.value(),
                position="distance" if self.page_run.disp.isChecked() else "zero",
            )
        super().accept()
        main.run_recipe(recipe, companion=companion)


def le_bail_companion(
    main: MainWindow,
    data_path: str,
    cif_path: str,
    inst: object,
    sample: object,
    *,
    n_background: int,
    position: str,
) -> Callable[[Callable[[str], None]], str]:
    """Rietveld のあとに同じデータを Le Bail で合わせ、Rwp の下限と比べる文を返す処理を作る。

    Le Bail が Rietveld より明らかに良ければ、残りの差はプロファイルや背景では説明できない
    分で、構造モデル(占有率・選択配向・第二相)の問題(opXRD の比較で NbBr4 4.5% 対 7.6%)。
    """

    def run(progress: Callable[[str], None]) -> str:
        from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
        from kaisetsu.pattern.tof import TOFInstrument
        from kaisetsu.refine.lattice import refine_lattice

        assert isinstance(inst, CWInstrument | TOFInstrument)
        assert isinstance(sample, SampleGeometry)
        r = refine_lattice(
            data_path,
            [cif_path],
            inst,
            sample,
            n_background=n_background,
            position=position,
            widths="instrument",
            progress=progress,
            keep_project=False,
        )
        if not r.ok:
            return f"Le Bail 並走: 失敗しました({r.message})"
        lb = 100 * r.rwp
        rv = 100 * main.project.rfactors()["total"].rwp
        diff = rv - lb
        if diff > max(1.0, 0.2 * rv):
            return (
                f"Le Bail(構造なし)の Rwp = {lb:.2f}%。Rietveld の {rv:.2f}% との差 {diff:.1f} ポイントは"
                "プロファイルや背景では説明できない分です。構造モデル(占有率・選択配向・第二相)を"
                "疑ってください"
            )
        return (
            f"Le Bail(構造なし)の Rwp = {lb:.2f}%。Rietveld の {rv:.2f}% との差は小さく、"
            "残る残差はピーク形状や背景の限界です"
        )

    return run


def _phase_name(path: str) -> str:
    try:
        st = read_structure(path)
        name = st.name or Path(path).stem
    except Exception:
        name = Path(path).stem
    return name.replace(":", "_") or "phase"
