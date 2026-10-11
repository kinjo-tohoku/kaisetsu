"""はじめての精密化ウィザード: データ → 相 → 装置 → 実行。"""

from __future__ import annotations

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
            "回折データ (*.xy *.xye *.dat *.txt *.gsa *.gsas *.fxye *.raw *.xra *.cwn);;すべて (*)",
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
        self.coords = QCheckBox("原子座標も動かす")
        self.coords.setChecked(True)
        self.adp = QCheckBox("変位パラメータ(U)も動かす")
        self.adp.setChecked(True)
        self.disp = QCheckBox("ゼロ点ではなく試料高さずれを動かす(Bragg-Brentano で推奨)")
        layout.addWidget(self.coords)
        layout.addWidget(self.adp)
        layout.addWidget(self.disp)
        row = QHBoxLayout()
        row.addWidget(QLabel("背景(Chebyshev)の項数:"))
        self.n_background = QSpinBox()
        self.n_background.setRange(1, 36)
        self.n_background.setValue(8)
        self.n_background.setToolTip("目安 6〜12。低角の立ち上がりが合わなければ増やす")
        row.addWidget(self.n_background)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(QLabel(
            "手順: スケールと背景 → 格子定数とピーク位置 → ピーク幅 → 原子座標 → 変位パラメータ\n"
            "Rwp が大きく変わらなくなれば収束です。結果はパラメータ表と「結果の要約」で確認できます。"
        ))  # fmt: skip
        layout.addStretch(1)


class StartWizard(QWizard):
    def __init__(self, main: MainWindow) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("はじめての精密化")
        self.resize(760, 640)
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
        )
        super().accept()
        main.run_recipe(recipe)


def _phase_name(path: str) -> str:
    try:
        st = read_structure(path)
        name = st.name or Path(path).stem
    except Exception:
        name = Path(path).stem
    return name.replace(":", "_") or "phase"
