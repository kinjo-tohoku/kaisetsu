"""Le Bail で自動解析するダイアログ(格子定数とプロファイルだけ。一括処理、放射光マイクロ XRD にも)。

データファイル(複数可)と CIF(複数相可)を入れて「実行」すると、校正した装置テンプレート
(またはメインウィンドウの装置、プリセット)で Le Bail 精密化を走らせ、相ごとの格子定数と esd、
体積、Rwp を表にする。CSV に保存でき、最後の結果はメインウィンドウで図を見られる。
"""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.i18n import tr
from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.io.project import INSTRUMENT_TEMPLATE_FILTER, load_instrument_template
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.pattern.presets import PRESETS, instrument_from_preset
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.refine.lattice import (
    CELL_NAMES,
    LatticeBatch,
    PositionMode,
    WidthMode,
    format_with_esd,
    refine_lattice_batch,
)
from kaisetsu.refine.project import Project

DATA_FILTER = (
    "回折データ (*.xy *.xye *.chi *.dat *.txt *.csv *.asc *.ras *.rasx *.xrdml "
    "*.gsa *.gsas *.fxye *.raw *.xra *.cwn *.cif);;すべて (*)"
)
CIF_FILTER = "CIF (*.cif);;すべて (*)"

POSITION_CHOICES: list[tuple[str, PositionMode]] = [
    ("動かさない(校正済みのテンプレートを使う)", "none"),
    ("ゼロ点(2θ の定数ずれ)", "zero"),
    ("試料–検出器距離(ビーム方向の試料位置のずれ)", "distance"),
    ("ゼロ点と距離の両方", "both"),
    ("自動(ゼロ点 / 距離 / 両方を試して χ² で選ぶ)", "auto"),
]
WIDTH_CHOICES: list[tuple[str, WidthMode]] = [
    ("試料の広がりだけ(ローレンツ成分 X, Y。装置の幅は固定)", "sample"),
    ("装置の幅も動かす(U, V, W, X, Y。校正していない装置)", "instrument"),
    ("動かさない", "none"),
]


class _LatticeWorker(QObject):
    progress = Signal(str)
    finished = Signal(bool, str, object)

    def __init__(
        self,
        paths: list[str],
        cifs: list[str],
        instrument: CWInstrument | TOFInstrument,
        sample: SampleGeometry,
        options: dict[str, object],
    ) -> None:
        super().__init__()
        self.paths = paths
        self.cifs = cifs
        self.instrument = instrument
        self.sample = sample
        self.options = options

    def run(self) -> None:
        try:
            batch = refine_lattice_batch(
                self.paths,
                self.cifs,
                self.instrument,
                self.sample,
                progress=self.progress.emit,
                keep_projects=len(self.paths) <= 20,
                **self.options,
            )
            n_ok = sum(1 for r in batch.results if r.ok)
            self.finished.emit(True, f"{n_ok} / {len(batch.results)} 件が終わりました", batch)
        except Exception as exc:
            self.finished.emit(False, f"エラー: {exc}\n{traceback.format_exc(limit=3)}", None)


class LatticeDialog(QDialog):
    """データと CIF から Le Bail で格子定数を出す。"""

    def __init__(
        self, project: Project, parent: QWidget | None = None, main: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.main = main
        self.batch: LatticeBatch | None = None
        self._thread: QThread | None = None
        self._worker: _LatticeWorker | None = None
        self.setWindowTitle(tr("Le Bail で自動解析"))
        self.resize(1000, 720)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "構造を使わない Le Bail 法で、データと CIF から格子定数(と esd)、ピーク幅、背景を自動で決めます。"
            "CIF からは空間群と格子定数の初期値だけを使うので、原子座標が無くても、構造が合っているか"
            "分からなくても使えます。相同定のあとの格子定数チェック、Rietveld の前の下準備、"
            "μm ビームで結晶粒が少なく強度が当てにならない放射光マイクロ XRD(マッピング)に。\n"
            "放射光の 2 次元検出器では試料–検出器距離と波長の誤差が格子定数とほぼ完全に相関するので、"
            "標準試料で校正した装置テンプレートを使い、位置の補正は動かさないのが基本です。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        lists = QHBoxLayout()
        lists.addWidget(self._list_box("回折データ(複数可)", "data"), 1)
        lists.addWidget(self._list_box("相の CIF(複数相なら複数)", "cif"), 1)
        layout.addLayout(lists)

        g_inst = QGroupBox("装置と試料")
        v_inst = QVBoxLayout(g_inst)
        self.rb_main = QRadioButton("メインウィンドウの装置と試料(1 つ目のヒストグラム)")
        self.rb_template = QRadioButton("装置テンプレートファイル:")
        self.rb_preset = QRadioButton("プリセット:")
        self.template_path = QLineEdit()
        self.template_path.setPlaceholderText("標準試料で校正して保存した *.kaisetsu-inst.json")
        b_tpl = QPushButton("参照…")
        b_tpl.clicked.connect(self._browse_template)
        self.preset = QComboBox()
        for name in PRESETS:
            self.preset.addItem(name)
        idx = self.preset.findText("放射光 X 線 (キャピラリ, Debye-Scherrer)")
        if idx >= 0:
            self.preset.setCurrentIndex(idx)
        v_inst.addWidget(self.rb_main)
        row_t = QHBoxLayout()
        row_t.addWidget(self.rb_template)
        row_t.addWidget(self.template_path, 1)
        row_t.addWidget(b_tpl)
        v_inst.addLayout(row_t)
        row_p = QHBoxLayout()
        row_p.addWidget(self.rb_preset)
        row_p.addWidget(self.preset, 1)
        v_inst.addLayout(row_p)
        has_hist = bool(project.histograms)
        self.rb_main.setEnabled(has_hist)
        (self.rb_main if has_hist else self.rb_preset).setChecked(True)
        layout.addWidget(g_inst)

        g_opt = QGroupBox("精密化の条件")
        v_opt = QVBoxLayout(g_opt)
        row1 = QHBoxLayout()
        self.use_range = QCheckBox("範囲を限る 2θ")
        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(0.0, 180.0)
        self.x_min.setDecimals(3)
        self.x_min.setValue(2.0)
        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(0.0, 180.0)
        self.x_max.setDecimals(3)
        self.x_max.setValue(40.0)
        row1.addWidget(self.use_range)
        row1.addWidget(self.x_min)
        row1.addWidget(QLabel("〜"))
        row1.addWidget(self.x_max)
        row1.addSpacing(16)
        row1.addWidget(QLabel("背景の項数"))
        self.n_bkg = QSpinBox()
        self.n_bkg.setRange(1, 24)
        self.n_bkg.setValue(6)
        row1.addWidget(self.n_bkg)
        row1.addStretch(1)
        v_opt.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("位置の補正"))
        self.position = QComboBox()
        for label, _ in POSITION_CHOICES:
            self.position.addItem(label)
        self.position.setCurrentIndex(0)
        row2.addWidget(self.position, 1)
        v_opt.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("ピーク幅"))
        self.widths = QComboBox()
        for label, _ in WIDTH_CHOICES:
            self.widths.addItem(label)
        row3.addWidget(self.widths, 1)
        v_opt.addLayout(row3)
        layout.addWidget(g_opt)

        self.run_btn = QPushButton("実行")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)
        self.table = QTableWidget()
        layout.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_btn = buttons.addButton("CSV に保存…", QDialogButtonBox.ButtonRole.ActionRole)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_csv)
        self.open_btn = buttons.addButton(
            "最後の結果をメインウィンドウで開く", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.open_btn.setToolTip("図と差のプロットを見たり、続きを手で精密化したりできます")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_in_main)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _list_box(self, title: str, kind: str) -> QGroupBox:
        box = QGroupBox(title)
        h = QHBoxLayout(box)
        lst = QListWidget()
        h.addWidget(lst, 1)
        side = QVBoxLayout()
        b_add = QPushButton("追加…")
        b_clear = QPushButton("クリア")
        side.addWidget(b_add)
        side.addWidget(b_clear)
        side.addStretch(1)
        h.addLayout(side)
        if kind == "data":
            self.files = lst
            b_add.clicked.connect(self.add_files)
        else:
            self.cifs = lst
            b_add.clicked.connect(self.add_cifs)
        b_clear.clicked.connect(lst.clear)
        return box

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "回折データ(複数選択)", last_dir(), DATA_FILTER
        )
        for p in paths:
            self.files.addItem(p)
        if paths:
            remember_dir(paths[0])

    def add_cifs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "相の CIF(複数選択)", last_dir(), CIF_FILTER)
        for p in paths:
            self.cifs.addItem(p)
        if paths:
            remember_dir(paths[0])

    def set_paths(self, data: list[str], cifs: list[str]) -> None:
        self.files.clear()
        self.cifs.clear()
        for p in data:
            self.files.addItem(p)
        for p in cifs:
            self.cifs.addItem(p)

    def _browse_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "装置テンプレート", last_dir(), INSTRUMENT_TEMPLATE_FILTER
        )
        if path:
            self.template_path.setText(path)
            self.rb_template.setChecked(True)
            remember_dir(path)

    def instrument_and_sample(self) -> tuple[CWInstrument | TOFInstrument, SampleGeometry]:
        """選ばれた装置と試料形状。"""
        if self.rb_main.isChecked() and self.project.histograms:
            h = self.project.histograms[0]
            return h.instrument, h.sample
        if self.rb_template.isChecked():
            path = self.template_path.text().strip()
            if not path:
                raise ValueError("装置テンプレートのファイルを選んでください")
            return load_instrument_template(path)
        return instrument_from_preset(self.preset.currentText())

    def options(self) -> dict[str, object]:
        opts: dict[str, object] = {
            "n_background": self.n_bkg.value(),
            "position": POSITION_CHOICES[self.position.currentIndex()][1],
            "widths": WIDTH_CHOICES[self.widths.currentIndex()][1],
        }
        if self.use_range.isChecked():
            opts["x_min"] = self.x_min.value()
            opts["x_max"] = self.x_max.value()
        return opts

    # ------------------------------------------------------------------
    def run(self) -> None:
        paths = [self.files.item(i).text() for i in range(self.files.count())]
        cifs = [self.cifs.item(i).text() for i in range(self.cifs.count())]
        if not paths or not cifs:
            self.log.appendPlainText("回折データと CIF を追加してください")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        try:
            inst, smp = self.instrument_and_sample()
        except Exception as exc:
            QMessageBox.warning(self, "装置", f"装置を決められませんでした:\n{exc}")
            return
        if isinstance(inst, CWInstrument) and smp.kind is GeometryKind.BRAGG_BRENTANO:
            self.log.appendPlainText(
                "注意: 試料形状が Bragg-Brentano(平板)です。透過(2 次元検出器・キャピラリ)なら"
                "装置設定で Debye-Scherrer にしてください"
            )
        self._worker = _LatticeWorker(paths, cifs, inst, smp, self.options())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.finished.connect(self._thread.quit)
        self.run_btn.setEnabled(False)
        self.log.appendPlainText(f"{len(paths)} 件 × {len(cifs)} 相を Le Bail で精密化します…")
        self._thread.start()

    def _done(self, ok: bool, message: str, batch: object) -> None:
        self.run_btn.setEnabled(True)
        self.log.appendPlainText(message)
        if not ok or not isinstance(batch, LatticeBatch):
            return
        self.batch = batch
        self.fill_table(batch)
        self.save_btn.setEnabled(True)
        self.open_btn.setEnabled(any(r.ok and r.project is not None for r in batch.results))

    def fill_table(self, batch: LatticeBatch) -> None:
        headers = ["データ", "相", "a [Å]", "b [Å]", "c [Å]", "α [°]", "β [°]", "γ [°]", "V [Å³]",
                   "Rwp %", "χ²", "反射", "注意"]  # fmt: skip
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        rows: list[list[str]] = []
        for r in batch.results:
            if not r.ok:
                rows.append([r.name, "(失敗)", *[""] * 9, "", r.message])
                continue
            for c in r.cells:
                rows.append(
                    [
                        r.name,
                        c.phase,
                        *[format_with_esd(c.values[nm], c.esds.get(nm)) for nm in CELL_NAMES],
                        format_with_esd(c.volume, c.volume_esd, 3),
                        f"{100 * r.rwp:.3f}",
                        f"{r.chi2:.3f}",
                        str(c.n_reflections),
                        " / ".join(r.warnings),
                    ]
                )
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def save_csv(self) -> None:
        if self.batch is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV に保存", last_dir(), "CSV (*.csv)")
        if not path:
            return
        self.batch.to_csv(path)
        remember_dir(path)
        self.log.appendPlainText(f"保存しました: {Path(path).name}")

    def last_project(self) -> Project | None:
        if self.batch is None:
            return None
        for r in reversed(self.batch.results):
            if r.ok and r.project is not None:
                return r.project
        return None

    def open_in_main(self) -> None:
        proj = self.last_project()
        adopt = getattr(self.main, "adopt_project", None)
        if proj is None or adopt is None:
            QMessageBox.information(self, "結果", "メインウィンドウで開ける結果がありません")
            return
        adopt(proj, source=f"Le Bail 格子定数: {proj.name}")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(10000)
        event.accept()


__all__ = ["LatticeDialog"]
