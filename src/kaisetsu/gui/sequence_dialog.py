"""逐次精密化のダイアログ: データファイルを順に差し替えて精密化し、結果を表にする。"""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.refine.project import Project
from kaisetsu.refine.sequential import SequenceResult, refine_sequence


class _SeqWorker(QObject):
    progress = Signal(str)
    finished = Signal(bool, str, object)

    def __init__(self, project: Project, paths: list[str], reset_background: bool) -> None:
        super().__init__()
        self.project = project
        self.paths = paths
        self.reset_background = reset_background

    def run(self) -> None:
        try:
            res = refine_sequence(
                self.project, self.paths, reset_background=self.reset_background,
                progress=self.progress.emit,
            )  # fmt: skip
            self.finished.emit(True, f"{len(res.rows)} 件を処理しました", res)
        except Exception as exc:
            self.finished.emit(False, f"エラー: {exc}\n{traceback.format_exc(limit=3)}", None)


class SequenceDialog(QDialog):
    """ファイルを選び、いま精密化にチェックの入っているパラメータで順に精密化する。"""

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.result: SequenceResult | None = None
        self._thread: QThread | None = None
        self._worker: _SeqWorker | None = None
        self.setWindowTitle("連続データを順に精密化")
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "温度・時間系列など、同じ試料・同じ装置のデータを順に精密化します。\n"
            "現在チェックの入っているパラメータを、前のデータの結果を初期値にして動かします。"
        ))  # fmt: skip
        row = QHBoxLayout()
        self.files = QListWidget()
        row.addWidget(self.files, 1)
        side = QVBoxLayout()
        b_add = QPushButton("ファイルを追加…")
        b_add.clicked.connect(self.add_files)
        b_clear = QPushButton("クリア")
        b_clear.clicked.connect(self.files.clear)
        side.addWidget(b_add)
        side.addWidget(b_clear)
        side.addStretch(1)
        row.addLayout(side)
        layout.addLayout(row)
        self.reset_bkg = QCheckBox("データごとに背景を観測から推定し直す")
        self.reset_bkg.setChecked(True)
        layout.addWidget(self.reset_bkg)
        self.run_btn = QPushButton("実行")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)
        self.table = QTableWidget()
        layout.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_btn = buttons.addButton("CSV に保存…", QDialogButtonBox.ButtonRole.ActionRole)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_csv)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if not project.params.free():
            self.log.appendPlainText(
                "精密化するパラメータが選ばれていません(表でチェックしてから開いてください)"
            )

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "回折データ(複数選択)", last_dir(),
            "回折データ (*.xy *.xye *.chi *.dat *.txt *.csv *.ras *.rasx *.xrdml *.gsa *.fxye *.raw "
            "*.cif);;すべて (*)",
        )  # fmt: skip
        for p in paths:
            self.files.addItem(p)
        if paths:
            remember_dir(paths[0])

    def set_paths(self, paths: list[str]) -> None:
        self.files.clear()
        for p in paths:
            self.files.addItem(p)

    def run(self) -> None:
        paths = [self.files.item(i).text() for i in range(self.files.count())]
        if not paths:
            self.log.appendPlainText("ファイルを追加してください")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self._worker = _SeqWorker(self.project, paths, self.reset_bkg.isChecked())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.finished.connect(self._thread.quit)
        self.run_btn.setEnabled(False)
        self._thread.start()

    def _done(self, ok: bool, message: str, result: object) -> None:
        self.run_btn.setEnabled(True)
        self.log.appendPlainText(message)
        if not ok or not isinstance(result, SequenceResult):
            return
        self.result = result
        names = result.param_names
        self.table.setColumnCount(3 + len(names))
        self.table.setHorizontalHeaderLabels(["データ", "Rwp %", "χ²", *names])
        self.table.setRowCount(len(result.rows))
        for r, row in enumerate(result.rows):
            self.table.setItem(r, 0, QTableWidgetItem(row.name + ("" if row.ok else " (失敗)")))
            self.table.setItem(r, 1, QTableWidgetItem(f"{100 * row.rwp:.3f}" if row.ok else ""))
            self.table.setItem(r, 2, QTableWidgetItem(f"{row.chi2:.3f}" if row.ok else ""))
            for c, n in enumerate(names):
                v = row.values.get(n)
                e = row.esds.get(n)
                txt = "" if v is None else (f"{v:.6g}" + (f" ± {e:.2g}" if e else ""))
                self.table.setItem(r, 3 + c, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self.save_btn.setEnabled(True)

    def save_csv(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV に保存", last_dir(), "CSV (*.csv)")
        if not path:
            return
        self.result.to_csv(path)
        remember_dir(path)
        self.log.appendPlainText(f"保存しました: {Path(path).name}")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(10000)
        event.accept()


__all__ = ["SequenceDialog"]
