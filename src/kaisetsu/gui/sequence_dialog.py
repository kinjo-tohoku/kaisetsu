"""逐次精密化のダイアログ: データファイルを順に差し替えて精密化し、結果を表にする。

メインウィンドウのプロジェクトは書き換えない。独立なコピー(`clone_project`)で走らせ、
利用者が「結果をメインに反映」を押したときだけ、最後のデータの状態をメインに渡す
(以前はメインのヒストグラムが最後のファイルのデータに置き換わり、そのまま Ctrl+S で元の
プロジェクトファイルを上書きできた。レビュー 2026-10-17 GUI #4)。
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.dialog_utils import DATA_FILTER
from kaisetsu.gui.i18n import tr
from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.io.project import clone_project
from kaisetsu.refine.project import Project
from kaisetsu.refine.sequential import SequenceResult, refine_sequence


class _SequenceCancelled(Exception):
    pass


class _SeqWorker(QObject):
    progress = Signal(str)
    finished = Signal(bool, str, object)

    def __init__(self, project: Project, paths: list[str], reset_background: bool) -> None:
        super().__init__()
        self.project = project
        self.paths = paths
        self.reset_background = reset_background
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def _progress(self, message: str) -> None:
        """1 ファイル終わるごとに呼ばれる。中止が要求されていればここで止める。"""
        self.progress.emit(message)
        if self._cancel:
            raise _SequenceCancelled()

    def run(self) -> None:
        try:
            res = refine_sequence(
                self.project, self.paths, reset_background=self.reset_background,
                progress=self._progress,
            )  # fmt: skip
            self.finished.emit(True, f"{len(res.rows)} 件を処理しました", res)
        except _SequenceCancelled:
            self.finished.emit(False, "中止しました(途中までの結果はログにあります)", None)
        except Exception as exc:
            self.finished.emit(False, f"エラー: {exc}\n{traceback.format_exc(limit=3)}", None)


class SequenceDialog(QDialog):
    """ファイルを選び、いま精密化にチェックの入っているパラメータで順に精密化する。

    Args:
        project: メインのプロジェクト(コピーして使う。書き換えない)
        apply: 「結果をメインに反映」で呼ぶ関数(最後のデータの状態のプロジェクトを渡す)。
            None ならそのボタンは出さない
    """

    def __init__(
        self,
        project: Project,
        parent: QWidget | None = None,
        *,
        apply: Callable[[Project], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_project = project
        self.project = clone_project(project)
        self.apply = apply
        self.result: SequenceResult | None = None
        self._thread: QThread | None = None
        self._worker: _SeqWorker | None = None
        self._close_when_done = False
        self.setWindowTitle(tr("連続データを順に精密化"))
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "温度・時間系列など、同じ試料・同じ装置のデータを順に精密化します。\n"
            "現在チェックの入っているパラメータを、前のデータの結果を初期値にして動かします。\n"
            "メインウィンドウのプロジェクトは変わりません(最後のデータの状態を使うには"
            "「結果をメインに反映」)。"
        ))  # fmt: skip
        row = QHBoxLayout()
        self.files = QListWidget()
        row.addWidget(self.files, 1)
        side = QVBoxLayout()
        b_add = QPushButton(tr("ファイルを追加…"))
        b_add.clicked.connect(self.add_files)
        b_clear = QPushButton(tr("クリア"))
        b_clear.clicked.connect(self.files.clear)
        side.addWidget(b_add)
        side.addWidget(b_clear)
        side.addStretch(1)
        row.addLayout(side)
        layout.addLayout(row)
        self.reset_bkg = QCheckBox(tr("データごとに背景を観測から推定し直す"))
        self.reset_bkg.setChecked(True)
        layout.addWidget(self.reset_bkg)
        run_row = QHBoxLayout()
        self.run_btn = QPushButton(tr("実行"))
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QPushButton(tr("中止"))
        self.cancel_btn.setToolTip("いま精密化しているデータが終わったところで止めます")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.cancel_btn)
        layout.addLayout(run_row)
        self.table = QTableWidget()
        layout.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_btn = buttons.addButton(tr("CSV に保存…"), QDialogButtonBox.ButtonRole.ActionRole)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_csv)
        self.apply_btn = buttons.addButton(
            tr("結果をメインに反映"), QDialogButtonBox.ButtonRole.ActionRole
        )
        self.apply_btn.setToolTip(
            "最後のデータとその精密化結果をメインウィンドウのプロジェクトにします"
            "(メインの今のプロジェクトは閉じます。保存していなければ確認します)"
        )
        self.apply_btn.setEnabled(False)
        self.apply_btn.setVisible(apply is not None)
        self.apply_btn.clicked.connect(self.apply_to_main)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if not project.params.free():
            self.log.appendPlainText(
                "精密化するパラメータが選ばれていません(表でチェックしてから開いてください)"
            )

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("回折データ(複数選択)"), last_dir(), DATA_FILTER
        )
        for p in paths:
            self.files.addItem(p)
        if paths:
            remember_dir(paths[0])

    def set_paths(self, paths: list[str]) -> None:
        self.files.clear()
        for p in paths:
            self.files.addItem(p)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def run(self) -> None:
        paths = [self.files.item(i).text() for i in range(self.files.count())]
        if not paths:
            self.log.appendPlainText("ファイルを追加してください")
            return
        if self.is_running():
            return
        self._worker = _SeqWorker(self.project, paths, self.reset_bkg.isChecked())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.finished.connect(self._thread.quit)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self._thread.start()

    def cancel(self) -> None:
        """中止を要求する(いまのデータの精密化が終わったところで止まる)。"""
        if self._worker is None or not self.is_running():
            return
        if not self._worker.cancelled:
            self._worker.cancel()
            self.log.appendPlainText("中止を要求しました(いまのデータの精密化が終わると止まります)")
        self.cancel_btn.setEnabled(False)

    def _done(self, ok: bool, message: str, result: object) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.log.appendPlainText(message)
        if self._close_when_done:
            QTimer.singleShot(0, self._finish_close)
        if not ok or not isinstance(result, SequenceResult):
            return
        self.result = result
        names = result.param_names
        self.table.setColumnCount(4 + len(names))
        self.table.setHorizontalHeaderLabels([tr("データ"), "Rwp %", "χ²", tr("備考"), *names])
        self.table.setRowCount(len(result.rows))
        for r, row in enumerate(result.rows):
            self.table.setItem(r, 0, QTableWidgetItem(row.name + ("" if row.ok else " (失敗)")))
            self.table.setItem(r, 1, QTableWidgetItem(f"{100 * row.rwp:.3f}" if row.ok else ""))
            self.table.setItem(r, 2, QTableWidgetItem(f"{row.chi2:.3f}" if row.ok else ""))
            # 備考: 未収束(評価回数の上限)や格子定数の振り直し、失敗ならその理由
            self.table.setItem(r, 3, QTableWidgetItem(row.message))
            for c, n in enumerate(names):
                v = row.values.get(n)
                e = row.esds.get(n)
                txt = "" if v is None else (f"{v:.6g}" + (f" ± {e:.2g}" if e else ""))
                self.table.setItem(r, 4 + c, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self.save_btn.setEnabled(True)
        self.apply_btn.setEnabled(self.apply is not None)

    def apply_to_main(self) -> None:
        """最後のデータの状態(精密化済みのコピー)をメインウィンドウに渡す。"""
        if self.apply is None or self.result is None or self.is_running():
            return
        self.apply(self.project)
        self.log.appendPlainText("メインウィンドウに反映しました(最後のデータの状態)")

    def save_csv(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("CSV に保存"), last_dir(), "CSV (*.csv)")
        if not path:
            return
        self.result.to_csv(path)
        remember_dir(path)
        self.log.appendPlainText(f"保存しました: {Path(path).name}")

    def reject(self) -> None:
        """閉じる・Esc・×。実行中なら中止を要求し、ワーカーが終わってから閉じる。"""
        if self.is_running():
            if self._worker is not None and not self._worker.cancelled:
                ans = QMessageBox.question(
                    self, self.windowTitle(), "逐次精密化の実行中です。中止して閉じますか?"
                )
                if ans != QMessageBox.StandardButton.Yes:
                    return
                self.cancel()
            self._close_when_done = True
            self.log.appendPlainText("中止を待っています…")
            return
        super().reject()

    def _finish_close(self) -> None:
        if self._thread is not None:
            self._thread.wait(10000)
        super().reject()


__all__ = ["SequenceDialog"]
