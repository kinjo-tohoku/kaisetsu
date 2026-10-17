"""逐次精密化・MEM のダイアログ: メインのプロジェクトを書き換えない、閉じるときはワーカーを待つ。

(レビュー 2026-10-17 GUI #4、差分 #2 の備考列)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from kaisetsu.gui import mem_dialog as mem_mod
from kaisetsu.gui import sequence_dialog as seq_mod
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.mem_dialog import MEMDialog
from kaisetsu.gui.sequence_dialog import SequenceDialog
from kaisetsu.refine.project import Project
from kaisetsu.refine.sequential import SequenceResult
from tests.gui.conftest import write_tiny_pattern


def test_sequence_leaves_main_project_untouched(
    qtbot, tiny_window: MainWindow, tmp_path: Path, monkeypatch
) -> None:
    win = tiny_window
    win.project.vary("h0:bkg:*", "h0:Ni:scale")
    h = win.project.histograms[0]
    data0 = h.data
    y0 = h.data.y.copy()
    values0 = win.project.snapshot()
    paths = [
        str(write_tiny_pattern(tmp_path / f"s{i}.xy", scale=1.0 + 0.2 * i, seed=i))
        for i in range(2)
    ]
    applied: list[Project] = []
    dlg = SequenceDialog(win.project, apply=applied.append)
    qtbot.addWidget(dlg)
    assert dlg.project is not win.project and dlg.source_project is win.project
    dlg.set_paths(paths)
    dlg.run()
    qtbot.waitUntil(
        lambda: dlg._thread is not None and not dlg._thread.isRunning(), timeout=120_000
    )
    qtbot.waitUntil(lambda: dlg.run_btn.isEnabled(), timeout=5000)
    assert (
        dlg.result is not None and len(dlg.result.rows) == 2 and all(r.ok for r in dlg.result.rows)
    )
    # メインのヒストグラムは同じオブジェクトのまま、データも値も変わっていない
    assert h.data is data0 and np.array_equal(h.data.y, y0)
    assert h.data.metadata.get("path", "").endswith("tiny.xy")
    assert win.project.snapshot() == values0
    # コピー側は最後のファイルの状態
    assert dlg.project.histograms[0].data.metadata.get("path", "").endswith("s1.xy")
    # 表に備考の列がある
    headers = [dlg.table.horizontalHeaderItem(c).text() for c in range(dlg.table.columnCount())]
    assert headers[:4] == ["データ", "Rwp %", "χ²", "備考"]
    assert dlg.table.item(0, 3) is not None
    # 「結果をメインに反映」は明示的に押したときだけ
    assert dlg.apply_btn.isEnabled() and not dlg.apply_btn.isHidden()
    dlg.apply_to_main()
    assert applied == [dlg.project]


def test_apply_to_main_replaces_project_as_unsaved(
    qtbot, tiny_window: MainWindow, tmp_path: Path, monkeypatch
) -> None:
    win = tiny_window
    win.project_path = tmp_path / "main.kaisetsu.json"
    assert win.save_project() and not win.is_dirty()
    other = seq_mod.clone_project(win.project)
    other.histograms[0].data.y[:] = 1.0
    win._adopt_sequence_result(other)
    assert win.project is other and win.project_path is None and win.is_dirty()
    assert win._undo_values is None and not win.act_undo.isEnabled()


def test_sequence_reject_waits_for_worker(qtbot, tiny_window: MainWindow, monkeypatch) -> None:
    win = tiny_window

    def slow_sequence(project: Project, paths, *, progress=None, **_kw) -> SequenceResult:  # type: ignore[no-untyped-def]
        for p in paths:
            time.sleep(0.03)
            if progress is not None:
                progress(f"{Path(p).stem}: ok")  # 中止が要求されていればここで止まる
        return SequenceResult()

    monkeypatch.setattr(seq_mod, "refine_sequence", slow_sequence)
    monkeypatch.setattr(
        seq_mod.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    dlg = SequenceDialog(win.project)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.apply_btn.isHidden()  # apply を渡していない
    dlg.set_paths([f"f{i}.xy" for i in range(100)])
    dlg.run()
    assert dlg.is_running() and dlg.cancel_btn.isEnabled()
    dlg.reject()  # 実行中: 「中止して閉じますか?」→ はい
    assert dlg.isVisible() and dlg._worker is not None and dlg._worker.cancelled
    assert "中止を待っています" in dlg.log.toPlainText()
    qtbot.waitUntil(lambda: not dlg.isVisible(), timeout=10000)
    assert dlg._thread is not None and not dlg._thread.isRunning()
    assert "中止しました" in dlg.log.toPlainText()
    # 走っていなければすぐ閉じる
    dlg2 = SequenceDialog(win.project)
    qtbot.addWidget(dlg2)
    dlg2.show()
    dlg2.reject()
    assert not dlg2.isVisible()


def test_sequence_reject_can_be_declined(qtbot, tiny_window: MainWindow, monkeypatch) -> None:
    win = tiny_window

    def slow_sequence(project: Project, paths, *, progress=None, **_kw) -> SequenceResult:  # type: ignore[no-untyped-def]
        for p in paths:
            time.sleep(0.02)
            if progress is not None:
                progress(str(p))
        return SequenceResult()

    monkeypatch.setattr(seq_mod, "refine_sequence", slow_sequence)
    monkeypatch.setattr(
        seq_mod.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
    )
    dlg = SequenceDialog(win.project)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.set_paths([f"f{i}.xy" for i in range(20)])
    dlg.run()
    dlg.reject()  # 「いいえ」: 走り続ける
    assert dlg.isVisible() and dlg._worker is not None and not dlg._worker.cancelled
    qtbot.waitUntil(lambda: not dlg.is_running(), timeout=10000)
    assert dlg.isVisible()
    dlg.reject()
    assert not dlg.isVisible()


def test_mem_dialog_reject_waits_for_worker(qtbot, tiny_window: MainWindow, monkeypatch) -> None:
    win = tiny_window

    def slow_mem(project, hist, phase, grid, *, progress=None, **_kw):  # type: ignore[no-untyped-def]
        for i in range(200):
            time.sleep(0.02)
            if progress is not None:
                progress(i, 1.0, 2.0)
        raise AssertionError("中止されなかった")

    monkeypatch.setattr("kaisetsu.mem.mpf.mem_from_project", slow_mem)  # ワーカーが遅延 import する
    monkeypatch.setattr(
        mem_mod.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    dlg = MEMDialog(win.project)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.run()
    assert dlg.is_running() and dlg.cancel_btn.isEnabled()
    dlg.reject()
    assert dlg.isVisible() and dlg._worker is not None and dlg._worker.cancelled
    qtbot.waitUntil(lambda: not dlg.isVisible(), timeout=10000)
    assert dlg._thread is not None and not dlg._thread.isRunning()
    assert "中止しました" in dlg.log.toPlainText()
