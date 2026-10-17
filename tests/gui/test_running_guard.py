"""精密化の実行中は、モデルを変える操作を全部止める(レビュー 2026-10-17 GUI #3, #13)。"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem

from kaisetsu.gui import main_window as mw
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.param_model import COL_VALUE, COL_VARY
from kaisetsu.gui.worker import RefineWorker


def _row(win: MainWindow, name: str) -> int:
    for r in range(win.param_model.rowCount()):
        if win.param_model.parameter(r).name == name:
            return r
    raise AssertionError(name)


def test_set_running_blocks_model_changes(qtbot, tiny_window: MainWindow, monkeypatch) -> None:
    win = tiny_window
    h = win.project.histograms[0]
    win._set_running(True)
    assert win.is_running()
    blocked = (
        win.act_new, win.act_open, win.act_save, win.act_save_as, win.act_load,
        win.act_add_phase, win.act_instrument, win.act_background, win.act_range, win.act_phase,
        win.act_lattice, win.act_wizard, win.act_sequence, win.act_mem, win.act_residual,
        win.act_calc, win.act_refine, win.act_recipe, win.act_pdf_recipe, win.act_undo,
        win.act_reset, win.act_fix_all, win.act_width_penalty, win.act_export_pattern,
        win.btn_vary_shown, win.btn_fix_shown, win.btn_reset_shown,
    )  # fmt: skip
    assert all(not a.isEnabled() for a in blocked)
    assert win.act_stop.isEnabled()
    # 表は読み取り専用(値も「精密化」のチェックも)
    vidx = win.param_model.index(_row(win, "h0:bkg:0"), COL_VALUE)
    cidx = win.param_model.index(_row(win, "h0:bkg:0"), COL_VARY)
    assert not (win.param_model.flags(vidx) & Qt.ItemFlag.ItemIsEditable)
    assert not (win.param_model.flags(cidx) & Qt.ItemFlag.ItemIsUserCheckable)
    before = win.project.params["h0:bkg:0"].value
    assert not win.param_model.setData(vidx, "99", Qt.ItemDataRole.EditRole)
    assert not win.param_model.setData(
        cidx, Qt.CheckState.Checked.value, Qt.ItemDataRole.CheckStateRole
    )
    assert win.project.params["h0:bkg:0"].value == before
    assert not win.project.params["h0:bkg:0"].vary
    # 除外領域の帯(Shift + ドラッグ、帯の移動)は効かない
    assert not win.plot.exclude_editable()
    win._on_exclude_requested(45.0, 46.0)
    win._on_excluded_changed([(45.0, 46.0)])
    assert h.excluded == []
    # 表示中にチェック / 固定 / 初期値に戻す
    win._set_vary_shown(True)
    assert not win.project.params.free()
    # ツリーのクリックは再計算せず、ワーカーが送った曲線だけ表示する
    monkeypatch.setattr(
        win.project, "calculate", lambda *a, **k: pytest.fail("実行中に calculate が呼ばれた")
    )
    win._last_curves = {"hist": [(h.data.x, h.data.y * 0.5, h.data.y * 0.1)], "pdf": []}
    item = QTreeWidgetItem(["h0"])
    item.setData(0, Qt.ItemDataRole.UserRole, ("hist", 0))
    win._on_tree_clicked(item, 0)
    win.recalculate()
    assert win.plot._calc.yData is not None and len(win.plot._calc.yData) == len(h.data)
    # 逐次精密化・MEM・ウィザードはガードで止まる
    msgs: list[str] = []
    monkeypatch.setattr(
        mw.QMessageBox, "information", staticmethod(lambda *a, **k: msgs.append(a[2]))
    )
    win.run_sequence()
    win.run_mem()
    assert len(msgs) == 2 and all("実行中" in m for m in msgs)
    assert not win.new_project(confirm=False) and win.project.histograms
    # 終わると戻る
    win._set_running(False)
    assert not win.is_running() and win.act_save.isEnabled() and win.act_refine.isEnabled()
    assert not win.act_stop.isEnabled() and win.plot.exclude_editable()
    assert win.param_model.flags(vidx) & Qt.ItemFlag.ItemIsEditable
    assert win.param_model.setData(vidx, "99", Qt.ItemDataRole.EditRole)


class _SlowWorker(RefineWorker):
    """中止されるまで待つだけのワーカー(閉じる操作の確認用)。"""

    def run(self) -> None:
        while not self._cancel:
            time.sleep(0.02)
        self.finished.emit(False, "中止しました(途中までの値が入っています)")


def test_close_requests_cancel_and_waits(qtbot, tiny_window: MainWindow, monkeypatch) -> None:
    win = tiny_window
    win.show()
    monkeypatch.setattr(win, "_ask", lambda box: QMessageBox.StandardButton.Discard)
    worker = _SlowWorker(win.project, None)
    win._start_worker(worker)
    assert win.is_running() and not win.act_save.isEnabled()
    assert win.close() is False  # すぐには閉じず、中止を要求して待つ
    assert win.isVisible() and worker._cancel
    assert "中止を待っています" in win.progress_label.text()
    qtbot.waitUntil(lambda: not win.isVisible(), timeout=10000)
    assert not win.is_running()
    assert win._thread is not None and not win._thread.isRunning()


def test_refine_worker_toggles_running(qtbot, tiny_window: MainWindow) -> None:
    win = tiny_window
    win.project.vary("h0:bkg:*", "h0:Ni:scale")
    worker = RefineWorker(win.project, None, max_cycles=3)
    with qtbot.waitSignal(worker.finished, timeout=60_000):
        win._start_worker(worker)
        assert win.is_running() and not win.act_open.isEnabled()
    qtbot.waitUntil(lambda: not win.is_running(), timeout=5000)
    assert win.act_open.isEnabled() and win.act_undo.isEnabled() and win.is_dirty()
