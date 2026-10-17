"""未保存の変更の管理(閉じるときの確認、タイトルの *、保存の失敗、Ctrl+Z の記録の破棄)。

レビュー 2026-10-17 GUI #1, #2, #7、差分 #2(ステータス行の注意)、差分 #3(古い版の注意)。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from kaisetsu.api import AtomSite, Lattice, Project, SpaceGroup, Structure
from kaisetsu.gui import main_window as mw
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.param_model import COL_VALUE, COL_VARY
from kaisetsu.io.project import project_to_dict


def _row(win: MainWindow, name: str) -> int:
    for r in range(win.param_model.rowCount()):
        if win.param_model.parameter(r).name == name:
            return r
    raise AssertionError(name)


def test_loading_marks_dirty_and_close_asks(qtbot, tiny_window: MainWindow, monkeypatch) -> None:
    win = tiny_window
    assert win.is_dirty() and win.isWindowModified()
    assert win.windowTitle().endswith("[*]")
    asked: list[str] = []

    def cancel(box: QMessageBox) -> QMessageBox.StandardButton:
        asked.append(box.text())
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(win, "_ask", cancel)
    win.show()
    assert win.close() is False
    assert win.isVisible() and asked and "保存していない変更" in asked[0]
    # 「保存しない」なら閉じる
    monkeypatch.setattr(win, "_ask", lambda box: QMessageBox.StandardButton.Discard)
    assert win.close() is True
    assert not win.isVisible()


def test_close_with_save_writes_file(
    qtbot, tiny_window: MainWindow, tmp_path: Path, monkeypatch
) -> None:
    win = tiny_window
    path = tmp_path / "p.kaisetsu.json"
    win.project_path = path
    monkeypatch.setattr(win, "_ask", lambda box: QMessageBox.StandardButton.Save)
    win.show()
    assert win.close() is True
    assert path.exists() and not win.is_dirty()


def test_save_failure_is_visible_and_keeps_dirty(
    qtbot, tiny_window: MainWindow, tmp_path: Path, monkeypatch
) -> None:
    win = tiny_window
    win.project_path = tmp_path / "missing" / "x.kaisetsu.json"  # 親フォルダが無いので書けない
    crit: list[str] = []
    monkeypatch.setattr(mw.QMessageBox, "critical", staticmethod(lambda *a, **k: crit.append(a[2])))
    assert win.save_project() is False
    assert crit and "保存できませんでした" in crit[0] and win.is_dirty()
    assert "保存しました" not in win.log_view.toPlainText()
    # 閉じるときに「保存する」を選んでも、保存に失敗すれば閉じない
    monkeypatch.setattr(win, "_ask", lambda box: QMessageBox.StandardButton.Save)
    win.show()
    assert win.close() is False and win.isVisible()
    # 書ける場所なら保存できて、未保存の印が消える
    win.project_path = tmp_path / "ok.kaisetsu.json"
    assert win.save_project() and not win.is_dirty() and not win.isWindowModified()
    # 保存の直後は聞かれない
    monkeypatch.setattr(win, "_ask", lambda box: pytest.fail("保存直後に確認が出た"))
    assert win._confirm_discard()


def test_edits_mark_dirty(qtbot, tiny_window: MainWindow, tmp_path: Path) -> None:
    win = tiny_window
    win.project_path = tmp_path / "a.kaisetsu.json"
    assert win.save_project() and not win.is_dirty()
    # 表の「精密化」チェック
    idx = win.param_model.index(_row(win, "h0:bkg:0"), COL_VARY)
    assert win.param_model.setData(idx, Qt.CheckState.Checked.value, Qt.ItemDataRole.CheckStateRole)
    assert win.is_dirty()
    assert win.save_project() and not win.is_dirty()
    # 表の値
    vidx = win.param_model.index(_row(win, "h0:bkg:0"), COL_VALUE)
    assert win.param_model.setData(vidx, "12.5", Qt.ItemDataRole.EditRole)
    assert win.is_dirty()
    assert win.save_project() and not win.is_dirty()
    # 除外領域(Shift + ドラッグ)
    win._on_exclude_requested(45.0, 46.0)
    assert win.project.histograms[0].excluded == [(45.0, 46.0)] and win.is_dirty()
    assert win.save_project() and not win.is_dirty()
    # 精密化の完了
    win._on_refine_finished(True, "完了: Rwp = 1.000%  χ² = 2.000")
    assert win.is_dirty()


def test_status_line_shows_second_line(qtbot, tiny_window: MainWindow) -> None:
    win = tiny_window
    win._on_refine_finished(
        True,
        "完了: Rwp = 1.000%  χ² = 2.000\n未収束(評価回数の上限で停止)。もう一度「精密化を実行」",
    )
    text = win.progress_label.text()
    assert text.startswith("完了: Rwp") and "未収束" in text
    assert "未収束" in win.progress_label.toolTip()


def test_undo_record_is_cleared_when_project_is_replaced(
    qtbot, tiny_window: MainWindow, tmp_path: Path, monkeypatch
) -> None:
    win = tiny_window
    path = tmp_path / "a.kaisetsu.json"
    win.project_path = path
    assert win.save_project()
    # プロジェクト A で精密化した体(_start_worker と同じ記録)
    old = win.project.snapshot()
    old["h0:inst:zero"] = 0.123
    win._undo_values = old
    win.act_undo.setEnabled(True)
    # 保存直後なので確認なしで開ける
    monkeypatch.setattr(win, "_ask", lambda box: pytest.fail("確認が出た"))
    assert win.open_project_file(str(path))
    assert win._undo_values is None and not win.act_undo.isEnabled() and not win.is_dirty()
    win.undo_refine()  # 何も起きない
    assert win.project.params["h0:inst:zero"].value == 0.0
    # 新規プロジェクトでも消える
    win._undo_values = old
    win.act_undo.setEnabled(True)
    assert win.new_project(confirm=False)
    assert win._undo_values is None and not win.act_undo.isEnabled()
    assert not win.project.histograms and not win.is_dirty()


def test_old_project_with_translation_tie_warns_on_open(qtbot, tmp_path: Path, monkeypatch) -> None:
    st = Structure(
        Lattice.tetragonal(5.5, 7.8),
        SpaceGroup.from_symbol("I4/mcm"),
        [AtomSite("Sr", "Sr", [0, 0.5, 0.25]), AtomSite("O2", "O", [0.22, 0.72, 0.0])],
    )
    proj = Project("sto")
    proj.add_phase(st, name="STO")
    d = dict(project_to_dict(proj), version="0.1.12")
    path = tmp_path / "old.kaisetsu.json"
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    win = MainWindow()
    qtbot.addWidget(win)
    shown: list[str] = []
    monkeypatch.setattr(
        mw.QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2]))
    )
    assert win.open_project_file(str(path))
    assert shown and "CIF から読み直して" in shown[0] and "STO:O2" in shown[0]
    assert any("注意:" in line for line in win.log_view.toPlainText().splitlines())
