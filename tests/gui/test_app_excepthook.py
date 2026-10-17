"""捕まえられなかった例外のダイアログとファイルログ(レビュー 2026-10-17 GUI #2)。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from kaisetsu.gui import app as app_mod


def _drop_file_handler() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if h.get_name() == "kaisetsu-file":
            root.removeHandler(h)
            h.close()


def test_excepthook_logs_and_shows_dialog(qtbot, tmp_path: Path, monkeypatch) -> None:
    old_hook = sys.excepthook
    try:
        path = app_mod.setup_logging(tmp_path / "log")
        assert path == tmp_path / "log" / "kaisetsu.log"
        assert app_mod.setup_logging(tmp_path / "log") == path  # 2 回呼んでも重複しない
        assert sum(h.get_name() == "kaisetsu-file" for h in logging.getLogger().handlers) == 1
        shown: list[str] = []
        monkeypatch.setattr(
            app_mod.QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a[2]))
        )
        app_mod.install_excepthook(path)
        assert sys.excepthook is not old_hook
        try:
            raise ValueError("boom 保存できない")
        except ValueError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
        assert shown and "ValueError: boom" in shown[0] and str(path) in shown[0]
        for h in logging.getLogger().handlers:
            h.flush()
        text = path.read_text(encoding="utf-8")
        assert "boom 保存できない" in text and "Traceback" in text and "kaisetsu" in text
    finally:
        sys.excepthook = old_hook
        _drop_file_handler()


def test_log_dir_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(app_mod.LOG_DIR_ENV, str(tmp_path / "x"))
    assert app_mod.log_dir() == tmp_path / "x"
    monkeypatch.delenv(app_mod.LOG_DIR_ENV)
    d = app_mod.log_dir()
    assert d.name == "log" and "kaisetsu" in d.parts
    # 書けない場所なら None(落ちない)
    bad = tmp_path / "file"
    bad.write_text("x", encoding="utf-8")
    try:
        assert app_mod.setup_logging(bad / "sub") is None
    finally:
        _drop_file_handler()
