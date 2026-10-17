"""更新確認のスレッドが走ったまま終了しても異常終了しない(レビュー 2026-10-17 GUI #13)。"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from kaisetsu.gui import update_dialog as ud
from kaisetsu.gui.update_dialog import UpdateChecker, wait_for_detached
from kaisetsu.update import ReleaseInfo

NEW = ReleaseInfo(version="99.0.0", tag="v99.0.0", url="https://example.invalid/v99")


def test_detach_discards_result(qtbot, tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    win = QWidget()
    qtbot.addWidget(win)

    def slow() -> ReleaseInfo:
        time.sleep(0.5)
        return NEW

    checker = UpdateChecker(win, settings, fetch=slow)
    got: list[object] = []
    checker.checked.connect(lambda *a: got.append(a))
    assert checker.check(manual=False)
    assert checker._thread is not None and checker._thread.isRunning()
    checker.detach()
    assert checker._thread is None and checker._worker is None and not checker._busy
    assert len(ud._DETACHED) == 1
    wait_for_detached(5000)
    qtbot.wait(200)  # 届いているシグナルがあれば処理される
    assert got == [] and checker.dialog is None and ud._DETACHED == []
    # 切り離した後も新しい確認はできる
    checker.fetch = lambda: NEW
    with qtbot.waitSignal(checker.checked, timeout=5000):
        assert checker.check(manual=False)
    assert checker.dialog is not None
    checker.dialog.close()
    checker.detach()  # 走っていなければ何もしない
    assert ud._DETACHED == []


def test_exit_while_check_is_running_is_clean(tmp_path: Path) -> None:
    """窓を閉じて終了するとき、確認スレッド(6 秒かかる)を壊さずに終わる(以前は終了コード 127)。"""
    script = textwrap.dedent(
        """
        import gc, os, sys, time
        from PySide6.QtCore import QTimer
        from kaisetsu.gui.app import create_app
        from kaisetsu.gui.main_window import MainWindow
        from kaisetsu.gui.update_dialog import wait_for_detached
        from kaisetsu.update import ReleaseInfo

        def slow():
            time.sleep(4.0)
            return ReleaseInfo(version="0.0.1", tag="v0.0.1", url="https://example.invalid")

        app = create_app([])
        w = MainWindow()
        w.updater.fetch = slow
        w.show()
        assert w.updater.check(manual=False)
        QTimer.singleShot(200, w.close)
        QTimer.singleShot(400, app.quit)
        app.exec()
        wait_for_detached()
        del w
        gc.collect()
        print("clean exit")
        """
    )
    env = dict(os.environ)
    env.update(
        QT_QPA_PLATFORM="offscreen",
        KAISETSU_NO_UPDATE_CHECK="1",
        KAISETSU_SETTINGS_FILE=str(tmp_path / "s.ini"),
        KAISETSU_LOG_DIR=str(tmp_path / "log"),
        PYTHONUTF8="1",
    )
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "clean exit" in r.stdout
    assert time.perf_counter() - t0 < 60
