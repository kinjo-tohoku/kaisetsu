"""パラメータ欄を狭くできること、窓の配置が次回に引き継がれること。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from kaisetsu.gui.main_window import PARAM_DOCK_WIDTH, MainWindow
from kaisetsu.gui.settings import app_settings


def test_param_dock_can_be_narrow(qtbot) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    # オフスクリーンの画面(800×600)に収まる大きさ(収まらないと窓が詰められ、ドックの幅も変わる)
    win.resize(780, 560)
    win.show()
    QApplication.processEvents()
    # 以前はドックの最小幅 400 px と横並びのボタン(9 個 + 3 個)で 600 px 以上から狭くできなかった
    assert win.param_dock.widget().minimumSizeHint().width() < 300
    assert win.param_dock.minimumWidth() < 300
    win.resizeDocks([win.param_dock], [260], Qt.Orientation.Horizontal)
    QApplication.processEvents()
    assert win.param_dock.width() <= 270
    # 広げるのも自由(窓 780 px から中央の最小幅を引いても 500 px は入る)
    win.resizeDocks([win.param_dock], [500], Qt.Orientation.Horizontal)
    QApplication.processEvents()
    assert win.param_dock.width() >= 490


def test_layout_is_remembered(qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAISETSU_SETTINGS_FILE", str(tmp_path / "k.ini"))
    win = MainWindow()
    qtbot.addWidget(win)
    # オフスクリーンの画面(800×600)に収まる大きさにする(収まらないと復元時に幅が詰められる)
    win.resize(780, 560)
    win.show()
    QApplication.processEvents()
    # 初回はパラメータ欄が既定の幅
    assert abs(win.param_dock.width() - PARAM_DOCK_WIDTH) <= 2
    win.resizeDocks([win.param_dock], [300], Qt.Orientation.Horizontal)
    QApplication.processEvents()
    win.close()
    QApplication.processEvents()
    assert (tmp_path / "k.ini").exists()
    # 設定は環境変数の ini に書かれ、利用者の本当の設定(レジストリ)には触らない
    assert Path(app_settings().fileName()) == tmp_path / "k.ini"

    win2 = MainWindow()
    qtbot.addWidget(win2)
    win2.show()
    QApplication.processEvents()
    assert abs(win2.param_dock.width() - 300) <= 2
    # 表示/非表示も引き継ぐ(「表示」メニューで戻せる)
    win2.param_dock.hide()
    win2.close()
    QApplication.processEvents()
    win3 = MainWindow()
    qtbot.addWidget(win3)
    win3.show()
    QApplication.processEvents()
    assert not win3.param_dock.isVisible()
    win3.param_dock.toggleViewAction().trigger()
    assert win3.param_dock.isVisible()


def test_settings_env_is_isolated_in_tests() -> None:
    # tests/gui/conftest.py が一時フォルダの ini に向けている
    assert os.environ.get("KAISETSU_SETTINGS_FILE", "").endswith("kaisetsu.ini")
