"""更新の確認: 偽の fetch で新しい版のダイアログを出し、スキップ・ダウンロード → インストーラ起動の流れを
確かめる(オフスクリーン、ネットワークには行かない)。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QWidget

from kaisetsu.gui import update_dialog as ud
from kaisetsu.gui.main_window import MainWindow
from kaisetsu.gui.update_dialog import (
    SETTING_CHECK_ON_START,
    SETTING_LAST_CHECK,
    SETTING_SKIPPED,
    UpdateChecker,
    UpdateDialog,
    should_auto_check,
)
from kaisetsu.update import ReleaseInfo

NEW = ReleaseInfo(
    version="99.0.0",
    tag="v99.0.0",
    url="https://example.invalid/releases/tag/v99.0.0",
    notes="## kaisetsu 99.0.0\n\n- テスト用の変更点",
    published="2026-10-15T08:00:00Z",
    installer_url="https://example.invalid/kaisetsu-windows-setup.exe",
)
OLD = ReleaseInfo(version="0.0.1", tag="v0.0.1", url="https://example.invalid/old")


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "test.ini"), QSettings.Format.IniFormat)


def test_should_auto_check(settings: QSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAISETSU_NO_UPDATE_CHECK", raising=False)
    assert should_auto_check(settings)
    settings.setValue(SETTING_LAST_CHECK, ud._now().isoformat())
    assert not should_auto_check(settings)
    settings.setValue(SETTING_LAST_CHECK, (ud._now() - dt.timedelta(days=2)).isoformat())
    assert should_auto_check(settings)
    settings.setValue(SETTING_LAST_CHECK, "not a date")
    assert should_auto_check(settings)
    settings.setValue(SETTING_CHECK_ON_START, False)
    assert not should_auto_check(settings)
    settings.setValue(SETTING_CHECK_ON_START, True)
    monkeypatch.setenv("KAISETSU_NO_UPDATE_CHECK", "1")
    assert not should_auto_check(settings)


def test_checker_shows_dialog_and_skip(
    qtbot, settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ud, "install_kind", lambda: "installer")
    win = QWidget()
    qtbot.addWidget(win)
    checker = UpdateChecker(win, settings, fetch=lambda: NEW)
    with qtbot.waitSignal(checker.checked, timeout=5000):
        assert checker.check(manual=False)
    assert checker.last_info == NEW and checker.last_error == ""
    dlg = checker.dialog
    assert dlg is not None and dlg.isVisible()
    assert not dlg.install_btn.isHidden()
    assert "99.0.0" in dlg.notes.toPlainText()
    assert settings.value(SETTING_LAST_CHECK)

    # 「この版は知らせない」→ 自動確認ではもう出ない、手動なら出る
    dlg.skip_version()
    assert settings.value(SETTING_SKIPPED) == "99.0.0" and not dlg.isVisible()
    checker.dialog = None
    with qtbot.waitSignal(checker.checked, timeout=5000):
        checker.check(manual=False)
    assert checker.dialog is None
    with qtbot.waitSignal(checker.checked, timeout=5000):
        checker.check(manual=True)
    assert checker.dialog is not None and checker.dialog.isVisible()
    checker.dialog.close()


def test_checker_up_to_date_and_error(
    qtbot, settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    msgs: list[tuple[str, str]] = []
    monkeypatch.setattr(ud.QMessageBox, "information", lambda *a, **k: msgs.append(("info", a[2])))
    monkeypatch.setattr(ud.QMessageBox, "warning", lambda *a, **k: msgs.append(("warn", a[2])))
    win = QWidget()
    qtbot.addWidget(win)
    checker = UpdateChecker(win, settings, fetch=lambda: OLD)
    with qtbot.waitSignal(checker.checked, timeout=5000):
        checker.check(manual=True)
    assert msgs[-1][0] == "info" and "最新版" in msgs[-1][1]
    assert checker.dialog is None

    def boom() -> ReleaseInfo:
        raise OSError("no network")

    checker.fetch = boom
    with qtbot.waitSignal(checker.checked, timeout=5000):
        checker.check(manual=True)
    assert msgs[-1][0] == "warn" and "no network" in msgs[-1][1]
    # 自動確認の失敗はログだけ
    logs: list[str] = []
    checker.log = logs.append
    with qtbot.waitSignal(checker.checked, timeout=5000):
        checker.check(manual=False)
    assert len(msgs) == 2 and logs and "no network" in logs[-1]


def test_dialog_download_then_launch(
    qtbot, settings: QSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"MZ" + bytes(5000)
    monkeypatch.setattr(ud, "download_dir", lambda: tmp_path)

    def fake_download(url: str, dest: Path, *, progress=None, cancel=None) -> Path:
        assert url == NEW.installer_url
        dest.write_bytes(payload)
        if progress is not None:
            progress(len(payload), len(payload))
        return dest

    monkeypatch.setattr(ud, "download_file", fake_download)
    launched: list[Path] = []
    confirmed: list[bool] = []

    def confirm() -> bool:
        confirmed.append(True)
        return True

    dlg = UpdateDialog(NEW, "installer", settings, confirm_quit=confirm, launch=launched.append)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.start_download()
    qtbot.waitUntil(lambda: dlg.installer_path is not None, timeout=5000)
    expect = tmp_path / "kaisetsu-99.0.0-windows-setup.exe"
    assert dlg.installer_path == expect and expect.read_bytes() == payload
    assert confirmed == [True] and launched == [expect]
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert dlg.progress.value() == 1000 or dlg.progress.isHidden()


def test_dialog_download_refused_quit(
    qtbot, settings: QSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ud, "download_dir", lambda: tmp_path)
    monkeypatch.setattr(
        ud, "download_file", lambda url, dest, **kw: dest.write_bytes(b"x") and dest
    )
    msgs: list[str] = []
    monkeypatch.setattr(ud.QMessageBox, "information", lambda *a, **k: msgs.append(a[2]))
    launched: list[Path] = []
    dlg = UpdateDialog(
        NEW, "installer", settings, confirm_quit=lambda: False, launch=launched.append
    )
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.start_download()
    qtbot.waitUntil(lambda: dlg.installer_path is not None, timeout=5000)
    assert launched == [] and msgs and "保存しました" in msgs[-1]
    assert dlg.isVisible()
    dlg.close()


def test_dialog_for_portable_and_python(settings: QSettings, qtbot) -> None:
    dlg = UpdateDialog(NEW, "portable", settings)
    qtbot.addWidget(dlg)
    assert dlg.install_btn.isHidden() and "zip" in dlg.hint.text()
    dlg2 = UpdateDialog(NEW, "python", settings)
    qtbot.addWidget(dlg2)
    assert dlg2.install_btn.isHidden() and "uv tool upgrade" in dlg2.hint.text()


def test_main_window_menu_and_manual_check(qtbot, settings: QSettings) -> None:
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.act_auto_update.isCheckable()
    win.updater.settings = settings  # 利用者の本物の設定に書かない
    win.updater.fetch = lambda: NEW
    with qtbot.waitSignal(win.updater.checked, timeout=5000):
        win.check_for_updates()
    assert win.updater.dialog is not None and win.updater.dialog.isVisible()
    win.updater.dialog.close()
