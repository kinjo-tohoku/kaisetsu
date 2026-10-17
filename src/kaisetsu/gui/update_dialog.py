"""新しい版の確認(ヘルプ → 更新を確認、起動時の自動確認)とインストーラのダウンロード。

確認は GitHub の Releases にアクセスするだけで、こちらから送る情報は無い。
起動時の自動確認は設定 "update/check_on_start"(既定: 有効)で、1 日 1 回まで。
環境変数 KAISETSU_NO_UPDATE_CHECK があれば自動確認はしない(テストと --smoke-test 用)。
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QStandardPaths, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from kaisetsu import __version__
from kaisetsu.gui.i18n import tr
from kaisetsu.update import (
    DownloadCancelled,
    InstallKind,
    ReleaseInfo,
    check_for_update,
    download_file,
    fetch_latest,
    install_kind,
    launch_installer,
)

SETTING_CHECK_ON_START = "update/check_on_start"
SETTING_SKIPPED = "update/skipped_version"
SETTING_LAST_CHECK = "update/last_check"
AUTO_CHECK_INTERVAL = _dt.timedelta(hours=20)
"""起動時の自動確認の間隔(1 日 1 回のつもり。日付が変わって最初の起動で確認する)"""
NO_CHECK_ENV = "KAISETSU_NO_UPDATE_CHECK"
#: GitHub への問い合わせの待ち時間 [秒]。終了時はこの間だけプロセスが残りうる
CHECK_TIMEOUT = 5.0

Fetcher = Callable[[], ReleaseInfo]

#: 終了時に切り離した確認スレッド(走ったまま QThread を壊すと Qt が異常終了するので、
#: `wait_for_detached()` で終わるのを待ってから壊す)
_DETACHED: list[tuple[QThread, QObject]] = []


def wait_for_detached(msec: int = 15000) -> None:
    """切り離した確認スレッドが終わるのを待つ(app.main の最後で呼ぶ)。"""
    for thread, _worker in _DETACHED:
        if thread.isRunning():
            thread.wait(msec)
    _DETACHED.clear()


def check_on_start(settings: QSettings) -> bool:
    v = settings.value(SETTING_CHECK_ON_START, True)
    return str(v).lower() not in ("false", "0", "no")


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def should_auto_check(settings: QSettings) -> bool:
    """起動時に自動で確認するか(設定が有効で、前回から AUTO_CHECK_INTERVAL 以上たっている)。"""
    if os.environ.get(NO_CHECK_ENV):
        return False
    if not check_on_start(settings):
        return False
    last = settings.value(SETTING_LAST_CHECK, "")
    if last:
        try:
            t = _dt.datetime.fromisoformat(str(last))
            if t.tzinfo is None:
                t = t.replace(tzinfo=_dt.UTC)
            if _now() - t < AUTO_CHECK_INTERVAL:
                return False
        except ValueError:
            pass
    return True


class _CheckWorker(QObject):
    finished = Signal(object, str)  # ReleaseInfo | None, エラー文(無ければ "")

    def __init__(self, fetch: Fetcher | None) -> None:
        super().__init__()
        self.fetch = fetch

    def run(self) -> None:
        try:
            fetch = self.fetch if self.fetch is not None else _default_fetch
            info = check_for_update(fetch=fetch)
            self.finished.emit(info, "")
        except Exception as exc:
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


def _default_fetch() -> ReleaseInfo:
    return fetch_latest(timeout=CHECK_TIMEOUT)


class _DownloadWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(bool, str, str)  # 成否、メッセージ、保存先

    def __init__(self, url: str, dest: Path, expected_size: int = 0) -> None:
        super().__init__()
        self.url = url
        self.dest = dest
        self.expected_size = expected_size
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            path = download_file(
                self.url, self.dest, progress=self.progress.emit, cancel=lambda: self._cancel
            )
            size = path.stat().st_size
            if self.expected_size and size != self.expected_size:
                path.unlink(missing_ok=True)
                raise OSError(
                    f"ファイルの大きさが Release の記載と違います({size} / {self.expected_size} バイト)"
                )
            self.finished.emit(True, "", str(path))
        except DownloadCancelled:
            self.finished.emit(False, "中止しました", "")
        except Exception as exc:
            self.finished.emit(False, f"{type(exc).__name__}: {exc}", "")


def download_dir() -> Path:
    """インストーラの保存先(ダウンロードフォルダ。無ければ一時フォルダ)。"""
    loc = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
    if loc and Path(loc).is_dir():
        return Path(loc)
    return Path(tempfile.gettempdir())


class UpdateDialog(QDialog):
    """新しい版の案内。インストーラ版ならその場でダウンロードして入れ替える。"""

    def __init__(
        self,
        info: ReleaseInfo,
        kind: InstallKind,
        settings: QSettings,
        parent: QWidget | None = None,
        *,
        confirm_quit: Callable[[], bool] | None = None,
        launch: Callable[[Path], None] = launch_installer,
    ) -> None:
        super().__init__(parent)
        self.info = info
        self.kind = kind
        self.settings = settings
        self.confirm_quit = confirm_quit
        self.launch = launch
        self.installer_path: Path | None = None
        self._thread: QThread | None = None
        self._worker: _DownloadWorker | None = None
        self.setWindowTitle(tr("更新の確認"))
        self.resize(640, 520)
        layout = QVBoxLayout(self)

        when = ""
        if info.published:
            try:  # GitHub の UTC を手元の日付にする
                t = _dt.datetime.fromisoformat(info.published.replace("Z", "+00:00"))
                when = f"({t.astimezone().strftime('%Y-%m-%d')} 公開)"
            except ValueError:
                when = f"({info.published[:10]} 公開)"
        head = QLabel(
            f"<b>kaisetsu {info.version}</b> が公開されています {when}<br>使用中の版: {__version__}"
        )
        head.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(head)

        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        if info.notes.strip():
            self.notes.setMarkdown(info.notes)
        else:
            self.notes.setPlainText("変更点は Release ページで確認してください。")
        layout.addWidget(self.notes, 1)

        hint = {
            "installer": (
                "「ダウンロードしてインストール」を押すと、インストーラをダウンロードしてから "
                "kaisetsu を終了し、インストーラを起動します(管理者権限は要りません)。"
                "設定とプロジェクトファイルはそのまま残ります。"
            ),
            "portable": (
                "zip 版をお使いです。Release ページから kaisetsu-windows.zip をダウンロードして展開し、"
                "フォルダごと差し替えてください。設定とプロジェクトファイルはそのまま残ります。"
            ),
            "python": (
                "Python 環境に入っています。ターミナルで `uv tool upgrade kaisetsu` を実行してください"
                "(開発版は `git pull` のあと `uv sync --extra gui`)。"
            ),
        }[kind]
        self.hint = QLabel(hint)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.status = QLabel("")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        self.install_btn = QPushButton(tr("ダウンロードしてインストール"))
        self.install_btn.clicked.connect(self.start_download)
        self.install_btn.setDefault(True)
        if kind == "installer":
            row.addWidget(self.install_btn)
        else:
            self.install_btn.hide()
        self.page_btn = QPushButton(tr("ダウンロードページを開く"))
        self.page_btn.clicked.connect(self.open_page)
        row.addWidget(self.page_btn)
        row.addStretch(1)
        self.skip_btn = QPushButton(tr("この版は知らせない"))
        self.skip_btn.clicked.connect(self.skip_version)
        row.addWidget(self.skip_btn)
        self.later_btn = QPushButton(tr("あとで"))
        self.later_btn.clicked.connect(self.reject)
        row.addWidget(self.later_btn)
        layout.addLayout(row)

    # ------------------------------------------------------------------
    def open_page(self) -> None:
        QDesktopServices.openUrl(QUrl(self.info.url))

    def skip_version(self) -> None:
        self.settings.setValue(SETTING_SKIPPED, self.info.version)
        self.reject()

    def start_download(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            return
        dest = download_dir() / f"kaisetsu-{self.info.version}-windows-setup.exe"
        self._worker = _DownloadWorker(self.info.installer_url, dest, self.info.installer_size)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_downloaded)
        # quit はワーカーのスレッドから直接呼ぶ(キュー経由だとメインのイベントループが回る
        # までスレッドが終わらず、closeEvent の wait() が待ち時間いっぱいまで止まる)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self.install_btn.setText(tr("中止"))
        self.skip_btn.setEnabled(False)
        self.later_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText(f"ダウンロード中: {dest}")
        self._thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * done / total))
            self.status.setText(f"ダウンロード中: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        else:
            self.progress.setRange(0, 0)
            self.status.setText(f"ダウンロード中: {done / 1e6:.0f} MB")

    def _on_downloaded(self, ok: bool, message: str, path: str) -> None:
        self.install_btn.setText(tr("ダウンロードしてインストール"))
        self.skip_btn.setEnabled(True)
        self.later_btn.setEnabled(True)
        self.progress.hide()
        if not ok:
            self.status.setText(message)
            if message != "中止しました":
                QMessageBox.warning(
                    self, tr("更新の確認"), f"ダウンロードできませんでした:\n{message}"
                )
            return
        self.installer_path = Path(path)
        self.status.setText(f"ダウンロードしました: {path}")
        self.run_installer()

    def run_installer(self) -> None:
        """kaisetsu を終了してインストーラを起動する。終了を断られたら保存先だけ知らせる。"""
        path = self.installer_path
        if path is None:
            return
        if self.confirm_quit is not None and not self.confirm_quit():
            QMessageBox.information(
                self,
                tr("更新の確認"),
                f"インストーラを保存しました:\n{path}\n\nあとで実行すると新しい版に入れ替わります。",
            )
            return
        try:
            self.launch(path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                tr("更新の確認"),
                f"インストーラを起動できませんでした:\n{exc}\n\n保存先: {path}",
            )
            return
        self.accept()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.closeAllWindows()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            self._thread.wait(5000)
        event.accept()


class UpdateChecker(QObject):
    """メインウィンドウに 1 つ持たせる。確認をスレッドで走らせ、結果を出す。

    `fetch` を差し替えるとネットワーク無しで試せる(テスト用)。
    """

    checked = Signal(object, str)  # ReleaseInfo | None, エラー文

    def __init__(
        self,
        parent: QWidget,
        settings: QSettings,
        *,
        fetch: Fetcher | None = None,
        log: Callable[[str], None] | None = None,
        confirm_quit: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.widget = parent
        self.settings = settings
        self.fetch = fetch
        self.log = log
        self.confirm_quit = confirm_quit
        self.dialog: UpdateDialog | None = None
        self.last_info: ReleaseInfo | None = None
        self.last_error = ""
        self._manual = False
        self._busy = False
        self._thread: QThread | None = None
        self._worker: _CheckWorker | None = None

    def should_auto_check(self) -> bool:
        return should_auto_check(self.settings)

    def check(self, *, manual: bool) -> bool:
        """確認を始める。すでに走っていれば False。"""
        if self._busy:
            return False
        if self._thread is not None and self._thread.isRunning():
            # 前回の結果は受け取り済みで、スレッドが終わるところ(macOS では数十 ms かかる)
            self._thread.wait(5000)
        self._busy = True
        self._manual = manual
        self._worker = _CheckWorker(self.fetch)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._done)
        # quit はワーカーのスレッドから直接呼ぶ(キュー経由だと、メインのイベントループが
        # 止まっている wait_for_detached() の中でスレッドが終われず、待ち時間を使い切る)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        if manual and self.log is not None:
            self.log("新しい版があるか GitHub に問い合わせています…")
        self._thread.start()
        return True

    def wait(self, msec: int = 5000) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(msec)

    def detach(self) -> None:
        """終了時: 走っている確認を切り離して結果を捨てる(待たない)。

        親付きの QThread を走ったまま壊すと Qt が異常終了する(終了コード 127、配布物では
        「動作を停止しました」。レビュー 2026-10-17 GUI #13)。親を外してモジュールの一覧に移し、
        結果の受け取りを切る。終わるのは `wait_for_detached()`(app.main の最後)で待つ。
        """
        thread, worker = self._thread, self._worker
        self._busy = False
        if thread is None or not thread.isRunning():
            return
        if worker is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                worker.finished.disconnect(self._done)
            _DETACHED.append((thread, worker))
        thread.setParent(None)
        self._thread = None
        self._worker = None

    def _done(self, info: object, error: str) -> None:
        self._busy = False
        self.settings.setValue(SETTING_LAST_CHECK, _now().isoformat(timespec="seconds"))
        self.last_error = error
        self.last_info = info if isinstance(info, ReleaseInfo) else None
        self.checked.emit(self.last_info, error)
        if error:
            if self._manual:
                QMessageBox.warning(
                    self.widget,
                    tr("更新の確認"),
                    f"新しい版を確認できませんでした(ネットワークに繋がっていますか?):\n{error}",
                )
            elif self.log is not None:
                self.log(f"更新の確認: 問い合わせできませんでした({error})")
            return
        if self.last_info is None:
            if self._manual:
                QMessageBox.information(
                    self.widget, tr("更新の確認"), f"kaisetsu {__version__} は最新版です。"
                )
            return
        skipped = str(self.settings.value(SETTING_SKIPPED, ""))
        if not self._manual and skipped == self.last_info.version:
            if self.log is not None:
                self.log(f"更新の確認: {self.last_info.version} は知らせない設定です")
            return
        self.show_dialog(self.last_info)

    def show_dialog(self, info: ReleaseInfo) -> UpdateDialog:
        if self.dialog is not None:
            self.dialog.close()
        self.dialog = UpdateDialog(
            info, install_kind(), self.settings, self.widget, confirm_quit=self.confirm_quit
        )
        self.dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.dialog.show()
        return self.dialog


__all__ = [
    "SETTING_CHECK_ON_START",
    "SETTING_SKIPPED",
    "UpdateChecker",
    "UpdateDialog",
    "check_on_start",
    "download_dir",
    "should_auto_check",
    "wait_for_detached",
]
