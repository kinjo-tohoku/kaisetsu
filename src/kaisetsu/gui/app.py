"""アプリケーションの起動、ファイルへのログ、捕まえられなかった例外の表示。"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import cast

from PySide6.QtCore import QLocale, Qt, QThread
from PySide6.QtWidgets import QApplication, QMessageBox

from kaisetsu import __version__

LOG_DIR_ENV = "KAISETSU_LOG_DIR"
LOG_FILE_NAME = "kaisetsu.log"

log = logging.getLogger(__name__)


def log_dir() -> Path:
    """ログの置き場所。

    Windows は ``%LOCALAPPDATA%\\kaisetsu\\log``、macOS は ``~/Library/Logs/kaisetsu``、
    それ以外は ``$XDG_STATE_HOME``(無ければ ``~/.local/state``)``/kaisetsu/log``。
    環境変数 KAISETSU_LOG_DIR があればそこ(テスト用)。
    """
    override = os.environ.get(LOG_DIR_ENV)
    if override:
        return Path(override)
    system = str(sys.platform)  # mypy が実行環境の OS で分岐を消さないように文字列にする
    if system == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "kaisetsu" / "log"
    if system == "darwin":
        return Path.home() / "Library" / "Logs" / "kaisetsu"
    state = os.environ.get("XDG_STATE_HOME")
    root = Path(state) if state else Path.home() / ".local" / "state"
    return root / "kaisetsu" / "log"


def setup_logging(directory: Path | None = None) -> Path | None:
    """ファイルへのログ(kaisetsu.log、1 MB × 3 世代)を始める。書けなければ None。

    配布物は console=False で stderr が無いので、警告やエラーはここにしか残らない。
    """
    directory = directory if directory is not None else log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOG_FILE_NAME
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError as exc:
        log.warning("ログファイルを開けませんでした: %s", exc)
        return None
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    handler.set_name("kaisetsu-file")
    root = logging.getLogger()
    for h in list(root.handlers):
        if h.get_name() == "kaisetsu-file":  # 2 回呼ばれても重複して書かない
            root.removeHandler(h)
            h.close()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    log.info(
        "kaisetsu %s を起動(Python %s, %s)",
        __version__,
        platform.python_version(),
        platform.platform(),
    )
    return path


def install_excepthook(log_path: Path | None = None) -> None:
    """捕まえられなかった例外をログに残し、ダイアログで知らせる。

    PySide6 はスロット(メニューやボタンの処理)の中で起きた例外を sys.excepthook に渡して
    処理を続けるので、ここで受ければ配布物でも「保存できなかった」「書き出せなかった」が
    見える(レビュー 2026-10-17 GUI #2)。ワーカースレッドの例外は各ワーカーが受けてメッセージに
    しているので、ここに来るのはメインスレッドの例外。
    """
    showing = False

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        nonlocal showing
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("kaisetsu").error("捕まえられなかった例外:\n%s", text)
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        app = QApplication.instance()
        if app is None or QThread.currentThread() is not app.thread() or showing:
            sys.__excepthook__(exc_type, exc, tb)
            return
        showing = True
        try:
            where = f"\n\n詳しい記録: {log_path}" if log_path is not None else ""
            QMessageBox.critical(
                None,
                "kaisetsu のエラー",
                "処理の途中でエラーが起きました(その操作は完了していません):\n"
                f"{exc_type.__name__}: {exc}{where}",
            )
        except Exception:  # ダイアログも出せない状態なら標準の出力に任せる
            sys.__excepthook__(exc_type, exc, tb)
        finally:
            showing = False

    sys.excepthook = hook


def create_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("kaisetsu")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("kaisetsu")
    QLocale.setDefault(QLocale(QLocale.Language.Japanese, QLocale.Country.Japan))
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    return cast(QApplication, app)


def main(argv: list[str] | None = None) -> int:
    """起動。``--lang en`` または設定 "language" で英語 UI。

    ``--smoke-test 結果ファイル`` は配布物(PyInstaller / インストーラ)の起動確認用。
    ウィンドウを出した後に `kaisetsu.selftest.quick_check` を走らせ、結果をファイルに書いて終了する
    (成功なら終了コード 0、ファイルの先頭行が ``ok <version>``)。
    """
    args = list(argv if argv is not None else sys.argv[1:])
    smoke_out: str | None = None
    if "--smoke-test" in args:
        i = args.index("--smoke-test")
        has_path = i + 1 < len(args) and not args[i + 1].startswith("--")
        smoke_out = args[i + 1] if has_path else ""
        del args[i : i + (2 if has_path else 1)]
        os.environ["KAISETSU_NO_UPDATE_CHECK"] = "1"  # 起動確認では GitHub に行かない
    log_path = setup_logging()
    install_excepthook(log_path)
    from kaisetsu.gui.i18n import set_language
    from kaisetsu.gui.settings import app_settings

    lang = str(app_settings().value("language", "ja"))
    if "--lang" in args:
        i = args.index("--lang")
        if i + 1 < len(args):
            lang = args[i + 1]
            del args[i : i + 2]
    set_language(lang)
    app = create_app(argv)
    from kaisetsu.gui.main_window import MainWindow
    from kaisetsu.gui.update_dialog import wait_for_detached

    win = MainWindow()
    win.show()
    for a in args:
        if a.endswith(".json"):
            win.open_project_file(a)
    if smoke_out is not None:
        _schedule_smoke_test(app, smoke_out)
    code = app.exec()
    # 終了時に切り離した更新確認のスレッドを、壊す前に終わらせる(走ったまま壊すと異常終了)
    wait_for_detached()
    log.info("終了(コード %d)", code)
    return code


def _schedule_smoke_test(app: QApplication, out_path: str) -> None:
    """ウィンドウが出た直後に自己診断を走らせ、結果を書いて終了する。"""
    from PySide6.QtCore import QTimer

    def _run() -> None:
        from kaisetsu.selftest import quick_check

        code = 0
        try:
            text = f"ok {__version__}\n" + quick_check()
        except Exception as exc:  # 起動確認なので何が起きても記録して終了する
            text = f"error {__version__}\n{type(exc).__name__}: {exc}\n"
            code = 1
        if out_path:
            Path(out_path).write_text(text, encoding="utf-8")
        app.exit(code)

    QTimer.singleShot(500, _run)
