"""アプリケーションの起動。"""

from __future__ import annotations

import sys
from typing import cast

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import QApplication

from kaisetsu import __version__


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
        import os

        os.environ["KAISETSU_NO_UPDATE_CHECK"] = "1"  # 起動確認では GitHub に行かない
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

    win = MainWindow()
    win.show()
    for a in args:
        if a.endswith(".json"):
            win.open_project_file(a)
    if smoke_out is not None:
        _schedule_smoke_test(app, smoke_out)
    return app.exec()


def _schedule_smoke_test(app: QApplication, out_path: str) -> None:
    """ウィンドウが出た直後に自己診断を走らせ、結果を書いて終了する。"""
    from pathlib import Path

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
