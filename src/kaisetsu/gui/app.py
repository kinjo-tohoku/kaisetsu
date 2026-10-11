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
    """起動。``--lang en`` または設定 "language" で英語 UI。"""
    args = list(argv if argv is not None else sys.argv[1:])
    from PySide6.QtCore import QSettings

    from kaisetsu.gui.i18n import set_language

    lang = str(QSettings("kaisetsu", "kaisetsu").value("language", "ja"))
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
    return app.exec()
