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
    app = create_app(argv)
    from kaisetsu.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    args = argv if argv is not None else sys.argv[1:]
    for a in args:
        if a.endswith(".json"):
            win.open_project_file(a)
    return app.exec()
