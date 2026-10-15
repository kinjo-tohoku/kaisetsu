"""ヘルプ → 利用者ガイド: 同梱の説明書が窓に出て、目次・リンク・検索が働く(オフスクリーン)。"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl
from PySide6.QtGui import QKeySequence

from kaisetsu.gui import help_viewer as hv
from kaisetsu.gui.help_viewer import HelpWindow
from kaisetsu.gui.main_window import MainWindow


def test_help_window_shows_guide_with_toc_and_links(qtbot) -> None:  # type: ignore[no-untyped-def]
    win = HelpWindow()
    qtbot.addWidget(win)
    win.show()  # 表示していないと本文の高さが無く、スクロールできない
    assert win.current == "getting-started.md"
    text = win.browser.toPlainText()
    assert "ウィザードで最初の精密化" in text
    assert "困ったとき" in text
    assert win.toc.count() >= 10
    assert "利用者ガイド" in win.windowTitle()

    # 目次をクリックすると見出しまで送る
    item = next(
        win.toc.item(i) for i in range(win.toc.count()) if "困ったとき" in win.toc.item(i).text()
    )
    assert win.scroll_to_heading(item.text())
    assert win.browser.verticalScrollBar().value() > 0

    # 本文の "install.md" リンクはこの窓で開く
    win._on_anchor(QUrl("install.md"))
    assert win.current == "install.md"
    assert "SmartScreen" in win.browser.toPlainText()
    assert win.doc_list.currentItem().data(0x0100) == "install.md"

    # 検索
    win.show_doc("getting-started.md")
    win.search.setText("Rwp")
    win.find_next()
    assert win.browser.textCursor().selectedText() == "Rwp"
    win.find_prev()


def test_help_window_without_bundled_docs(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(hv, "user_guide_dir", lambda: None)
    win = HelpWindow()
    qtbot.addWidget(win)
    assert win.directory is None
    assert "GitHub" in win.browser.toPlainText()
    win.show_doc("install.md")  # 何も起きない
    assert win.current == ""


def test_main_window_help_menu_opens_guide(qtbot) -> None:  # type: ignore[no-untyped-def]
    mw = MainWindow()
    qtbot.addWidget(mw)
    assert mw.act_guide.shortcut() == QKeySequence("F1")
    assert mw._help_window is None
    mw.show_user_guide()
    assert mw._help_window is not None
    assert mw._help_window.isVisible()
    assert mw._help_window.current == "getting-started.md"
    mw.show_user_guide("install.md")
    assert mw._help_window.current == "install.md"
    mw._help_window.close()
