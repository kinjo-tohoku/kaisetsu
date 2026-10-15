"""ヘルプ → 利用者ガイド。同梱の説明書(Markdown)をアプリの中で読む窓。

左に説明書の一覧と目次、右に本文。Markdown の表示は Qt の QTextDocument に任せる
(見出し・表・コードブロック・リンクが出る)。説明書の中のリンクは、別の説明書(``install.md``)なら
この窓で開き、外部の URL ならブラウザで開く。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.i18n import tr
from kaisetsu.userdocs import DOCS, GUIDE_URL, doc_title, read_doc, section_headings, user_guide_dir


class HelpWindow(QWidget):
    """利用者ガイドの表示窓(モードレス。メインウィンドウと並べて読める)。"""

    def __init__(self, parent: QWidget | None = None, directory: Path | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(tr("利用者ガイド"))
        self.resize(1000, 720)
        self.directory = directory if directory is not None else user_guide_dir()
        self.current: str = ""
        self._text: str = ""

        self.doc_list = QListWidget()
        self.doc_list.setMaximumHeight(90)
        for name, title in DOCS:
            it = QListWidgetItem(tr(title))
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.doc_list.addItem(it)
        self.doc_list.currentItemChanged.connect(self._on_doc_selected)

        self.toc = QListWidget()
        self.toc.itemClicked.connect(self._on_toc_clicked)

        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self._on_anchor)
        self.browser.document().setDocumentMargin(12)
        self.browser.document().setDefaultStyleSheet(
            "code, pre { font-family: Consolas, 'Courier New', monospace; }"
            "table { border-collapse: collapse; }"
            "th, td { border: 1px solid #aaa; padding: 2px 6px; }"
        )

        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("本文を検索(Enter で次へ)"))
        self.search.returnPressed.connect(self.find_next)
        btn_prev = QPushButton(tr("前を検索"))
        btn_prev.clicked.connect(self.find_prev)
        btn_next = QPushButton(tr("次を検索"))
        btn_next.clicked.connect(self.find_next)
        self.btn_online = QPushButton(tr("ブラウザで開く(GitHub)"))
        self.btn_online.clicked.connect(self.open_online)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel(tr("説明書")))
        ll.addWidget(self.doc_list)
        ll.addWidget(QLabel(tr("目次")))
        ll.addWidget(self.toc, 1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(self.search, 1)
        bar.addWidget(btn_prev)
        bar.addWidget(btn_next)
        bar.addWidget(self.btn_online)
        rl.addLayout(bar)
        rl.addWidget(self.browser, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 740])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        QShortcut(QKeySequence.StandardKey.Find, self, self._focus_search)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

        if self.directory is None:
            self.browser.setMarkdown(
                tr("説明書がこの配布物に入っていません。")
                + "\n\n"
                + tr("GitHub の説明書を開いてください:")
                + f" <{GUIDE_URL}>"
            )
        else:
            self.doc_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    def show_doc(self, name: str, heading: str | None = None) -> None:
        """説明書 ``name`` を表示する。``heading`` を渡すとその節まで送る。"""
        if self.directory is None:
            return
        if name != self.current:
            try:
                text = read_doc(name, self.directory)
            except OSError:
                text = tr("説明書が見つかりません:") + f" {name}"
            self.current = name
            self._text = text
            self.browser.setMarkdown(text)
            self.browser.moveCursor(QTextCursor.MoveOperation.Start)
            self.setWindowTitle(tr("利用者ガイド") + " — " + doc_title(text, name))
            self.toc.clear()
            for h in section_headings(text):
                self.toc.addItem(h)
            for i in range(self.doc_list.count()):
                if self.doc_list.item(i).data(Qt.ItemDataRole.UserRole) == name:
                    self.doc_list.blockSignals(True)
                    self.doc_list.setCurrentRow(i)
                    self.doc_list.blockSignals(False)
        if heading:
            self.scroll_to_heading(heading)

    def scroll_to_heading(self, heading: str) -> bool:
        """見出しの行まで送る(本文の中で見出しの文を探す)。"""
        self.browser.moveCursor(QTextCursor.MoveOperation.Start)
        found = self.browser.find(heading, QTextDocument.FindFlag.FindWholeWords)
        if not found:
            found = self.browser.find(heading)
        if found:
            cur = self.browser.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            self.browser.setTextCursor(cur)
            self.browser.ensureCursorVisible()
            # 見出しが画面の上に来るように、見つけた位置を上端へ
            rect = self.browser.cursorRect(cur)
            bar = self.browser.verticalScrollBar()
            bar.setValue(bar.value() + rect.top())
        return found

    def find_next(self) -> None:
        q = self.search.text()
        if q and not self.browser.find(q):
            self.browser.moveCursor(QTextCursor.MoveOperation.Start)
            self.browser.find(q)

    def find_prev(self) -> None:
        q = self.search.text()
        if q and not self.browser.find(q, QTextDocument.FindFlag.FindBackward):
            self.browser.moveCursor(QTextCursor.MoveOperation.End)
            self.browser.find(q, QTextDocument.FindFlag.FindBackward)

    def open_online(self) -> None:
        QDesktopServices.openUrl(QUrl(GUIDE_URL))

    # ------------------------------------------------------------------
    def _focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _on_doc_selected(self, item: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if item is not None:
            self.show_doc(str(item.data(Qt.ItemDataRole.UserRole)))

    def _on_toc_clicked(self, item: QListWidgetItem) -> None:
        self.scroll_to_heading(item.text())

    def _on_anchor(self, url: QUrl) -> None:
        """本文のリンク: 別の説明書ならこの窓で、外部 URL ならブラウザで開く。"""
        s = url.toString()
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        path, _, frag = s.partition("#")
        name = Path(path).name if path else self.current
        if name.endswith(".md"):
            self.show_doc(name)
        if frag:
            # "#見出し" は見出しの文で探す(Markdown の自動アンカーは当てにしない)
            self.scroll_to_heading(frag.replace("-", " "))


__all__ = ["HelpWindow"]
