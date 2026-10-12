"""ダイアログ共通の小道具: 画面に収まらないダイアログをスクロールできるようにする。

ノート PC(拡大率 125〜150% で使える高さが 700〜860 px)では、装置や相の設定ダイアログが
画面からはみ出して OK が押せなくなる。ここの関数を各ダイアログの最後で呼ぶと、中身を
QScrollArea に移し、OK/キャンセルのボタンだけを常に見える位置に残す。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget


def available_height(widget: QWidget) -> int:
    """ウィジェットが表示される画面の、タスクバーを除いた高さ [論理 px]。"""
    return int(widget.screen().availableGeometry().height())


def fit_dialog_to_screen(dialog: QDialog, buttons: QWidget | None = None, margin: int = 90) -> bool:
    """中身が画面より高いダイアログをスクロール領域に入れ、ボタンは常に見えるようにする。

    ダイアログに既に付いているレイアウトをそのまま QScrollArea の中身に移し、
    `buttons`(QDialogButtonBox など)だけを外に出す。収まるときは何もしない。

    Returns:
        スクロール化したら True
    """
    hint = dialog.sizeHint()
    limit = available_height(dialog) - margin
    if hint.height() <= limit:
        return False
    old = dialog.layout()
    if old is None:
        return False
    if buttons is not None:
        old.removeWidget(buttons)
    content = QWidget()
    content.setLayout(old)  # 既存レイアウトを奪う(Qt が親を付け替える)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(scroll, 1)
    if buttons is not None:
        row = QHBoxLayout()
        row.setContentsMargins(11, 6, 11, 11)
        row.addWidget(buttons)
        outer.addLayout(row)
    width = hint.width() + scroll.verticalScrollBar().sizeHint().width() + 8
    dialog.resize(width, max(limit, 320))
    return True
