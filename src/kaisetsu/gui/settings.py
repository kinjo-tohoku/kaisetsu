"""アプリの設定(QSettings)の置き場所。"""

from __future__ import annotations

import os

from PySide6.QtCore import QSettings

ENV_SETTINGS_FILE = "KAISETSU_SETTINGS_FILE"


def app_settings() -> QSettings:
    """アプリの設定を開く。

    通常は OS の標準の置き場所(Windows ならレジストリの kaisetsu/kaisetsu)。環境変数
    KAISETSU_SETTINGS_FILE に ini ファイルのパスを入れるとそのファイルを使う。テストで利用者の
    設定(窓の配置、前回のフォルダ)を汚さないため。USB メモリで持ち歩くときにも使える。
    QSettings(org, app) の形は setDefaultFormat() が効かないので、ここで分ける。
    """
    path = os.environ.get(ENV_SETTINGS_FILE)
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings("kaisetsu", "kaisetsu")
