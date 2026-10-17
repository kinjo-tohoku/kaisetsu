"""PyInstaller 用の起動スクリプト(`uv run kaisetsu-gui` と同じ)。"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # UTF-8 モードは kaisetsu.spec の EXE(options=[("X utf8_mode=1", ...)]) で有効にしている。
    # ここで環境変数 PYTHONUTF8 を設定しても、起動後では効かない。
    # 配布フォルダ内のサンプルデータ(gui/paths.py が初回にドキュメントへコピーする)
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ.setdefault("KAISETSU_SAMPLE_DIR", os.path.join(base, "data", "public"))
    # 同梱の説明書(ヘルプ → 利用者ガイド)
    os.environ.setdefault("KAISETSU_DOCS_DIR", os.path.join(base, "docs", "user-guide"))
    from kaisetsu.gui.app import main as gui_main

    return gui_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
