"""PyInstaller 用の起動スクリプト(`uv run kaisetsu-gui` と同じ)。"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    # 配布フォルダ内のサンプルデータを初期フォルダの候補にする
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ.setdefault("KAISETSU_SAMPLE_DIR", os.path.join(base, "data", "public"))
    from kaisetsu.gui.app import main as gui_main

    return gui_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
