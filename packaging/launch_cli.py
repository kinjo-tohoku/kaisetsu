"""PyInstaller 用のコマンドライン起動スクリプト(配布物の kaisetsu-cli.exe)。"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ.setdefault("KAISETSU_SAMPLE_DIR", os.path.join(base, "data", "public"))
    os.environ.setdefault("KAISETSU_DOCS_DIR", os.path.join(base, "docs", "user-guide"))
    from kaisetsu.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
