"""PyInstaller 用のコマンドライン起動スクリプト(配布物の kaisetsu-cli.exe)。"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # UTF-8 モードは kaisetsu.spec の EXE(options=[("X utf8_mode=1", ...)]) で有効にしている。
    # 起動後に環境変数 PYTHONUTF8 を設定しても効かない。build_windows.ps1 が
    # `--utf8-mode-check` で効いているかを確かめる(配布物の点検用)。
    if sys.argv[1:] == ["--utf8-mode-check"]:
        import locale

        print(f"utf8_mode={sys.flags.utf8_mode} preferred={locale.getpreferredencoding(False)}")
        return 0
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ.setdefault("KAISETSU_SAMPLE_DIR", os.path.join(base, "data", "public"))
    os.environ.setdefault("KAISETSU_DOCS_DIR", os.path.join(base, "docs", "user-guide"))
    from kaisetsu.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
