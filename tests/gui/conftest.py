"""GUI テストはオフスクリーンで動かす。"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# offscreen プラットフォームはフォントを自動で見つけないので、Windows のフォントを指す
_win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if _win_fonts.exists():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_win_fonts))
