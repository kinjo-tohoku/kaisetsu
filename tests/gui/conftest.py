"""GUI テストはオフスクリーンで動かす。"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# テスト中は起動時の更新確認で GitHub に行かない
os.environ.setdefault("KAISETSU_NO_UPDATE_CHECK", "1")
# offscreen プラットフォームはフォントを自動で見つけないので、Windows のフォントを指す
_win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if _win_fonts.exists():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_win_fonts))
# アプリの設定(窓の配置、前回のフォルダ)を利用者のレジストリではなく一時フォルダの ini に向ける。
# テストで閉じた窓の大きさが実際の設定に残らないようにする(kaisetsu.gui.settings.app_settings)
os.environ.setdefault(
    "KAISETSU_SETTINGS_FILE",
    str(Path(tempfile.mkdtemp(prefix="kaisetsu-test-settings-")) / "kaisetsu.ini"),
)
