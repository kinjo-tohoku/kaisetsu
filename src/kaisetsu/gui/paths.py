"""ファイルダイアログの初期フォルダ。

配布物(PyInstaller)では、インストール先(``_internal`` の中のサンプルデータなど)を初期フォルダに
しない。インストール先は更新で書き換わり、アンインストールで消えるので、そこに保存したプロジェクトは
失われる(外部レビュー 2026-10-17 #12)。初期フォルダは「ドキュメント\\kaisetsu」(無ければ作る)で、
同梱のサンプルデータは初回にその中の「サンプルデータ」へコピーする(初心者が見つけられるように)。
ソースから動かしているときは従来どおりリポジトリの data/public。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths

from kaisetsu.gui.settings import app_settings

SAMPLE_COPY_NAME = "サンプルデータ"
"""ドキュメント\\kaisetsu の中に作るサンプルデータのコピーのフォルダ名"""
SETTING_LAST_DIR = "last_dir"


def is_frozen() -> bool:
    """PyInstaller の配布物として動いているか。"""
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path | None:
    """配布物のインストール先(kaisetsu.exe のあるフォルダ)。Python 環境なら None。"""
    if not is_frozen():
        return None
    return Path(sys.executable).resolve().parent


def is_inside_install_dir(path: str | Path) -> bool:
    """path が配布物のインストール先の中か(Python 環境では常に False)。"""
    base = install_dir()
    if base is None:
        return False
    try:
        Path(path).resolve().relative_to(base)
    except (ValueError, OSError):
        return False
    return True


def documents_dir() -> Path:
    """保存の初期フォルダ「ドキュメント\\kaisetsu」。無ければ作る(作れなければドキュメント、ホーム)。"""
    loc = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    base = Path(loc) if loc else Path.home()
    target = base / "kaisetsu"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        return base if base.is_dir() else Path.home()
    return target


def sample_data_dir() -> Path | None:
    """サンプルデータ data/public の場所(無ければ None)。

    配布物では起動スクリプトが環境変数 ``KAISETSU_SAMPLE_DIR`` に同梱先を入れる(無ければ
    PyInstaller の展開先 ``_internal/data/public``)。ソースから動かしているときはリポジトリの
    data/public。
    """
    candidates: list[Path] = []
    env = os.environ.get("KAISETSU_SAMPLE_DIR")
    if env:
        candidates.append(Path(env))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "data" / "public")
    if not is_frozen():
        candidates.extend(p / "data" / "public" for p in Path(__file__).resolve().parents)
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def copy_samples(src: Path, dest: Path) -> Path:
    """サンプルデータを dest にコピーする。すでにあるファイルは触らない(利用者の保存物を守る)。"""
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        out = dest / f.relative_to(src)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    return dest


def initial_dir() -> Path:
    """前回のフォルダが無いときの初期フォルダ。

    配布物: ドキュメント\\kaisetsu(同梱のサンプルデータを「サンプルデータ」にコピーしておく)。
    ソースから実行: リポジトリの data/public。それも無ければ(``uv tool`` で入れたとき)
    ドキュメント\\kaisetsu。
    """
    sample = sample_data_dir()
    if not is_frozen() and sample is not None:
        return sample
    docs = documents_dir()
    if sample is not None:
        try:
            copy_samples(sample, docs / SAMPLE_COPY_NAME)
        except OSError:
            pass  # コピーできなくてもダイアログは出せる
    return docs


def last_dir(settings: QSettings | None = None) -> str:
    """前回使ったフォルダ。無いとき(配布物のインストール先の中だったときも)は initial_dir()。"""
    settings = settings or app_settings()
    saved = settings.value(SETTING_LAST_DIR)
    if saved and Path(str(saved)).is_dir() and not is_inside_install_dir(str(saved)):
        return str(saved)
    return str(initial_dir())


def remember_dir(path: str, settings: QSettings | None = None) -> None:
    """path のフォルダを次回の初期フォルダにする。配布物のインストール先の中なら覚えない。"""
    folder = Path(path).parent
    if is_inside_install_dir(folder):
        return
    settings = settings or app_settings()
    settings.setValue(SETTING_LAST_DIR, str(folder))


__all__ = [
    "SAMPLE_COPY_NAME",
    "copy_samples",
    "documents_dir",
    "initial_dir",
    "install_dir",
    "is_frozen",
    "is_inside_install_dir",
    "last_dir",
    "remember_dir",
    "sample_data_dir",
]
