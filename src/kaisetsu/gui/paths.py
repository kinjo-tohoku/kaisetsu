"""ファイルダイアログの初期フォルダ。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings


def sample_data_dir() -> Path | None:
    """リポジトリから動かしているときの data/public(無ければ None)。"""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "public"
        if cand.is_dir():
            return cand
    return None


def last_dir(settings: QSettings | None = None) -> str:
    """前回使ったフォルダ。初回はサンプルデータのフォルダ、無ければホーム。"""
    settings = settings or QSettings("kaisetsu", "kaisetsu")
    saved = settings.value("last_dir")
    if saved and Path(str(saved)).is_dir():
        return str(saved)
    sample = sample_data_dir()
    return str(sample if sample is not None else Path.home())


def remember_dir(path: str, settings: QSettings | None = None) -> None:
    settings = settings or QSettings("kaisetsu", "kaisetsu")
    settings.setValue("last_dir", str(Path(path).parent))
