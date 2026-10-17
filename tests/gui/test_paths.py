"""ファイルダイアログの初期フォルダ: 配布物ではインストール先の中にしない(外部レビュー 2026-10-17 #12)。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QStandardPaths

from kaisetsu.gui import paths


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "test.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def frozen_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """PyInstaller の配布物のふり: app/kaisetsu.exe、app/_internal/data/public、ドキュメントは tmp の中。"""
    app = tmp_path / "app"
    sample = app / "_internal" / "data" / "public"
    (sample / "fap").mkdir(parents=True)
    (sample / "fap" / "FAP.XRA").write_text("data", encoding="utf-8")
    (sample / "README.md").write_text("samples", encoding="utf-8")
    (app / "kaisetsu.exe").write_bytes(b"")
    docs = tmp_path / "Documents" / "kaisetsu"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "kaisetsu.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(app / "_internal"), raising=False)
    monkeypatch.setenv("KAISETSU_SAMPLE_DIR", str(sample))
    monkeypatch.setattr(paths, "documents_dir", lambda: docs)
    return {"app": app, "sample": sample, "docs": docs}


def test_frozen_initial_dir_is_documents_with_sample_copy(
    frozen_app: dict[str, Path], settings: QSettings
) -> None:
    docs = frozen_app["docs"]
    assert paths.install_dir() == frozen_app["app"].resolve()
    assert paths.sample_data_dir() == frozen_app["sample"]
    assert paths.last_dir(settings) == str(docs)
    copied = docs / paths.SAMPLE_COPY_NAME / "fap" / "FAP.XRA"
    assert copied.read_text(encoding="utf-8") == "data"
    # 2 回目はコピー済みのファイルを触らない(利用者が書き換えていても戻さない)
    copied.write_text("edited", encoding="utf-8")
    assert paths.last_dir(settings) == str(docs)
    assert copied.read_text(encoding="utf-8") == "edited"


def test_frozen_ignores_saved_dir_inside_install_dir(
    frozen_app: dict[str, Path], settings: QSettings
) -> None:
    inside = frozen_app["sample"] / "fap"
    settings.setValue("last_dir", str(inside))  # v0.1.13 以前の設定に残っている値
    assert paths.is_inside_install_dir(inside)
    assert paths.last_dir(settings) == str(frozen_app["docs"])
    # サンプルを開いても、次回の初期フォルダをインストール先の中にしない
    paths.remember_dir(str(inside / "FAP.XRA"), settings)
    assert settings.value("last_dir") == str(inside)  # 古い値はそのまま(上書きしない)
    settings.remove("last_dir")
    paths.remember_dir(str(inside / "FAP.XRA"), settings)
    assert settings.value("last_dir") is None
    # インストール先の外なら覚える
    outside = frozen_app["docs"].parent / "work"
    outside.mkdir(parents=True)
    paths.remember_dir(str(outside / "a.json"), settings)
    assert paths.last_dir(settings) == str(outside)


def test_not_frozen_keeps_repository_sample_dir(
    settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("KAISETSU_SAMPLE_DIR", raising=False)
    assert paths.install_dir() is None
    assert not paths.is_inside_install_dir(Path(sys.executable))
    sample = paths.sample_data_dir()
    assert sample is not None and (sample / "fap" / "FAP.XRA").is_file()
    assert paths.last_dir(settings) == str(sample)


def test_documents_dir_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "Docs"
    base.mkdir()
    monkeypatch.setattr(QStandardPaths, "writableLocation", staticmethod(lambda _loc: str(base)))
    d = paths.documents_dir()
    assert d == base / "kaisetsu" and d.is_dir()
