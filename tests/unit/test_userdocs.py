"""同梱の説明書(docs/user-guide)の置き場所の解決と Markdown の見出し取り出し。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaisetsu.userdocs import DOCS, doc_title, read_doc, section_headings, user_guide_dir


def test_repo_docs_are_found_and_complete() -> None:
    d = user_guide_dir()
    assert d is not None
    for name, _title in DOCS:
        assert (d / name).is_file(), name
    text = read_doc("getting-started.md")
    assert doc_title(text).startswith("はじめに")
    heads = section_headings(text)
    assert any(h.startswith("1.") for h in heads)
    assert any("困ったとき" in h for h in heads)


def test_env_override_and_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "getting-started.md").write_text(
        "# 題名\n\n## A\n\n```\n## コードの中は見出しではない\n```\n\n## B\n", encoding="utf-8"
    )
    monkeypatch.setenv("KAISETSU_DOCS_DIR", str(tmp_path))
    assert user_guide_dir() == tmp_path
    text = read_doc("getting-started.md")
    assert doc_title(text) == "題名"
    assert section_headings(text) == ["A", "B"]
    with pytest.raises(FileNotFoundError):
        read_doc("nothing.md")


def test_missing_env_dir_falls_back_to_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KAISETSU_DOCS_DIR", str(tmp_path / "nowhere"))
    d = user_guide_dir()
    assert d is not None and d != tmp_path / "nowhere"
    assert (d / "install.md").is_file()


def test_doc_title_fallback() -> None:
    assert doc_title("no heading here", "x.md") == "x.md"
