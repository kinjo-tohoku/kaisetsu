"""利用者向けの説明書(docs/user-guide の Markdown)の置き場所を探す。

配布物(PyInstaller)では起動スクリプトが環境変数 ``KAISETSU_DOCS_DIR`` に同梱先を入れる。
ソースから動かすときはリポジトリの ``docs/user-guide`` を使う。どちらも無いとき(``uv tool`` で
wheel から入れたときなど)は GitHub 上の説明書を案内する。Qt には依存しない。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_URL = "https://github.com/kinjo-tohoku/kaisetsu"
GUIDE_URL = f"{REPO_URL}/blob/master/docs/user-guide/getting-started.md"
"""GitHub 上の説明書(ローカルに無いときの行き先)"""

DOCS: tuple[tuple[str, str], ...] = (
    ("getting-started.md", "はじめに(使い方)"),
    ("install.md", "インストールと更新"),
)
"""説明書のファイル名と一覧に出す題名(並び順)"""


def user_guide_dir() -> Path | None:
    """説明書のフォルダ。見つからなければ None。

    探す順: 環境変数 ``KAISETSU_DOCS_DIR`` → PyInstaller の展開先 ``docs/user-guide`` →
    リポジトリの ``docs/user-guide``(このファイルから 2 つ上)。
    """
    candidates: list[Path] = []
    env = os.environ.get("KAISETSU_DOCS_DIR")
    if env:
        candidates.append(Path(env))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "docs" / "user-guide")
    candidates.append(Path(__file__).resolve().parents[2] / "docs" / "user-guide")
    for c in candidates:
        if (c / DOCS[0][0]).is_file():
            return c
    return None


def read_doc(name: str, directory: Path | None = None) -> str:
    """説明書 1 本の Markdown を返す。無ければ FileNotFoundError。"""
    d = directory if directory is not None else user_guide_dir()
    if d is None:
        raise FileNotFoundError(name)
    return (d / name).read_text(encoding="utf-8")


def doc_title(markdown: str, fallback: str = "") -> str:
    """先頭の見出し(``# ...``)を題名として返す。"""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def section_headings(markdown: str) -> list[str]:
    """目次用に ``## `` の見出しを順に返す(コードブロックの中は無視)。"""
    out: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.startswith("## "):
            out.append(line[3:].strip())
    return out


__all__ = [
    "DOCS",
    "GUIDE_URL",
    "REPO_URL",
    "doc_title",
    "read_doc",
    "section_headings",
    "user_guide_dir",
]
