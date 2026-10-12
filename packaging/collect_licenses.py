"""配布物に同梱する依存ライブラリのライセンス一覧(THIRD_PARTY_LICENSES.txt)を作る。

    uv run python packaging/collect_licenses.py dist/THIRD_PARTY_LICENSES.txt

kaisetsu[gui] の依存を importlib.metadata で再帰的にたどり、各パッケージのライセンス種別と
dist-info に入っているライセンス全文をまとめる。Python 本体(PSF)と、PyInstaller の
ブートローダ(GPL にリンク例外付き。生成物には GPL の制約は及ばない)の注記も付ける。
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LICENSE_FILE_RE = re.compile(r"(?i)^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)")

NOTES = {
    "pyside6": (
        "PySide6 / shiboken6 と Qt のライブラリは LGPL v3 で配布されている。"
        "kaisetsu は Qt を動的リンク(別ファイルの DLL)で使っており、利用者は配布フォルダ内の "
        "Qt の DLL を同じバージョンの別ビルドに差し替えられる。Qt のソースコードは "
        "https://download.qt.io/official_releases/qt/ から入手できる。"
    ),
    "numba": "numba と llvmlite は BSD-2。LLVM 本体は Apache-2.0 with LLVM exception。",
    "numpy": "numpy と scipy の wheel には OpenBLAS(BSD-3)が同梱されている。",
}


def dependency_closure(root: str, extras: set[str]) -> dict[str, metadata.Distribution]:
    """root パッケージ(とその extras)から到達できる配布物を集める。"""
    seen: dict[str, metadata.Distribution] = {}
    stack: list[tuple[str, set[str]]] = [(root, extras)]
    while stack:
        name, ex = stack.pop()
        key = canonicalize_name(name)
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[key] = dist
        for req_str in dist.requires or []:
            req = Requirement(req_str)
            if req.marker is not None:
                envs = [{"extra": e} for e in (ex or {""})]
                if not any(req.marker.evaluate(env) for env in envs):
                    continue
            stack.append((req.name, set(req.extras)))
    return seen


def license_kind(dist: metadata.Distribution) -> str:
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return str(expr)
    classifiers = [c for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "; ".join(c.split("::")[-1].strip() for c in classifiers)
    lic = md.get("License")
    if lic and len(str(lic)) < 80:
        return str(lic)
    return "(dist-info のライセンスファイルを参照)"


def homepage(dist: metadata.Distribution) -> str:
    md = dist.metadata
    for url in md.get_all("Project-URL") or []:
        label, _, link = url.partition(",")
        if label.strip().lower() in {"homepage", "source", "repository", "source code"}:
            return link.strip()
    return str(md.get("Home-page") or "")


def license_texts(dist: metadata.Distribution) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for f in dist.files or []:
        parts = f.parts
        if not any(p.endswith(".dist-info") for p in parts[:-1]):
            continue
        if not LICENSE_FILE_RE.match(parts[-1]):
            continue
        path = Path(str(dist.locate_file(f)))
        if path.is_file():
            try:
                out.append((f.as_posix(), path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return out


def main(argv: list[str]) -> int:
    out_path = Path(argv[1]) if len(argv) > 1 else Path("dist/THIRD_PARTY_LICENSES.txt")
    dists = dependency_closure("kaisetsu", {"gui"})
    dists.pop("kaisetsu", None)
    lines: list[str] = []
    lines.append("kaisetsu が同梱・利用しているサードパーティソフトウェアのライセンス一覧")
    lines.append("=" * 72)
    lines.append("")
    lines.append("kaisetsu 自体は MIT ライセンス(同梱の LICENSE.txt)。以下は配布物に含まれる")
    lines.append("Python パッケージとその依存の一覧と、各パッケージのライセンス全文。")
    lines.append("")
    lines.append(f"Python {sys.version.split()[0]} — PSF License(Python Software Foundation)")
    lines.append("")
    lines.append(
        "PyInstaller のブートローダ(kaisetsu.exe の起動部分)は GPL v2 にリンク例外が付いた"
        "ライセンスで、生成した実行ファイルには GPL の義務は及ばない。"
    )
    lines.append("")
    lines.append("サンプルデータ(data/public)の出典とライセンスは data/public/README.md を参照。")
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"{'パッケージ':<28}{'バージョン':<14}ライセンス")
    lines.append("-" * 72)
    ordered = sorted(dists.values(), key=lambda d: canonicalize_name(d.metadata["Name"]))
    for d in ordered:
        name = str(d.metadata["Name"])
        lines.append(f"{name:<28}{d.version:<14}{license_kind(d)}")
    lines.append("")
    for key, note in NOTES.items():
        if key in dists:
            lines.append(f"注: {note}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("ライセンス全文")
    lines.append("=" * 72)
    py_lic = Path(sys.base_prefix) / "LICENSE.txt"
    if py_lic.is_file():
        lines.append("")
        lines.append("#### Python (PSF License)")
        lines.append("")
        lines.append(py_lic.read_text(encoding="utf-8", errors="replace").rstrip())
    for d in ordered:
        name = str(d.metadata["Name"])
        texts = license_texts(d)
        lines.append("")
        lines.append(f"#### {name} {d.version} — {license_kind(d)}")
        hp = homepage(d)
        if hp:
            lines.append(hp)
        lines.append("")
        if not texts:
            lines.append("(dist-info にライセンスファイルが無い。上記のライセンス種別を参照)")
        for fname, text in texts:
            lines.append(f"--- {fname} ---")
            lines.append(text.rstrip())
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{out_path}: {len(ordered)} パッケージ")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
