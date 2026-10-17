"""配布物に同梱する依存ライブラリのライセンス一覧(THIRD_PARTY_LICENSES.txt)を作る。

    uv run python packaging/collect_licenses.py dist/THIRD_PARTY_LICENSES.txt \\
        --manifest build/kaisetsu/bundle_manifest.json --bundle-dir dist/kaisetsu/_internal

- `--manifest` に PyInstaller の spec(packaging/kaisetsu.spec)が書き出した「実際に同梱した
  モジュール・バイナリ・データ」の一覧を渡すと、それを importlib.metadata で配布物(パッケージ)に
  対応づけて一覧にする。対応づけられないものや、開発用パッケージ(hypothesis、coverage、pytest、
  setuptools)が混ざっていれば終了コード 1 にしてビルドを失敗させる(外部レビュー 2026-10-17 #11)。
- `--bundle-dir` を渡すと、その中の *.dist-info とフォルダが一覧に載っているかも突き合わせる。
- 引数が無ければ kaisetsu[gui] の依存を再帰的にたどる(手元で中身を見るときとテスト用)。

各パッケージの dist-info に入っているライセンス全文に加えて、dist-info に無い本文を
packaging/licenses/ の固定ファイルから必ず入れる:
  LGPL-3.0.txt、GPL-3.0.txt   Qt / PySide6 / shiboken6(LGPL v3 第 4 条で両方の写しを添付する)
  Qt-third-party.txt          Qt の中の第三者ソフトウェア(harfbuzz、freetype、pcre2 など)
  Apache-2.0.txt              OpenSSL 3(libssl-3.dll / libcrypto-3.dll、Python 本体に付属)
Python 本体(PSF)と、PyInstaller のブートローダ(GPL にリンク例外付き。生成物には GPL の
制約は及ばない)の注記も付ける。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LICENSES_DIR = Path(__file__).resolve().parent / "licenses"
LICENSE_FILE_RE = re.compile(r"(?i)^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)")
SKIP_LICENSE_FILE_RE = re.compile(r"(?i)Qt-Commercial")
"""wheel に入っている Qt の商用ライセンス文書。kaisetsu には適用しない(LGPL v3 で使う)"""

DEV_PACKAGES = {"hypothesis", "coverage", "pytest", "pytest-cov", "pytest-qt", "setuptools", "mypy"}
"""配布物に入っていてはいけない開発用パッケージ(正規化した名前)"""

QT_PACKAGES = {
    "pyside6",
    "pyside6-essentials",
    "pyside6-addons",
    "pyside6-pdf",
    "pyside6-webengine",
    "shiboken6",
}

NOTES = {
    "pyside6": (
        "PySide6 / shiboken6 と Qt のライブラリは LGPL v3 で使っている(wheel の METADATA は "
        "LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only)。kaisetsu は Qt を動的リンク"
        "(別ファイルの DLL)で使っており、利用者は配布フォルダ内の Qt の DLL を同じバージョンの"
        "別ビルドに差し替えられる。Qt のソースコードは "
        "https://download.qt.io/official_releases/qt/ 、PySide6 のソースコードは "
        "https://code.qt.io/cgit/pyside/pyside-setup.git/ から入手できる。"
        "LGPL v3 と GPL v3 の本文、Qt の中の第三者ソフトウェアの一覧は下の「固定の本文」に載せた。"
    ),
    "numba": "numba と llvmlite は BSD-2。LLVM 本体は Apache-2.0 with LLVM exception。",
    "numpy": "numpy と scipy の wheel には OpenBLAS(BSD-3)が同梱されている。",
}

MODULE_ALIASES = {
    "pkg_resources": "setuptools",
    "_distutils_hack": "setuptools",
}
"""packages_distributions() で引けないトップレベル名 → 配布物名"""

PYINSTALLER_RUNTIME_RE = re.compile(r"^(pyimod\d+|pyi_rth_|_pyi_rth|pyiboot|pyi_splash|_pyi_)")
"""PyInstaller の実行時ブートストラップ(GPL + 例外。配布物には及ばない)"""

OWN_DATA_ROOTS = {"data", "docs"}
"""spec で同梱する kaisetsu 自身のデータ(サンプルデータ、説明書)"""

KNOWN_BINARIES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)^python3\d*\.dll$"), "Python 本体(PSF License。本文は上の Python の節)"),
    (re.compile(r"(?i)^base_library\.zip$"), "Python 標準ライブラリ(PSF License)"),
    (
        re.compile(r"(?i)^(libssl|libcrypto)-[\w.]+\.dll$"),
        "OpenSSL 3(Apache License 2.0。Python 本体に付属。本文は下の「固定の本文」)",
    ),
    (
        re.compile(r"(?i)^libffi-[\w.]+\.dll$"),
        "libffi(MIT。Python 本体に付属。本文は Python の LICENSE.txt の中)",
    ),
    (re.compile(r"(?i)^sqlite3\.dll$"), "SQLite(パブリックドメイン。Python 本体に付属)"),
    (re.compile(r"(?i)^libexpat[\w.-]*\.dll$"), "expat(MIT。Python 本体に付属)"),
    (
        re.compile(
            r"(?i)^((vcruntime|msvcp|concrt|vcomp|vcamp|vccorlib)\d+[\w]*\.dll|api-ms-win-[\w.-]+\.dll"
            r"|ucrtbase\.dll)$"
        ),
        "Microsoft Visual C++ 再頒布可能パッケージ(Microsoft のライセンス条項で再頒布可。"
        "Python の LICENSE.txt の Microsoft Distributable Code の節を参照)",
    ),
]
"""配布物の直下に置かれる、Python パッケージ由来でないファイル → 出どころとライセンス"""

OPENSSL_RE = re.compile(r"(?i)^(libssl|libcrypto)-[\w.]+\.dll$")


@dataclass
class Manifest:
    """spec が書き出した、実際に同梱したものの一覧。"""

    modules: list[str] = field(default_factory=list)
    """純 Python モジュールの完全修飾名(PYZ に入るもの)"""
    binaries: list[str] = field(default_factory=list)
    """拡張モジュール・DLL の配布物内の相対パス(区切りは /)"""
    datas: list[str] = field(default_factory=list)
    """データファイルの配布物内の相対パス(区切りは /)"""


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        modules=[str(x) for x in data.get("modules", [])],
        binaries=[str(x) for x in data.get("binaries", [])],
        datas=[str(x) for x in data.get("datas", [])],
    )


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


def _top_level_of_path(rel: str) -> str:
    """配布物内の相対パス → 先頭の要素(フォルダ名、直下のファイルならファイル名)。"""
    return rel.replace("\\", "/").split("/")[0]


def _module_name_of_file(filename: str) -> str:
    """直下のファイル名からモジュール名を取り出す("gemmi.cp314-win_amd64.pyd" → "gemmi")。"""
    return filename.split(".")[0]


@dataclass
class BundleResolution:
    """同梱物を配布物(パッケージ)に対応づけた結果。"""

    dists: dict[str, metadata.Distribution] = field(default_factory=dict)
    root_files: list[tuple[str, str]] = field(default_factory=list)
    """パッケージ由来でない直下のファイル(名前, 説明)"""
    unknown: list[str] = field(default_factory=list)
    """対応づけられなかったもの(あればビルドを失敗させる)"""
    dev_packages: list[str] = field(default_factory=list)
    """混ざっていた開発用パッケージ"""
    has_openssl: bool = False


def resolve_bundle(manifest: Manifest) -> BundleResolution:
    """manifest の各項目を配布物(パッケージ)に対応づける。"""
    pkg_dists = metadata.packages_distributions()
    stdlib = set(sys.stdlib_module_names)
    res = BundleResolution()
    seen_root: set[str] = set()

    def add_dist(name: str) -> None:
        key = canonicalize_name(name)
        if key in res.dists:
            return
        try:
            res.dists[key] = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            res.unknown.append(f"配布物 {name}(dist-info が見つからない)")

    def resolve_top_level(top: str, origin: str) -> bool:
        """トップレベルのモジュール名を配布物に対応づける。できなければ False。"""
        if top == "kaisetsu" or top in stdlib or PYINSTALLER_RUNTIME_RE.match(top):
            return True
        if top in MODULE_ALIASES:
            add_dist(MODULE_ALIASES[top])
            return True
        names = pkg_dists.get(top)
        if names:
            for n in names:
                add_dist(n)
            return True
        return False

    for mod in manifest.modules:
        top = mod.split(".")[0]
        if not resolve_top_level(top, mod):
            res.unknown.append(f"モジュール {mod}")

    for rel in manifest.binaries + manifest.datas:
        top = _top_level_of_path(rel)
        if "/" in rel.replace("\\", "/"):
            # パッケージのフォルダの中(numpy/..., PySide6/plugins/..., numpy.libs/...)
            base = top[:-5] if top.endswith(".libs") else top
            if base in OWN_DATA_ROOTS:
                continue
            if base.endswith(".dist-info"):
                add_dist(base[: -len(".dist-info")].rsplit("-", 1)[0])
                continue
            if not resolve_top_level(base, rel):
                res.unknown.append(f"フォルダ {top}/({rel})")
            continue
        # 配布物の直下のファイル
        if top in seen_root:
            continue
        seen_root.add(top)
        if OPENSSL_RE.match(top):
            res.has_openssl = True
        note = next((text for pat, text in KNOWN_BINARIES if pat.match(top)), None)
        if note is not None:
            res.root_files.append((top, note))
            continue
        modname = _module_name_of_file(top)
        if resolve_top_level(modname, top):
            continue
        res.unknown.append(f"ファイル {top}")

    res.dev_packages = sorted(k for k in res.dists if k in DEV_PACKAGES)
    return res


def check_bundle_dir(internal: Path, dists: dict[str, metadata.Distribution]) -> list[str]:
    """dist/kaisetsu/_internal の直下の *.dist-info とフォルダが一覧に載っているか突き合わせる。"""
    problems: list[str] = []
    pkg_dists = metadata.packages_distributions()
    stdlib = set(sys.stdlib_module_names)
    for entry in sorted(internal.iterdir()):
        name = entry.name
        if name.endswith(".dist-info"):
            dist_name = canonicalize_name(name[: -len(".dist-info")].rsplit("-", 1)[0])
            if dist_name not in dists:
                problems.append(f"{name} が一覧に無い")
            continue
        if not entry.is_dir():
            continue
        base = name[:-5] if name.endswith(".libs") else name
        if base in OWN_DATA_ROOTS or base == "kaisetsu" or base in stdlib:
            continue
        if base in DEV_PACKAGES or canonicalize_name(base) in DEV_PACKAGES:
            problems.append(f"開発用パッケージ {name} が同梱されている")
            continue
        names = pkg_dists.get(base) or ([MODULE_ALIASES[base]] if base in MODULE_ALIASES else [])
        if not names:
            problems.append(f"フォルダ {name} の出どころが分からない")
        elif not any(canonicalize_name(n) in dists for n in names):
            problems.append(f"フォルダ {name}({', '.join(names)})が一覧に無い")
    return problems


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
        if not LICENSE_FILE_RE.match(parts[-1]) or SKIP_LICENSE_FILE_RE.search(parts[-1]):
            continue
        path = Path(str(dist.locate_file(f)))
        if path.is_file():
            try:
                out.append((f.as_posix(), path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return out


def fixed_sections(
    dists: dict[str, metadata.Distribution], *, openssl: bool
) -> list[tuple[str, Path]]:
    """dist-info に無い本文(packaging/licenses/)のうち、この配布物に要るもの。"""
    out: list[tuple[str, Path]] = []
    if any(k in dists for k in QT_PACKAGES):
        out.append(
            (
                "Qt / PySide6 / shiboken6 — GNU Lesser General Public License v3",
                LICENSES_DIR / "LGPL-3.0.txt",
            )
        )
        out.append(
            (
                "GNU General Public License v3(LGPL v3 第 4 条により添付する写し)",
                LICENSES_DIR / "GPL-3.0.txt",
            )
        )
        out.append(("Qt の中で使われている第三者ソフトウェア", LICENSES_DIR / "Qt-third-party.txt"))
    if openssl:
        out.append(
            (
                "OpenSSL 3(libssl-3.dll / libcrypto-3.dll。Python 本体に付属)— Apache License 2.0",
                LICENSES_DIR / "Apache-2.0.txt",
            )
        )
    return out


def build_report(
    dists: dict[str, metadata.Distribution],
    *,
    root_files: list[tuple[str, str]] | None = None,
    openssl: bool = True,
    source_note: str,
) -> str:
    """一覧の本文を組み立てる。"""
    lines: list[str] = []
    lines.append("kaisetsu が同梱・利用しているサードパーティソフトウェアのライセンス一覧")
    lines.append("=" * 72)
    lines.append("")
    lines.append("kaisetsu 自体は MIT ライセンス(同梱の LICENSE.txt)。以下は配布物に含まれる")
    lines.append("Python パッケージとその依存の一覧と、各パッケージのライセンス全文。")
    lines.append(source_note)
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
    if root_files:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Python パッケージ以外で配布フォルダに入っているファイル")
        lines.append("-" * 72)
        for name, note in sorted(root_files):
            lines.append(f"{name:<28}{note}")
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
            if canonicalize_name(name) in QT_PACKAGES:
                lines.append("(LGPL v3。本文は下の「固定の本文」の LGPL-3.0 / GPL-3.0 を参照)")
            else:
                lines.append("(dist-info にライセンスファイルが無い。上記のライセンス種別を参照)")
        for fname, text in texts:
            lines.append(f"--- {fname} ---")
            lines.append(text.rstrip())
            lines.append("")
    lines.append("")
    lines.append("=" * 72)
    lines.append("固定の本文(dist-info に入っていないもの。packaging/licenses/ から)")
    lines.append("=" * 72)
    for title, path in fixed_sections(dists, openssl=openssl):
        lines.append("")
        lines.append(f"#### {title}")
        lines.append(f"--- {path.name} ---")
        lines.append("")
        lines.append(path.read_text(encoding="utf-8").rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    manifest_path: Path | None = None
    bundle_dir: Path | None = None
    positional: list[str] = []
    while args:
        a = args.pop(0)
        if a == "--manifest":
            manifest_path = Path(args.pop(0))
        elif a == "--bundle-dir":
            bundle_dir = Path(args.pop(0))
        else:
            positional.append(a)
    out_path = Path(positional[0]) if positional else Path("dist/THIRD_PARTY_LICENSES.txt")

    root_files: list[tuple[str, str]] | None = None
    openssl = True
    if manifest_path is not None:
        res = resolve_bundle(load_manifest(manifest_path))
        problems = [f"出どころが分からない: {u}" for u in res.unknown]
        problems += [f"開発用パッケージが同梱されている: {d}" for d in res.dev_packages]
        if bundle_dir is not None and bundle_dir.is_dir():
            problems += check_bundle_dir(bundle_dir, res.dists)
        if problems:
            print("ライセンス一覧と同梱物の突き合わせに失敗しました:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        dists = res.dists
        root_files = res.root_files
        openssl = res.has_openssl
        source_note = f"(一覧は PyInstaller が実際に同梱したものから作った: {manifest_path.name})"
    else:
        dists = dependency_closure("kaisetsu", {"gui"})
        source_note = "(一覧は kaisetsu[gui] の依存関係から作った)"
    dists.pop("kaisetsu", None)
    for key in DEV_PACKAGES:
        dists.pop(key, None)

    text = build_report(dists, root_files=root_files, openssl=openssl, source_note=source_note)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"{out_path}: {len(dists)} パッケージ")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
