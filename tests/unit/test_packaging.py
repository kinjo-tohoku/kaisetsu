"""配布の設定(packaging/)の回帰テスト(外部レビュー 2026-10-17 #11、#12、#14)。ネットワーク不要。

- collect_licenses.py の出力に LGPL-3.0 / GPL-3.0 / OpenSSL の本文が入る。
- 同梱物の一覧(manifest)から一覧を作るとき、未知のものや開発用パッケージがあれば失敗する。
- kaisetsu.spec が UTF-8 モードと開発用パッケージの除外を持つ。
- build_windows.ps1 / release.yml が dev グループ無しで同期する。
- kaisetsu.iss がアンインストールで {app} を丸ごと消さない。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging"


def _load_collect_licenses() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "collect_licenses", PACKAGING / "collect_licenses.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_licenses"] = mod  # dataclass(field(default_factory=...)) が参照する
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cl() -> ModuleType:
    pytest.importorskip("PySide6")
    return _load_collect_licenses()


def test_fixed_license_texts_exist() -> None:
    lic = PACKAGING / "licenses"
    lgpl = (lic / "LGPL-3.0.txt").read_text(encoding="utf-8")
    gpl = (lic / "GPL-3.0.txt").read_text(encoding="utf-8")
    apache = (lic / "Apache-2.0.txt").read_text(encoding="utf-8")
    qt = (lic / "Qt-third-party.txt").read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in lgpl and "Version 3, 29 June 2007" in lgpl
    assert "GNU GENERAL PUBLIC LICENSE" in gpl and "Version 3, 29 June 2007" in gpl
    assert gpl.count("\n") == 674 and len(gpl.encode()) == 35147  # 定型文そのもの
    assert lgpl.count("\n") == 165 and len(lgpl.encode()) == 7651
    assert "Apache License" in apache and "Version 2.0, January 2004" in apache
    for name in ("HarfBuzz", "FreeType", "PCRE2", "libpng", "zlib"):
        assert name in qt


def test_closure_report_contains_lgpl_gpl_and_openssl(cl: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "THIRD_PARTY_LICENSES.txt"
    assert cl.main(["collect_licenses.py", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text
    assert "GNU GENERAL PUBLIC LICENSE" in text and "GPL-3.0" in text
    assert "OpenSSL" in text and "Apache License" in text
    assert "HarfBuzz" in text and "FreeType" in text
    assert "PySide6" in text and "shiboken6" in text
    # 開発用パッケージは一覧(表)に載せない。Qt の商用ライセンス文書も載せない
    table = text.split("ライセンス全文", 1)[0]
    assert re.search(r"^(hypothesis|coverage|pytest|setuptools)\b", table, re.M) is None
    assert "Qt-Commercial" not in text


def _write_manifest(path: Path, **parts: list[str]) -> Path:
    data = {"modules": [], "binaries": [], "datas": []}
    data.update(parts)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_manifest_mode_lists_only_bundled_packages(cl: ModuleType, tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "m.json",
        modules=[
            "numpy._core.multiarray",
            "PySide6.QtCore",
            "shiboken6",
            "kaisetsu.api",
            "os",
            "pyimod02_importers",
        ],
        binaries=[
            "libssl-3.dll",
            "libcrypto-3.dll",
            "python314.dll",
            "VCRUNTIME140.dll",
            "numpy/_core/_multiarray_umath.pyd",
            "PySide6/Qt6Core.dll",
            "gemmi.cp314-win_amd64.pyd",
            "select.pyd",
        ],
        datas=["data/public/fap/FAP.XRA", "docs/user-guide/install.md", "base_library.zip"],
    )
    res = cl.resolve_bundle(cl.load_manifest(manifest))
    assert res.unknown == [] and res.dev_packages == [] and res.has_openssl
    assert {"numpy", "pyside6", "shiboken6", "gemmi"} <= set(res.dists)
    assert "matplotlib" not in res.dists  # 同梱していないものは載せない
    names = {n for n, _ in res.root_files}
    assert {"libssl-3.dll", "python314.dll", "VCRUNTIME140.dll", "base_library.zip"} <= names

    out = tmp_path / "out.txt"
    assert cl.main(["collect_licenses.py", str(out), "--manifest", str(manifest)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "OpenSSL" in text and "GNU LESSER GENERAL PUBLIC LICENSE" in text
    assert "libssl-3.dll" in text and "matplotlib" not in text


def test_manifest_mode_fails_on_unknown_and_dev_packages(cl: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    unknown = _write_manifest(tmp_path / "u.json", modules=["totally_unknown_module_xyz.sub"])
    assert cl.main(["collect_licenses.py", str(out), "--manifest", str(unknown)]) == 1
    assert not out.exists()
    unknown_file = _write_manifest(tmp_path / "f.json", binaries=["mystery-1.0.dll"])
    assert cl.main(["collect_licenses.py", str(out), "--manifest", str(unknown_file)]) == 1
    pytest.importorskip("hypothesis")
    dev = _write_manifest(tmp_path / "d.json", modules=["numpy", "hypothesis.core"])
    assert cl.main(["collect_licenses.py", str(out), "--manifest", str(dev)]) == 1


def test_bundle_dir_check_flags_missing_and_dev(cl: ModuleType, tmp_path: Path) -> None:
    internal = tmp_path / "_internal"
    (internal / "numpy").mkdir(parents=True)
    (internal / "hypothesis").mkdir()
    (internal / "data").mkdir()
    (internal / "scipy-1.0.0.dist-info").mkdir()
    problems = cl.check_bundle_dir(internal, {})
    assert any("hypothesis" in p for p in problems)
    assert any("numpy" in p for p in problems)
    assert any("scipy-1.0.0.dist-info" in p for p in problems)
    assert not any("data" in p for p in problems)


def test_spec_enables_utf8_mode_and_excludes_dev_packages() -> None:
    text = (PACKAGING / "kaisetsu.spec").read_text(encoding="utf-8")
    assert "X utf8_mode=1" in text
    # GUI と CLI の両方の EXE に options を付ける
    assert text.count("options=[UTF8_OPTION]") == 2
    tree = ast.parse(text)
    excludes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEV_EXCLUDES" for t in node.targets
        ):
            excludes = [ast.literal_eval(e) for e in node.value.elts]  # type: ignore[attr-defined]
    assert {"hypothesis", "coverage", "pytest", "setuptools"} <= set(excludes)
    # 起動スクリプトで環境変数を設定しても効かないので、していない
    for name in ("launch.py", "launch_cli.py"):
        assert 'setdefault("PYTHONUTF8"' not in (PACKAGING / name).read_text(encoding="utf-8")


def test_build_scripts_sync_without_dev_groups() -> None:
    sync = "uv sync --no-default-groups --extra gui --group build"
    ps1 = (PACKAGING / "build_windows.ps1").read_text(encoding="utf-8")
    assert sync in ps1
    assert "--manifest" in ps1 and "bundle_manifest.json" in ps1
    for line in ps1.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert re.search(r"uv run(?! --no-sync)", line) is None, (
            f"同期し直すと dev グループが戻る: {line}"
        )
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert sync in release
    assert "--draft" in release and "--draft=false" in release
    assert 'pytest -q -m "not slow"' in release
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert sync in ci


def test_iss_does_not_delete_whole_app_dir() -> None:
    iss = (PACKAGING / "kaisetsu.iss").read_text(encoding="utf-8-sig")
    section = iss.split("[UninstallDelete]", 1)[1]
    assert re.search(r'Name:\s*"\{app\}"\s*$', section, re.M) is None
    assert "DeleteInternalExceptData" in iss and "ssInstall" in iss


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の配布物の設定")
def test_spec_parses() -> None:
    ast.parse((PACKAGING / "kaisetsu.spec").read_text(encoding="utf-8"))
