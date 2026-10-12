"""`kaisetsu-gui --smoke-test 結果ファイル`(配布物の起動確認)がウィンドウを出して自己診断を書き、終了すること。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from kaisetsu import __version__
from kaisetsu.gui import app as gui_app

pytestmark = pytest.mark.slow


def test_smoke_test_flag_writes_report_and_exits(qapp, tmp_path: Path) -> None:
    out = tmp_path / "smoke.txt"
    code = gui_app.main(["--smoke-test", str(out)])
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"ok {__version__}"
    assert any(line.startswith("FAP pattern: Rwp") for line in lines)


def test_smoke_test_flag_without_path(qapp) -> None:
    # パス無しでも終了コードで結果が分かる
    assert gui_app.main(["--smoke-test"]) == 0
