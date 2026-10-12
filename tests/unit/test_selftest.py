"""配布物の起動確認で使う自己診断(kaisetsu.selftest)。"""

from __future__ import annotations

from pathlib import Path

from kaisetsu.selftest import quick_check, sample_dir


def test_sample_dir_found_in_repository(data_dir: Path) -> None:
    d = sample_dir()
    assert d is not None and d.resolve() == data_dir.resolve()


def test_sample_dir_env_override(tmp_path: Path, monkeypatch) -> None:
    # 環境変数が指す場所に FAP.XRA があればそちらを優先する
    (tmp_path / "fap").mkdir()
    (tmp_path / "fap" / "FAP.XRA").write_text("", encoding="utf-8")
    monkeypatch.setenv("KAISETSU_SAMPLE_DIR", str(tmp_path))
    assert sample_dir() == tmp_path
    # 無いところを指していたらリポジトリのものに戻る
    monkeypatch.setenv("KAISETSU_SAMPLE_DIR", str(tmp_path / "nowhere"))
    d = sample_dir()
    assert d is not None and (d / "fap" / "fluorapatite_COD9001232.cif").is_file()


def test_quick_check_computes_pattern() -> None:
    report = quick_check()
    assert report.startswith("numpy ")
    assert "FAP pattern: Rwp" in report
