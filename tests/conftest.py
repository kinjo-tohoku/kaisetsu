"""pytest 共通設定。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "public"
REFERENCE_DIR = ROOT / "validation" / "reference"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def pbso4_cif() -> Path:
    return DATA_DIR / "pbso4" / "PbSO4-Wyckoff.cif"


@pytest.fixture(scope="session")
def diamond_cif() -> Path:
    return DATA_DIR / "cif" / "diamond.cif"


def load_reference(name: str) -> dict:
    """GSAS-II 参照値 JSON を読む。無ければテストをスキップする。"""
    path = REFERENCE_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(
            f"参照値 {path.name} がありません。"
            "`uv run python validation/gsas2/run_gsas2.py` で生成してください"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)
