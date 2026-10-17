"""テキスト形式のリーダーの端のケース(レビュー 2026-10-17 GUI #5, #17)。

- ヘッダ無しの 3 列 .xye(「10.000 1234 35.1」で始まる)を RIETAN の「点数 開始角 刻み」と誤判定しない
- BOM 付き UTF-8 で最初の 1 点を落とさない、UTF-16 も読める
- 小数点カンマのデータは日本語のメッセージで止める
- 未知のバイナリ・Excel(zip)を 2 点のデータとして読まない
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.io.readers import MIN_POINTS, read_powder_data, too_few_points_message
from kaisetsu.io.readers.rietan import is_rietan_block
from kaisetsu.io.readers.xy import decode_text, looks_binary

X = np.round(np.arange(10.0, 90.0001, 0.02), 4)
Y = np.round(100 + 900 * np.exp(-0.5 * ((X - 30) / 0.05) ** 2)).astype(int)
S = np.sqrt(Y)


def _write(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


def _xye(x: np.ndarray) -> str:
    return "".join(f"{a:.3f} {b} {c:.2f}\n" for a, b, c in zip(x, Y, S, strict=True))


def _xy(x: np.ndarray = X) -> str:
    return "".join(f"{a:.3f} {b}\n" for a, b in zip(x, Y, strict=True))


@pytest.mark.parametrize("start", [10.0, 20.0, 10.01])
def test_headerless_three_column_xye_is_not_rietan(tmp_path: Path, start: float) -> None:
    """1 行目が「10.000 1234 35.1」でも、2 行目以降が 3 列・総数が点数と違うので xy として読む。"""
    p = _write(tmp_path, "b.xye", _xye(X + (start - 10.0)))
    assert not is_rietan_block(p.read_text(encoding="utf-8"), complete=True)
    d = read_powder_data(p)
    assert d.metadata["format"] == "xy" and len(d) == len(X)
    assert d.x[0] == pytest.approx(start) and d.x[-1] == pytest.approx(start + 80.0)
    assert d.weight[0] == pytest.approx(1.0 / S[0] ** 2)


def test_rietan_block_still_detected(tmp_path: Path) -> None:
    text = (
        "* Cu3Fe4(PO4)6, CuKa1\n   12    10.000     0.010\n"
        "   9104   9375   9335   9396   9149   9450   9180   9274   9158   9382   9192\n"
        "   9115\n"
    )
    assert is_rietan_block(text) and is_rietan_block(text, complete=True)
    d = read_powder_data(_write(tmp_path, "a.int", text))
    assert d.metadata["format"] == "rietan" and len(d) == 12
    # 開始角が角度でない(1234)、2 行目以降の列数がばらつく、総数が点数より多い → 対象外
    assert not is_rietan_block("10 1234 35.1\n100 200 300\n100 200 300\n")
    assert not is_rietan_block("12 10.0 0.01\n1 2 3\n4 5\n6 7 8\n")
    assert not is_rietan_block("12 10.0 0.01\n" + "1 2 3 4 5 6\n" * 10)
    assert not is_rietan_block("12 10.0 0.01\n1 2 3\n4 5 6\n", complete=True)  # 6 点しかない


def test_bom_keeps_first_point(tmp_path: Path) -> None:
    d = read_powder_data(_write(tmp_path, "e.xy", "﻿" + _xy()))
    assert len(d) == len(X) and d.x[0] == pytest.approx(10.0)


def test_utf16_text_is_read(tmp_path: Path) -> None:
    p = _write(tmp_path, "u.txt", "2theta\tI\n" + _xy().replace(" ", "\t"), "utf-16")
    d = read_powder_data(p)
    assert len(d) == len(X) and d.x[0] == pytest.approx(10.0) and d.y[-1] == Y[-1]
    assert decode_text(p.read_bytes()).startswith("2theta")


@pytest.mark.parametrize("sep", ["\t", ";"])
def test_decimal_comma_is_reported(tmp_path: Path, sep: str) -> None:
    text = "".join(f"{a:.3f}{sep}{b:.1f}\n".replace(".", ",") for a, b in zip(X, Y, strict=True))
    with pytest.raises(ValueError, match="小数点がカンマ"):
        read_powder_data(_write(tmp_path, "i.txt", text))


def test_integer_csv_is_not_mistaken_for_decimal_comma(tmp_path: Path) -> None:
    text = "".join(f"{10 + i},{b}\n" for i, b in enumerate(Y[:200]))
    d = read_powder_data(_write(tmp_path, "int.csv", text))
    assert len(d) == 200 and d.x[0] == 10.0 and d.y[0] == Y[0]


def test_joined_scans_are_averaged(tmp_path: Path) -> None:
    d = read_powder_data(_write(tmp_path, "p.xy", _xy() * 3))
    assert len(d) == len(X)


def test_binary_and_excel_are_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    counts = (100 + 1000 * rng.random(4000)).astype("<f4").tobytes()
    unknown = tmp_path / "unknown.raw"
    unknown.write_bytes(b"RINT2000 RAW FILE".ljust(512, b"\0") + counts)
    assert looks_binary(unknown.read_bytes()[:4000])
    with pytest.raises(ValueError, match="バイナリ"):
        read_powder_data(unknown)
    mixed = tmp_path / "mixed.raw"
    mixed.write_bytes(b"10.00 0.02 90.00\r\n12 34\r\n" + counts)  # ASCII が 2 行あるだけ
    with pytest.raises(ValueError, match="バイナリ"):
        read_powder_data(mixed)
    xlsx = tmp_path / "data.csv"
    with zipfile.ZipFile(xlsx, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", "<sheetData/>")
    with pytest.raises(ValueError, match="Excel"):
        read_powder_data(xlsx)
    # UTF-16 は NUL を含むがバイナリではない
    assert not looks_binary("10.0 1\n".encode("utf-16"))


def test_too_few_points_message(tmp_path: Path) -> None:
    d = read_powder_data(_write(tmp_path, "t.xy", "10.0 100\n10.1 0\n10.2 50\n"))
    msg = too_few_points_message(d, "t.xy")
    assert msg is not None and "3 点" in msg and len(d) < MIN_POINTS
    assert too_few_points_message(read_powder_data(_write(tmp_path, "a.xy", _xy())), "a") is None
