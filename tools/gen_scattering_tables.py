"""cctbx のソースから散乱因子テーブルを生成する(開発時のみ使用)。

使い方:
    python -I tools/gen_scattering_tables.py <cctbx ファイルのあるディレクトリ> <出力ディレクトリ>

入力(cctbx_project, BSD-3-Clause, https://github.com/cctbx/cctbx_project):
    cctbx/eltbx/xray_scattering/it1992.cpp  … IT Vol. C Table 6.1.1.4 (Cromer-Mann 4 ガウス)
    cctbx/eltbx/xray_scattering/wk1995.cpp  … Waasmaier & Kirfel (1995) 5 ガウス
    cctbx/eltbx/neutron.cpp                 … Sears, Neutron News 3(3), 26-37 (1992) 散乱長

出力:
    it1992.py, wk1995.py, sears1992.py(純粋な Python 辞書)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _data_section(text: str) -> str:
    m = re.search(
        r"BEGIN_COMPILED_IN_REFERENCE_DATA(.*?)END_COMPILED_IN_REFERENCE_DATA", text, flags=re.S
    )
    if not m:
        raise SystemExit("BEGIN/END_COMPILED_IN_REFERENCE_DATA が見つかりません")
    return m.group(1)


def parse_gaussian_table(path: Path, n: int) -> list[tuple[str, list[float], list[float], float]]:
    text = _strip_comments(_data_section(path.read_text(encoding="utf-8")))
    pat = re.compile(
        r'\{\s*"([^"]+)"\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*(' + _NUM + r")\s*\}", re.S
    )
    out = []
    for label, a_s, b_s, c_s in pat.findall(text):
        a = [float(v) for v in re.findall(_NUM, a_s)]
        b = [float(v) for v in re.findall(_NUM, b_s)]
        if len(a) != n or len(b) != n:
            raise SystemExit(f"{path.name}: {label} の係数が {n} 個ではありません")
        out.append((label, a, b, float(c_s)))
    if not out:
        raise SystemExit(f"{path.name}: エントリを読めませんでした")
    return out


def parse_neutron_table(path: Path) -> list[tuple[str, float, float, float]]:
    text = _strip_comments(_data_section(path.read_text(encoding="utf-8")))
    pat = re.compile(
        r'\{\s*"([^"]+)"\s*,\s*(' + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\}"
    )
    out = [(lab, float(re_), float(im), float(ab)) for lab, re_, im, ab in pat.findall(text)]
    if not out:
        raise SystemExit(f"{path.name}: エントリを読めませんでした")
    return out


def _header(title: str, source: str, reference: str) -> str:
    return (
        f'"""{title}\n\n'
        "このファイルは tools/gen_scattering_tables.py により自動生成されたもの。"
        "手で編集しないこと。\n\n"
        f"データ出典: {reference}\n"
        f"取り込み元: cctbx_project ({source}), BSD-3-Clause License,\n"
        "    Copyright (c) 2006-2026 The Regents of the University of California,\n"
        "    through Lawrence Berkeley National Laboratory.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
    )


def write_gaussian_module(
    rows: list[tuple[str, list[float], list[float], float]],
    out: Path,
    name: str,
    title: str,
    source: str,
    reference: str,
) -> None:
    lines = [_header(title, source, reference)]
    lines.append(
        "# 値: ラベル -> ((a_1..a_n), (b_1..b_n), c)。\n"
        "# f0(s) = sum_i a_i exp(-b_i s^2) + c, s = sinθ/λ [Å^-1]\n"
    )
    lines.append(f"{name}: dict[str, tuple[tuple[float, ...], tuple[float, ...], float]] = {{\n")
    for label, a, b, c in rows:
        a_s = ", ".join(repr(v) for v in a)
        b_s = ", ".join(repr(v) for v in b)
        lines.append(f"    {label!r}: (({a_s}), ({b_s}), {c!r}),\n")
    lines.append("}\n")
    out.write_text("".join(lines), encoding="utf-8")


def write_neutron_module(rows: list[tuple[str, float, float, float]], out: Path) -> None:
    lines = [
        _header(
            "中性子の束縛コヒーレント散乱長と吸収断面積(Sears 1992)。",
            "cctbx/eltbx/neutron.cpp",
            "V. F. Sears, Neutron News 3(3), 26-37 (1992); NIST https://www.ncnr.nist.gov/resources/n-lengths/",
        )
    ]
    lines.append("# 値: ラベル -> (Re b_coh [fm], Im b_coh [fm], σ_abs [barn, 2200 m/s])\n")
    lines.append("SEARS1992: dict[str, tuple[float, float, float]] = {\n")
    for label, re_, im, ab in rows:
        lines.append(f"    {label!r}: ({re_!r}, {im!r}, {ab!r}),\n")
    lines.append("}\n")
    out.write_text("".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = Path(argv[1])
    dst = Path(argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    it = parse_gaussian_table(src / "it1992.cpp", 4)
    write_gaussian_module(
        it,
        dst / "it1992.py",
        "IT1992",
        "X線原子散乱因子 f0 の解析近似係数(Cromer-Mann 4 ガウス + 定数)。",
        "cctbx/eltbx/xray_scattering/it1992.cpp",
        "International Tables for Crystallography Vol. C (1992), Table 6.1.1.4, pp. 500-502",
    )
    wk = parse_gaussian_table(src / "wk1995.cpp", 5)
    write_gaussian_module(
        wk,
        dst / "wk1995.py",
        "WK1995",
        "X線原子散乱因子 f0 の解析近似係数(Waasmaier-Kirfel 5 ガウス + 定数、s ≤ 6 Å^-1)。",
        "cctbx/eltbx/xray_scattering/wk1995.cpp",
        "D. Waasmaier & A. Kirfel, Acta Cryst. A51, 416-431 (1995)",
    )
    ne = parse_neutron_table(src / "neutron.cpp")
    write_neutron_module(ne, dst / "sears1992.py")
    print(f"it1992: {len(it)} entries, wk1995: {len(wk)} entries, sears1992: {len(ne)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
