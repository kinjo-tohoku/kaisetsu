"""計算結果の書き出し(CSV)。

GUI の「計算パターンを CSV に」「反射リストを CSV に」とコマンドラインで共用する。

CSV は `csv` モジュールで書く(相の名前にカンマや引用符があっても列がずれない)。文字コードは
BOM 付き UTF-8(utf-8-sig)に統一する: 日本語の Excel は BOM が無いと UTF-8 を cp932 と
みなして文字化けし、配布物の既定の文字コード(cp932)では ₅ のような文字が書けない
(レビュー 2026-10-17 GUI #14)。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kaisetsu.refine.project import Project
    from kaisetsu.refine.residual import ResidualFeatures

CSV_ENCODING = "utf-8-sig"


def _write_rows(path: Path, header: Sequence[object], rows: Iterable[Sequence[object]]) -> Path:
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def _number_rows(cols: Sequence[np.ndarray]) -> list[list[str]]:
    """列ごとの配列を、1 行ずつ最短の往復表現(repr)の文字列にする。"""
    table = np.column_stack(cols)
    return [[repr(float(v)) for v in row] for row in table]


def write_pattern_csv(project: Project, hist: int, path: str | Path) -> Path:
    """観測・計算・背景・差と相ごとの寄与を CSV に書く。

    列: two_theta, y_obs, y_calc, y_bkg, diff, 相の名前…。

    TOF では 1 列目は飛行時間(μs)だが列名は two_theta のまま(GUI と同じ形式)。
    """
    h = project.histograms[hist]
    res = project.calculate(hist)
    cols = [res.x, h.data.y, res.y_calc, res.y_background, h.data.y - res.y_calc]
    header: list[object] = ["two_theta", "y_obs", "y_calc", "y_bkg", "diff"]
    for name, yp in zip(h.contributions, res.y_phases, strict=True):
        cols.append(yp)
        header.append(name)
    return _write_rows(Path(path), header, _number_rows(cols))


def write_pdf_csv(project: Project, dataset: int, path: str | Path) -> Path:
    """PDF の観測 G(r)・計算・差と相ごとの寄与を CSV に書く。"""
    g = project.pdf_datasets[dataset]
    res = project.calculate_pdf(dataset)
    cols = [res.r, g.data.g, res.g_calc, g.data.g - res.g_calc]
    header: list[object] = ["r", "G_obs", "G_calc", "diff"]
    for name, gp in zip(g.phases, res.g_phases, strict=True):
        cols.append(gp)
        header.append(name)
    return _write_rows(Path(path), header, _number_rows(cols))


def write_reflections_csv(project: Project, hist: int, path: str | Path) -> Path:
    """反射リスト(相、hkl、多重度、d、2θ、F²、強度、FWHM)を CSV に書く。"""
    header = [
        "phase", "h", "k", "l", "mult", "d", "two_theta", "F2_calc", "F2_obs", "I_calc", "I_obs",
        "fwhm",
    ]  # fmt: skip

    def rows() -> Iterable[list[object]]:
        for name in project.histograms[hist].contributions:
            rep = project.reflection_report(hist, name)
            for k in range(len(rep.hkl)):
                hh, kk, ll = rep.hkl[k]
                yield [
                    name, int(hh), int(kk), int(ll), int(rep.multiplicity[k]),
                    f"{rep.d[k]:.5f}", f"{rep.two_theta[k]:.4f}", f"{rep.f2_calc[k]:.4f}",
                    f"{rep.f2_obs[k]:.4f}", f"{rep.i_calc[k]:.5g}", f"{rep.i_obs[k]:.5g}",
                    f"{rep.fwhm[k]:.4f}",
                ]  # fmt: skip

    return _write_rows(Path(path), header, rows())


def write_unexplained_peaks_csv(
    project: Project,
    path: str | Path,
    hist: int | None = None,
    *,
    include_misfit: bool = True,
    features: Sequence[ResidualFeatures] | None = None,
) -> Path:
    """残差の点検で見つかった、相の反射で説明できないピークの一覧を CSV に書く。

    他のソフト(検索・同定プログラム)で第二相を探すための、観測位置・d 値・高さ・相対強度の
    一覧。列は `kaisetsu.refine.residual.UNEXPLAINED_COLUMNS`(hist, kind, position, unit, d,
    height, rel_height_pct, sigma, nearest_reflection, lo, hi)。kind は unindexed(相の反射で
    説明できない)か misfit(反射はあるが観測が計算より強い。include_misfit が偽なら省く)。

    Args:
        hist: 対象のヒストグラム(None なら全部)
        features: 既に調べた結果(GUI の表と同じものを書くとき)。None なら調べ直す
    """
    from kaisetsu.refine.residual import find_residual_features, format_unexplained_peaks

    if features is None:
        hists = range(len(project.histograms)) if hist is None else [hist]
        features = [find_residual_features(project, i) for i in hists]
    elif hist is not None:
        features = [f for f in features if f.hist == hist]
    # タブ区切りで作って列に分け、csv モジュールでクォートして書く(反射名にカンマがあっても
    # 列がずれない)
    text = format_unexplained_peaks(
        features, sep="\t", include_misfit=include_misfit, japanese=False
    )
    lines = [line.split("\t") for line in text.splitlines() if line]
    header: Sequence[object] = lines[0] if lines else []
    return _write_rows(Path(path), header, lines[1:])


__all__ = [
    "CSV_ENCODING",
    "write_pattern_csv",
    "write_pdf_csv",
    "write_reflections_csv",
    "write_unexplained_peaks_csv",
]
