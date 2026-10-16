"""計算結果の書き出し(CSV)。

GUI の「計算パターンを CSV に」「反射リストを CSV に」とコマンドラインで共用する。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kaisetsu.refine.project import Project
    from kaisetsu.refine.residual import ResidualFeatures


def write_pattern_csv(project: Project, hist: int, path: str | Path) -> Path:
    """観測・計算・背景・差と相ごとの寄与を CSV に書く。

    列: two_theta, y_obs, y_calc, y_bkg, diff, 相の名前…。

    TOF では 1 列目は飛行時間(μs)だが列名は two_theta のまま(GUI と同じ形式)。
    """
    h = project.histograms[hist]
    res = project.calculate(hist)
    cols = [res.x, h.data.y, res.y_calc, res.y_background, h.data.y - res.y_calc]
    header = "two_theta,y_obs,y_calc,y_bkg,diff"
    for name, yp in zip(h.contributions, res.y_phases, strict=True):
        cols.append(yp)
        header += f",{name}"
    p = Path(path)
    np.savetxt(p, np.column_stack(cols), delimiter=",", header=header, comments="")
    return p


def write_pdf_csv(project: Project, dataset: int, path: str | Path) -> Path:
    """PDF の観測 G(r)・計算・差と相ごとの寄与を CSV に書く。"""
    g = project.pdf_datasets[dataset]
    res = project.calculate_pdf(dataset)
    cols = [res.r, g.data.g, res.g_calc, g.data.g - res.g_calc]
    header = "r,G_obs,G_calc,diff"
    for name, gp in zip(g.phases, res.g_phases, strict=True):
        cols.append(gp)
        header += f",{name}"
    p = Path(path)
    np.savetxt(p, np.column_stack(cols), delimiter=",", header=header, comments="")
    return p


def write_reflections_csv(project: Project, hist: int, path: str | Path) -> Path:
    """反射リスト(相、hkl、多重度、d、2θ、F²、強度、FWHM)を CSV に書く。"""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        f.write("phase,h,k,l,mult,d,two_theta,F2_calc,F2_obs,I_calc,I_obs,fwhm\n")
        for name in project.histograms[hist].contributions:
            rep = project.reflection_report(hist, name)
            for k in range(len(rep.hkl)):
                hh, kk, ll = rep.hkl[k]
                f.write(
                    f"{name},{hh},{kk},{ll},{rep.multiplicity[k]},{rep.d[k]:.5f},"
                    f"{rep.two_theta[k]:.4f},{rep.f2_calc[k]:.4f},{rep.f2_obs[k]:.4f},"
                    f"{rep.i_calc[k]:.5g},{rep.i_obs[k]:.5g},{rep.fwhm[k]:.4f}\n"
                )
    return p


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
    p = Path(path)
    p.write_text(
        format_unexplained_peaks(features, sep=",", include_misfit=include_misfit, japanese=False),
        encoding="utf-8",
        newline="\n",
    )
    return p
