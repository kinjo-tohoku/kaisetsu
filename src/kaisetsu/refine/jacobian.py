"""半解析的ヤコビアン: パラメータの種類ごとに安い微分を使う。

数値微分では変数 1 個につきパターン全体(プロファイル計算込み)を 1 回計算し直すが、
多くの変数はプロファイルの形を変えない:

- 背景係数 h{i}:bkg:k        ∂y/∂c_k = T_k(x)(Chebyshev 基底)。計算不要
- スケール h{i}:相:scale     ∂y/∂S = y_相 / S。計算不要
- 構造パラメータ 相:サイト:*  強度 I_k ∝ |F_k|² だけが変わる:
                               ∂y/∂p = Σ_k (∂|F_k|²/∂p / |F_k|²) c_k(x)
                               c_k は直近の計算で保存した反射 k の寄与(全波長の和)。
                               ∂|F|²/∂p は構造因子だけを中心差分で求める(プロファイル計算なし)
- 選択配向 h{i}:相:md_r      同様に PO 因子の比で
- それ以外(格子・装置・試料・サイズ・歪み・PDF)は従来どおり残差全体の数値微分

PDF データセットの行は、構造パラメータについても数値微分する(G(r) の計算は安い)。
拘束(tie)は、主パラメータを動かして従属パラメータを評価し直してから構造因子を計算するので
そのまま正しく入る。

出典: 線形パラメータの解析微分は Rietveld 法の標準的な実装(Young, Ch. 1.3)。
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from kaisetsu.pattern.calculator import PeakTable, PhaseContribution
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.profile.texture import coefficient_names
from kaisetsu.reflections.structure_factor import structure_factors

if TYPE_CHECKING:
    from kaisetsu.refine.parameters import Parameter
    from kaisetsu.refine.project import HistogramEntry, Project

FloatArray = npt.NDArray[np.float64]

_SITE_TAGS = frozenset({"x", "y", "z", "occ", "uiso", "u11", "u22", "u33", "u12", "u13", "u23"})
_HIST_RE = re.compile(r"^h(\d+):(.+)$")


def classify(project: Project, p: Parameter) -> tuple[str, int | None, str | None]:
    """パラメータの種類を返す: ("bkg"|"scale"|"intensity"|"po"|"other", ヒストグラム添字, 相名)。"""
    m = _HIST_RE.match(p.name)
    if m:
        i = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("bkg:"):
            return "bkg", i, None
        parts = rest.split(":")
        if len(parts) == 2 and parts[0] in project.phases:
            if parts[1] == "scale":
                return "scale", i, parts[0]
            if parts[1] == "md_r" or parts[1].startswith("sh"):
                return "po", i, parts[0]
        return "other", i, None
    parts = p.name.split(":")
    if len(parts) == 3 and parts[0] in project.phases and parts[1] != "cell":
        if parts[2] in _SITE_TAGS:
            return "intensity", None, parts[0]
    return "other", None, None


def _perturbed(project: Project, p: Parameter, h: float) -> None:
    """p を h だけ動かし、拘束を評価してモデルに書き込む。"""
    p.value += h
    project.params.evaluate_ties()
    project.params.apply_all()
    for ph in project.phases.values():
        ph.sync()


def _f2_for(project: Project, hist: HistogramEntry, phase: str) -> FloatArray:
    ph = project.phases[phase]
    ph.sync()
    res = hist.result
    assert res is not None
    k = list(hist.contributions).index(phase)
    tbl = res.peaks[k]
    f = structure_factors(
        ph.structure,
        tbl.hkl,
        hist.instrument.radiation,
        anomalous=hist.anomalous,
        override=hist.scattering_override,
    )
    return np.asarray(np.abs(f) ** 2)


def _extinction_for(
    hist: HistogramEntry, pc: PhaseContribution, tbl: PeakTable, f2: FloatArray
) -> FloatArray:
    """反射ごとの Sabine 消衰補正(CW は λ1 の位置、TOF はバンク 2θ と λ = 2 d sinθ)。"""
    inst = hist.instrument
    if isinstance(inst, TOFInstrument):
        lam = 2.0 * tbl.d * math.sin(math.radians(inst.two_theta / 2.0))
        return pc.extinction_factor(f2, lam, inst.two_theta, False)
    pos = np.where(np.isfinite(tbl.two_theta), tbl.two_theta, 90.0)
    return pc.extinction_factor(f2, inst.radiation.wavelength, pos, inst.radiation.is_xray)


def _assemble(hist: HistogramEntry, phase: str, ratio: FloatArray) -> FloatArray:
    """反射ごとの相対変化 ratio_k から ∂y/∂p(全点)を組み立てる。"""
    res = hist.result
    assert res is not None
    k = list(hist.contributions).index(phase)
    tbl = res.peaks[k]
    dy = np.zeros(len(res.x))
    for j in np.nonzero(np.isfinite(ratio) & (ratio != 0.0))[0]:
        a, b = int(tbl.window_start[j]), int(tbl.window_stop[j])
        if b > a and len(tbl.contributions[j]) == b - a:
            dy[a:b] += ratio[j] * tbl.contributions[j]
    return dy


def project_jacobian(project: Project, numerical_residual: FloatArray | None = None) -> FloatArray:
    """現在のパラメータでの残差ヤコビアン (N_resid, N_free)。

    呼ぶ前に `project.residuals()` が評価されていること(各ヒストグラムの result が現在の状態)。
    """
    params = project.params
    free = params.free()
    hists = project.histograms
    pdfs = project.pdf_datasets
    masks = [h.mask() for h in hists]
    pdf_masks = [g.mask() for g in pdfs]
    sw_h = []
    for h, m in zip(hists, masks, strict=True):
        assert h.data.weight is not None
        sw_h.append(np.sqrt(h.data.weight[m] * h.weight_factor))
    sw_g = [
        np.sqrt(g.data.weight[m] * g.weight_factor) for g, m in zip(pdfs, pdf_masks, strict=True)
    ]
    offsets_h: list[int] = []
    n = 0
    for m in masks:
        offsets_h.append(n)
        n += int(np.count_nonzero(m))
    offsets_g: list[int] = []
    for m in pdf_masks:
        offsets_g.append(n)
        n += int(np.count_nonzero(m))
    jac = np.zeros((n, len(free)))
    base = numerical_residual if numerical_residual is not None else project.residuals()
    for col, p in enumerate(free):
        kind, hi, phase = classify(project, p)
        if kind == "bkg" and hi is not None:
            h = hists[hi]
            k = int(p.name.rsplit(":", 1)[-1])
            basis = h.background.basis(h.data.x)[:, k]
            o = offsets_h[hi]
            jac[o : o + len(sw_h[hi]), col] = -sw_h[hi] * basis[masks[hi]]
            continue
        if kind == "scale" and hi is not None and phase is not None and p.value > 0:
            h = hists[hi]
            res = h.result
            assert res is not None
            k = list(h.contributions).index(phase)
            o = offsets_h[hi]
            jac[o : o + len(sw_h[hi]), col] = -sw_h[hi] * (res.y_phases[k][masks[hi]] / p.value)
            continue
        if kind in ("intensity", "po") and phase is not None:
            ph = project.phases[phase]
            if not ph.uses_f2_table:
                step = p.step
                targets = [(hi, hists[hi])] if hi is not None else list(enumerate(hists))
                done = True
                for i, h in targets:
                    if phase not in h.contributions or h.result is None:
                        continue
                    if kind == "intensity":
                        _perturbed(project, p, step)
                        f2_plus = _f2_for(project, h, phase)
                        _perturbed(project, p, -2.0 * step)
                        f2_minus = _f2_for(project, h, phase)
                        _perturbed(project, p, step)
                        tbl = h.result.peaks[list(h.contributions).index(phase)]
                        pc = h.contributions[phase]
                        if pc.extinction != 0.0:
                            # 消衰は |F|² に依存するので、強度 |F|² E(|F|²) ごと差分を取る
                            f2_plus = f2_plus * _extinction_for(h, pc, tbl, f2_plus)
                            f2_minus = f2_minus * _extinction_for(h, pc, tbl, f2_minus)
                            f2_base = tbl.f2 * _extinction_for(h, pc, tbl, tbl.f2)
                        else:
                            f2_base = tbl.f2
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio = np.where(
                                f2_base > 0, (f2_plus - f2_minus) / (2.0 * step * f2_base), 0.0
                            )
                    else:
                        pc = h.contributions[phase]
                        tbl = h.result.peaks[list(h.contributions).index(phase)]
                        pc.structure = ph.structure
                        if p.name.endswith(":md_r"):
                            po0 = pc.orientation_factor(tbl.hkl)
                            po1 = pc.orientation_factor(tbl.hkl, march_dollase_r=p.value + step)
                        else:
                            # 球面調和関数の係数: O は係数に線形なので差分が厳密
                            assert pc.sh_coefficients is not None
                            k_sh = coefficient_names(pc.texture_basis()).index(
                                p.name.split(":")[-1]
                            )
                            c1 = pc.sh_coefficients.copy()
                            c1[k_sh] += step
                            po0 = pc.orientation_factor(tbl.hkl)
                            po1 = pc.orientation_factor(tbl.hkl, sh_coefficients=c1)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio = np.where(po0 > 0, (po1 - po0) / (step * po0), 0.0)
                    dy = _assemble(h, phase, ratio)
                    o = offsets_h[i]
                    jac[o : o + len(sw_h[i]), col] = -sw_h[i] * dy[masks[i]]
                if pdfs and kind == "intensity":
                    # PDF の行は数値微分
                    _perturbed(project, p, step)
                    for gi, g in enumerate(pdfs):
                        gres = project.calculate_pdf(gi)
                        o = offsets_g[gi]
                        base_g = base[o : o + len(sw_g[gi])]
                        new_g = sw_g[gi] * (g.data.g[pdf_masks[gi]] - gres.g_calc[pdf_masks[gi]])
                        jac[o : o + len(sw_g[gi]), col] = (new_g - base_g) / step
                    _perturbed(project, p, -step)
                    for gi in range(len(pdfs)):
                        project.calculate_pdf(gi)
                if done:
                    continue
        # それ以外は数値微分(前進差分、上限に近ければ後退)
        step = p.step
        if math.isfinite(p.max) and p.value + step > p.max:
            step = -step
        _perturbed(project, p, step)
        r1 = project.residuals()
        _perturbed(project, p, -step)
        jac[:, col] = (r1 - base) / step
    # 数値微分で動かした結果を元に戻す(result を現在値で再計算)
    project.residuals()
    return jac
