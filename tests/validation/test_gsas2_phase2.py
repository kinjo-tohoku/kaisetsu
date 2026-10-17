"""Phase 2 の GSAS-II 突き合わせ: 計算パターン y_calc の点ごとの比較。

GSAS-II が 0 サイクルで出力した ycalc と、同じ装置・試料・相パラメータで kaisetsu が
計算したパターンを比べる。散乱因子(f', f'', b)は GSAS-II の値を注入し、
比較の対象をプロファイル・強度補正・ピーク位置に絞る(テーブルの差は Phase 1 で把握済み)。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.crystal.atoms import normalize_species
from kaisetsu.pattern import CWInstrument, PhaseContribution, SampleGeometry, calculate_pattern
from kaisetsu.profile import ChebyshevBackground
from tests.conftest import load_reference
from tests.validation.test_gsas2_phase1 import _anomalous_from_ref, _structure_with_gsas_adps

CASES = [
    "pbso4_xray",
    "pbso4_neutron",
    "diamond_xray_sim",
    "pbso4_xray_asym",
    "pbso4_neutron_abs",
    "pbso4_xray_lgmix",  # サイズ・歪みのローレンツ割合 0.5(ガウス成分の換算。物理 #2)
]

pytestmark = pytest.mark.validation


def _setup(ref: dict):
    x = np.asarray(ref["pattern"]["x"], dtype=float)
    inst = CWInstrument.from_gsas2(ref["instrument"])
    sample = SampleGeometry.from_gsas2(ref["sample"])
    st = _structure_with_gsas_adps(ref)
    bk = ref["background"][0]
    bkg = ChebyshevBackground(bk[3 : 3 + int(bk[2])], float(x[0]), float(x[-1]))
    phase = PhaseContribution.from_gsas2(
        st, ref["phase_hist"], hist_scale=float(ref["sample"]["Scale"][0])
    )
    kwargs: dict = {}
    if inst.radiation.is_neutron:
        phase.scale /= 100.0  # GSAS-II の中性子 F は 10^-12 cm 単位
        kwargs["scattering_override"] = {
            normalize_species(el): complex(bl["SL"][0] * 10.0, bl["SL"][1] * 10.0)
            for el, bl in ref["neutron_blens"].items()
        }
    else:
        kwargs["anomalous"] = {
            normalize_species(el): _anomalous_from_ref(ref, el)
            for el in ref["reflections"]["FF"]["El"]
        }
    d_min = float(np.asarray(ref["reflections"]["data"])[:, 4].min()) * (1 - 1e-9)
    return x, inst, sample, bkg, phase, d_min, kwargs


@pytest.mark.parametrize("case", CASES)
def test_pattern_matches_gsas2(case: str) -> None:
    ref = load_reference(case)
    x, inst, sample, bkg, phase, d_min, kwargs = _setup(ref)
    res = calculate_pattern(
        x, [phase], inst, sample, bkg, d_min=d_min, kalpha2_mode="gsas", fcj_method="gsas", **kwargs
    )
    y_ref = np.asarray(ref["pattern"]["ycalc"], dtype=float)
    b_ref = np.asarray(ref["pattern"]["background"], dtype=float)
    assert np.allclose(res.y_background, b_ref)

    bragg_ref = y_ref - b_ref
    bragg = res.y_bragg
    peak = bragg_ref > 0.01 * bragg_ref.max()
    rel = np.abs(bragg - bragg_ref)[peak] / bragg_ref[peak]
    scaled = np.abs(bragg - bragg_ref) / bragg_ref.max()
    area_ratio = float(np.sum(bragg) / np.sum(bragg_ref))
    print(
        f"[{case}] Bragg 成分の相対差: 中央値 {np.median(rel):.2e}, 最大 {rel.max():.2e}; "
        f"最大|Δ|/max {scaled.max():.2e}; 面積比 {area_ratio:.6f}"
    )
    assert np.median(rel) < 1e-4
    assert scaled.max() < 1e-3
    assert area_ratio == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("case", CASES)
def test_peak_positions_widths_intensities_match_gsas2(case: str) -> None:
    """反射ごとのピーク位置・幅・積分強度(GSAS-II の pos, sig, gam, Icorr·Fcalc2)を比べる。"""
    from kaisetsu.profile.pseudo_voigt import SQRT8LN2
    from kaisetsu.reflections import canonical_hkl

    ref = load_reference(case)
    x, inst, sample, bkg, phase, d_min, kwargs = _setup(ref)
    res = calculate_pattern(
        x, [phase], inst, sample, bkg, d_min=d_min, kalpha2_mode="gsas", fcj_method="gsas", **kwargs
    )
    tbl = res.peaks[0]
    cols = ref["reflections"]["columns"]
    data = np.asarray(ref["reflections"]["data"], dtype=float)
    g_hkl = canonical_hkl(phase.structure.spacegroup, data[:, :3].astype(int))
    index = {tuple(h): i for i, h in enumerate(tbl.hkl)}
    # GSAS-II のプロファイルは centideg あたりで規格化 → 積分強度は 1/100。
    # 中性子は F の単位(10^-12 cm vs fm)の 100 倍と scale の 1/100 が相殺する。
    unit = 100.0
    for row, h in zip(data, g_hkl, strict=True):
        i = index[tuple(h)]
        pos_g = row[cols.index("pos")]
        sig_g = row[cols.index("sig")]  # centideg^2 (σ^2)
        gam_g = row[cols.index("gam")]  # centideg (FWHM)
        assert tbl.two_theta[i] == pytest.approx(pos_g, abs=1e-6), f"位置 {h}"
        assert tbl.fwhm_gauss[i] == pytest.approx(SQRT8LN2 * np.sqrt(sig_g) / 100.0, rel=1e-6), (
            f"Γ_G {h}"
        )
        # GSAS-II は γ に下限 0.001 centideg を置く。それ以外は一致するはず。
        assert tbl.fwhm_lorentz[i] == pytest.approx(gam_g / 100.0, rel=1e-3, abs=2e-5), f"Γ_L {h}"
        i_g = row[cols.index("Icorr")] * row[cols.index("Fcalc2")] / unit
        if i_g > 1e-9 * np.max(data[:, cols.index("Fcalc2")]):
            assert tbl.intensity[i] == pytest.approx(i_g, rel=1e-6), f"強度 {h}"
