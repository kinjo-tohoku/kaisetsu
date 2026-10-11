"""TOF 中性子(POWGEN、NAC)の GSAS-II 突き合わせ: 計算パターンの比較。

参照値は `uv run python validation/gsas2/run_gsas2.py nac_tof`(精密化なし、0 サイクルの ycalc)。
GSAS-II の TOF プロファイル(EPSVOIGT)と同じ定式化なので、位置・幅・形が一致するはず。
スケール因子の定義(入射スペクトルや Lorentz 因子の扱い)は違うので、全体の比率は最小二乗で合わせて
比べ、その比率を記録する。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.pattern import PhaseContribution
from kaisetsu.pattern.tof import TOFInstrument, calculate_pattern_tof
from kaisetsu.profile import ChebyshevBackground
from tests.conftest import load_reference
from tests.validation.test_gsas2_phase1 import _structure_with_gsas_adps

pytestmark = pytest.mark.validation


def test_tof_pattern_matches_gsas2() -> None:
    ref = load_reference("nac_tof")
    x = np.asarray(ref["pattern"]["x"], dtype=float)
    inst = TOFInstrument.from_gsas2(ref["instrument"])
    assert inst.difc > 20000
    st = _structure_with_gsas_adps(ref)
    bk = ref["background"][0]
    bkg = ChebyshevBackground(bk[3 : 3 + int(bk[2])], float(x[0]), float(x[-1]))
    phase = PhaseContribution.from_gsas2(
        st, ref["phase_hist"], hist_scale=float(ref["sample"]["Scale"][0])
    )
    # GSAS-II の散乱長表(atmdata)はイオン名では引けず参照に入らないので、こちらの Sears 表を使う
    # (同じ出典。Ca 4.70, Al 3.449, Na 3.63, F 5.654 fm)
    override = None
    refl = np.asarray(ref["reflections"]["data"], dtype=float)
    d_min = float(refl[:, 4].min()) * (1 - 1e-9)
    d_max = float(refl[:, 4].max()) * (1 + 1e-9)
    res = calculate_pattern_tof(
        x, [phase], inst, bkg, d_min=d_min, d_max=d_max, scattering_override=override
    )
    y_ref = np.asarray(ref["pattern"]["ycalc"], dtype=float)
    b_ref = np.asarray(ref["pattern"]["background"], dtype=float)
    # GSAS-II は Limits の外を 0 にするので、中だけ比べる
    inside = b_ref != 0.0
    assert inside.sum() > 0.8 * len(x)
    assert np.allclose(res.y_background[inside], b_ref[inside], rtol=1e-6, atol=1e-6)
    bragg_ref = (y_ref - b_ref)[inside]
    bragg = res.y_bragg[inside]
    # ピーク位置: GSAS-II の反射リストの pos と一致
    pos_ref = refl[:, 5]
    tbl = res.peaks[0]
    order = np.argsort(tbl.d)
    pos_ours = np.interp(refl[:, 4], tbl.d[order], tbl.two_theta[order])
    assert np.allclose(pos_ours, pos_ref, atol=0.05)
    # 孤立した強い反射の積分強度は 2% 以内で一致(Lorentz 因子 d⁴ sinθ とスケールの定義が同じ)
    full_bragg = res.y_bragg
    full_ref = y_ref - b_ref
    ratios = []
    for k in range(len(tbl.hkl)):
        a, b = int(tbl.window_start[k]), int(tbl.window_stop[k])
        if b <= a or tbl.intensity[k] < 0.05 * tbl.intensity.max():
            continue
        near = np.abs(tbl.two_theta - tbl.two_theta[k]) < 3.0 * tbl.fwhm[k]
        if near.sum() > 1:
            continue
        ratios.append(float(full_ref[a:b].sum() / full_bragg[a:b].sum()))
    assert len(ratios) >= 5
    assert np.allclose(ratios, 1.0, atol=0.02), ratios
    # 形: スケールを最小二乗で合わせたあとの相対差。GSAS-II の EPSVOIGT は複素指数積分を
    # 近似で計算するため、立ち上がりの遅い(α の小さい)ピークの裾で最大 1 割ほど違う
    # (こちらの閉形式は数値積分と 7 桁一致。docs/validation/phase7_tof.md)
    scale = float(np.dot(bragg, bragg_ref) / np.dot(bragg, bragg))
    diff = scale * bragg - bragg_ref
    peak = bragg_ref > 0.01 * bragg_ref.max()
    rel = np.abs(diff)[peak] / bragg_ref[peak]
    print(
        f"[nac_tof] スケール比 {scale:.6g}; Bragg 成分の相対差: 中央値 {np.median(rel):.2e}, "
        f"最大 {rel.max():.2e}; 最大|Δ|/max {np.abs(diff).max() / bragg_ref.max():.2e}; "
        f"孤立反射の積分強度比 {np.min(ratios):.4f}〜{np.max(ratios):.4f}"
    )
    assert abs(scale - 1.0) < 0.01
    assert np.median(rel) < 0.15
    assert np.abs(diff).max() / bragg_ref.max() < 0.08
