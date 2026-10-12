"""Z-Rietveld 2.9.1 の Type0m 計算曲線との比較。

参照値は Z-Rietveld がサンプル #2_CeO2_iMATERIA の装置ファイル(BS 155° バンク)と既定相 Si で
0 サイクル実行したプロジェクト(.zrietveldn、zstd 圧縮の CBOR)から取り出した計算曲線
(`validation/reference/zrietveld_type0m.json`、スクリプトは docs/validation/phase7_tof.md 8 節)。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.pattern.tof import TOFInstrument, zrietveld_peak
from tests.conftest import load_reference

pytestmark = pytest.mark.validation


def _instrument(ref: dict) -> TOFInstrument:
    i = ref["instrument"]
    return TOFInstrument(
        difc=i["c1"],
        difa=i["c2"],
        zero=i["c0"],
        two_theta=i["two_theta"],
        sig0=i["SigmaSquare"][0],
        sig1=i["SigmaSquare"][1],
        sig2=i["SigmaSquare"][2],
        z=i["Gamma"][0],
        x=i["Gamma"][1],
        y=i["Gamma"][2],
        profile_kind="zrietveld",
        zr_alpha1=i["Alpha1"],
        zr_alpha2=i["Alpha2"],
        zr_beta1=i["Beta1_0"],
        zr_beta0_0=i["Beta0_0"],
        zr_beta0_1=i["Beta0_1"],
        zr_r01=i["r01"],
        zr_r02=i["r02"],
        zr_r03=i["r03"],
    )


def test_type0m_d_dependence_matches_zrietveld() -> None:
    """α', β₀', β₁', R, σ², γ, 位置の d 依存(Z-Rietveld が反射ごとに書き出した値と比べる)。"""
    ref = load_reference("zrietveld_type0m")
    inst = _instrument(ref)
    for pk in ref["peaks"]:
        d = np.array([pk["d"]])
        assert float(inst.position(d)[0]) == pytest.approx(pk["position"], abs=2e-3)
        assert float(inst.sigma2_of(d)[0]) == pytest.approx(pk["sigma2"], rel=1e-6)
        assert float(inst.gamma_of(d)[0]) == pytest.approx(pk["gamma"], rel=1e-6)
        alpha, beta0, beta1, r = inst.zr_rates(d)
        tau = inst.zr_time_scale(d)
        # 率 × 時定数の単位 = パラメータの逆数
        assert float(1.0 / (alpha[0] * tau[0])) == pytest.approx(pk["alpha"], rel=1e-6)
        assert float(1.0 / (beta0[0] * tau[0])) == pytest.approx(pk["beta0"], rel=1e-6)
        assert float(1.0 / (beta1[0] * tau[0])) == pytest.approx(pk["beta1"], rel=1e-6)
        assert float(r[0]) == pytest.approx(pk["R"], abs=1e-6)


def test_type0m_peak_shape_matches_zrietveld() -> None:
    """ピーク形状(面積 1 × 強度)が Z-Rietveld の計算曲線と一致する。

    形状の関数形は Si 111 で残差 2e-7(数値畳み込み)。時定数の単位 ρ(d) が経験式なので
    0.1% 程度、閉形式の擬 Voigt と Z-Rietveld の数値畳み込みの差を含めて最大 0.5% を許す。
    """
    ref = load_reference("zrietveld_type0m")
    inst = _instrument(ref)
    worst = 0.0
    for pk in ref["peaks"]:
        x = np.asarray(pk["x"], dtype=np.float64)
        y_ref = np.asarray(pk["y"], dtype=np.float64)
        y = pk["intensity"] * zrietveld_peak(inst, x, pk["d"], pk["sigma2"], pk["gamma"])
        for nb in pk["neighbors"]:  # 窓にかかる他の反射(Z-Rietveld の計算曲線は全反射の和)
            y = y + nb["intensity"] * zrietveld_peak(inst, x, nb["d"], nb["sigma2"], nb["gamma"])
        err = float(np.max(np.abs(y - y_ref)) / y_ref.max())
        worst = max(worst, err)
        assert err < 5e-3, (pk["hkl"], err)
        # 重心(位置の定義が同じ: 接合点)
        w = y_ref > 0
        c_ref = float(np.sum(x[w] * y_ref[w]) / np.sum(y_ref[w]))
        c = float(np.sum(x[w] * y[w]) / np.sum(y[w]))
        assert abs(c - c_ref) < 0.3, (pk["hkl"], c - c_ref)
    print(f"[zrietveld_type0m] worst max|dy|/max = {worst:.2e}")
