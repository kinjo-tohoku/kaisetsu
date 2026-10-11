"""FCJ 積分の精度: 置換法(既定)が収束していること、GSAS 流と一致方向であること。"""

import math

import numpy as np
import pytest

from kaisetsu.profile.fcj import fcj_limits, fcj_pseudo_voigt
from kaisetsu.profile.fcj_batch import fcj_profile_batch
from kaisetsu.profile.pseudo_voigt import pseudo_voigt_point, tch_fwhm_eta


def _reference(
    x: np.ndarray, pos: float, fg: float, fl: float, s_l: float, h_l: float, n: int = 300
) -> np.ndarray:
    """置換法を純 Python で n 節点(十分多い)で評価した参照値。"""
    tpm, tpi = fcj_limits(pos, s_l, h_l)
    span = pos - tpm
    fwhm, eta = tch_fwhm_eta(fg, fl)
    rad = math.pi / 180.0
    cos2t, sin2t = math.cos(pos * rad), math.sin(pos * rad)
    apb, small = s_l + h_l, min(s_l, h_l)
    u_infl = math.sqrt(min(max((pos - tpi) / span, 0.0), 1.0))
    panels = ([(0.0, u_infl)] if u_infl > 1e-9 else []) + [(u_infl, 1.0)]
    num = np.zeros_like(x)
    den = 0.0
    xs, ws = np.polynomial.legendre.leggauss(n)
    for a, b in panels:
        for xk, wk in zip(xs, ws, strict=True):
            u = a + (b - a) * (xk + 1) / 2
            d = pos - span * u * u
            cd, sd = math.cos(d * rad), math.sin(d * rad)
            tmp = (cd * cd - cos2t * cos2t) if pos > 45 else (sin2t * sin2t - sd * sd)
            f = abs(cos2t) / math.sqrt(tmp) if tmp > 0 else 1.0
            g = 2 * small * f / cd if abs(d - tpm) > abs(tpi - tpm) else (-1 + apb * f) / cd
            w = wk * (b - a) / 2 * g * 2 * span * u
            den += w
            num += w * np.array([pseudo_voigt_point(xi - d, fwhm, eta) for xi in x])
    return num / den


@pytest.mark.parametrize(
    ("pos", "fg", "fl", "s_l", "h_l"),
    [
        (16.5, 0.09, 0.015, 0.03, 0.02),
        (16.5, 0.09, 0.0, 0.025, 0.025),
        (8.0, 0.1, 0.02, 0.03, 0.03),
        (30.0, 0.06, 0.02, 0.025, 0.025),
        (120.0, 0.2, 0.1, 0.025, 0.025),
        (150.0, 0.3, 0.1, 0.02, 0.01),
    ],
)
def test_accurate_method_converges(
    pos: float, fg: float, fl: float, s_l: float, h_l: float
) -> None:
    x = np.arange(pos - 1.5, pos + 1.0, 0.01)
    ref = _reference(x, pos, fg, fl, s_l, h_l)
    ours = fcj_pseudo_voigt(x, pos, fg, fl, s_l, h_l)  # 既定 = accurate
    assert np.max(np.abs(ours - ref)) / ref.max() < 1e-4
    # GSAS 流は 1/n 収束なので自動の節点数では数 % の誤差が出る
    gsas = fcj_pseudo_voigt(x, pos, fg, fl, s_l, h_l, method="gsas")
    assert np.max(np.abs(gsas - ref)) / ref.max() < 6e-2
    # 面積は 1
    xw = np.arange(pos - 15.0, pos + 15.0, 0.01)  # ローレンツ裾を含む広い範囲で面積 1
    assert np.sum(fcj_pseudo_voigt(xw, pos, fg, fl, s_l, h_l)) * 0.01 == pytest.approx(
        1.0, abs=5e-3
    )


def test_batch_matches_single_and_handles_windows() -> None:
    x = np.arange(10.0, 60.0, 0.02)
    pos = np.array([15.0, 15.08, 30.0, 45.0])
    fg = np.array([0.08, 0.08, 0.06, 0.05])
    fl = np.array([0.02, 0.02, 0.03, 0.04])
    inten = np.array([100.0, 50.0, 80.0, 0.0])
    i0 = np.searchsorted(x, pos - 1.0)
    i1 = np.searchsorted(x, pos + 1.0)
    for method in ("accurate", "gsas"):
        out, offsets, lengths = fcj_profile_batch(
            x, pos, fg, fl, inten, i0, i1, 0.02, 0.02, method=method
        )
        assert lengths.sum() == out.size
        for p in range(4):
            part = out[offsets[p] : offsets[p] + lengths[p]]
            single = inten[p] * fcj_pseudo_voigt(
                x[i0[p] : i1[p]], pos[p], fg[p], fl[p], 0.02, 0.02, method=method
            )
            assert np.allclose(part, single, rtol=1e-10, atol=1e-12)
    # 対称(S/L = H/L = 0)でも動く
    out, offsets, lengths = fcj_profile_batch(x, pos, fg, fl, inten, i0, i1, 0.0, 0.0)
    part = out[offsets[0] : offsets[0] + lengths[0]]
    assert np.allclose(
        part, inten[0] * fcj_pseudo_voigt(x[i0[0] : i1[0]], pos[0], fg[0], fl[0], 0.0, 0.0)
    )
    with pytest.raises(ValueError):
        fcj_profile_batch(x, pos, fg, fl, inten, i0, i1, 0.02, 0.02, method="foo")
