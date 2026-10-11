"""TOF ピークの一括計算(numba): 複素指数積分の精度、ピークごとの計算との一致、パターンの一致。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import exp1

from kaisetsu.profile.tof_batch import scaled_exp1, tof_profile_batch
from kaisetsu.profile.tof_profile import tof_peak


def test_scaled_exp1_matches_scipy() -> None:
    rs = np.logspace(-3, 2.6, 41)
    args = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 91)
    r, a = np.meshgrid(rs, args)
    w = (r * np.exp(1j * a)).ravel()
    ref = np.exp(w) * exp1(w)
    got = scaled_exp1(w)
    rel = np.abs(got - ref) / np.abs(ref)
    assert rel.max() < 1e-10
    # 実際に出る引数(Δ ±300 μs、α = 0.13、β = 0.16、Γ = 1〜50 μs)
    d = np.linspace(-300.0, 300.0, 601)
    for gam in (1.0, 10.0, 50.0):
        for mult in (0.13, -0.16):
            wz = mult * (d - 0.5j * gam)
            ref = np.exp(wz) * exp1(wz)
            assert np.allclose(scaled_exp1(wz), ref, rtol=1e-10, atol=0.0)


def test_batch_matches_per_peak() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(10000.0, 60000.0, 20001)
    n = 40
    pos = rng.uniform(12000.0, 58000.0, n)
    alpha = rng.uniform(0.05, 0.5, n)
    beta = rng.uniform(0.02, 0.3, n)
    sig2 = rng.uniform(10.0, 4000.0, n)
    gam = np.where(rng.uniform(size=n) < 0.3, 1e-5, rng.uniform(0.5, 80.0, n))
    inten = rng.uniform(1.0, 100.0, n)
    i0 = np.searchsorted(x, pos - 800.0).astype(np.int64)
    i1 = np.searchsorted(x, pos + 1200.0).astype(np.int64)
    out, offsets, lengths = tof_profile_batch(x, pos, alpha, beta, sig2, gam, inten, i0, i1)
    for k in range(n):
        ref = inten[k] * tof_peak(x[i0[k] : i1[k]] - pos[k], alpha[k], beta[k], sig2[k], gam[k])
        got = out[offsets[k] : offsets[k] + lengths[k]]
        assert np.allclose(got, ref, rtol=1e-9, atol=1e-12 * inten[k]), k
    # 窓が空のピーク
    i1_empty = i0.copy()
    out2, _, lengths2 = tof_profile_batch(x, pos, alpha, beta, sig2, gam, inten, i0, i1_empty)
    assert out2.size == 0 and lengths2.sum() == 0


def test_pattern_uses_batch_and_is_fast() -> None:
    from kaisetsu.api import AtomSite, Lattice, PhaseContribution, SpaceGroup, Structure
    from kaisetsu.pattern.tof import TOFInstrument, calculate_pattern_tof

    st = Structure(
        Lattice.cubic(10.25),
        SpaceGroup.from_symbol("I 21 3"),
        [AtomSite("Ca", "Ca", [0.4665, 0.0, 0.25], u_iso=0.008)],
    )
    inst = TOFInstrument(
        difc=22600.0, alpha=0.13, beta0=0.11, beta1=0.0027, sig1=100.0, x=2.0, y=1.0, two_theta=90.0
    )
    x = np.geomspace(9000.0, 100000.0, 6000)
    res = calculate_pattern_tof(x, [PhaseContribution(st, scale=100.0, size_um=1.0)], inst)
    tbl = res.peaks[0]
    assert len(tbl.hkl) > 500
    # 寄与の和がパターンに一致する
    y = np.zeros_like(x)
    for k, c in enumerate(tbl.contributions):
        if len(c):
            y[tbl.window_start[k] : tbl.window_stop[k]] += c
    assert np.allclose(y, res.y_bragg, rtol=1e-12, atol=1e-9)
    assert pytest.approx(res.y_bragg.sum(), rel=1e-12) == y.sum()
