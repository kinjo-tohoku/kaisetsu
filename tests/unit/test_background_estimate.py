import numpy as np
import pytest

from kaisetsu.io import read_powder_data
from kaisetsu.profile.background import ChebyshevBackground, estimate_chebyshev_background


def test_estimate_recovers_smooth_background_under_peaks() -> None:
    rng = np.random.default_rng(3)
    x = np.arange(10.0, 120.0, 0.02)
    truth = 200.0 * np.exp(-(x - 10.0) / 15.0) + 40.0 + 0.3 * (x - 60.0)
    peaks = np.zeros_like(x)
    for pos, h in (
        (20.5, 5000),
        (28.0, 12000),
        (35.3, 3000),
        (44.0, 2500),
        (60.7, 1500),
        (90.2, 800),
    ):
        peaks += h * np.exp(-0.5 * ((x - pos) / 0.08) ** 2)
    y = rng.poisson(truth + peaks).astype(float)
    bkg = estimate_chebyshev_background(x, y, n_terms=8)
    est = bkg(x)
    # ピークのない所では真の背景に数カウント以内、ピーク位置ではピークに引きずられない
    free = peaks < 1.0
    assert np.abs(est[free] - truth[free]).mean() < 6.0
    assert np.all(est < truth + 25.0)
    assert est[np.argmin(np.abs(x - 28.0))] < 150.0  # 最強ピークの位置でも背景は背景


def test_estimate_on_real_data_sits_under_peaks(data_dir) -> None:
    d = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA")
    bkg = estimate_chebyshev_background(d.x, d.y, n_terms=10)
    est = bkg(d.x)
    assert isinstance(bkg, ChebyshevBackground) and bkg.order == 10
    # 低角の立ち上がりを捉える(10° で観測 179 に対し 130 以上)
    assert est[0] > 130.0
    # 全域で観測の 99 パーセンタイルより下、負にならない
    assert np.all(est < np.percentile(d.y, 99))
    assert np.all(est > 0)
    # ピークのない領域(例: 11〜15°)では観測の平均に近い
    m = (d.x > 11) & (d.x < 15)
    assert abs(est[m].mean() - d.y[m].mean()) < 8.0
    with pytest.raises(ValueError):
        estimate_chebyshev_background(d.x[:10], d.y[:10], n_terms=8)
