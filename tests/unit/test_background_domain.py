"""Chebyshev 背景の定義域はフィット範囲に合わせる(定義域が広いと係数が決まらない)。"""

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import Project, Radiation, SampleGeometry
from kaisetsu.crystal import Structure
from kaisetsu.io import read_powder_data
from kaisetsu.pattern.instrument import CWInstrument
from kaisetsu.profile import ChebyshevBackground


def test_with_domain_keeps_curve() -> None:
    bg = ChebyshevBackground([100.0, -20.0, 5.0, 1.0], 10.0, 90.0)
    bg2 = bg.with_domain(20.0, 40.0)
    x = np.linspace(20.0, 40.0, 11)
    assert bg2.order == bg.order and (bg2.x_min, bg2.x_max) == (20.0, 40.0)
    assert np.allclose(bg2(x), bg(x))
    with pytest.raises(ValueError):
        bg.with_domain(40.0, 20.0)


def _project(data_dir: Path, **kw: float) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA")
    inst = CWInstrument(Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01)
    proj = Project("bg")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=8, **kw)
    return proj


def test_background_domain_follows_fit_range(data_dir: Path) -> None:
    """add_histogram の x_min, x_max が定義域になり、set_range で曲線を変えずに付いてくる。"""
    proj = _project(data_dir, x_min=20.0, x_max=40.0)
    h = proj.histograms[0]
    assert (h.background.x_min, h.background.x_max) == (20.0, 40.0)
    x = np.linspace(25.0, 35.0, 7)
    before = h.background(x)
    proj.vary("h0:bkg:*")
    proj.set_range(0, 25.0, 35.0)
    h = proj.histograms[0]
    assert (h.x_min, h.x_max) == (25.0, 35.0)
    assert (h.background.x_min, h.background.x_max) == (25.0, 35.0)
    assert np.allclose(h.background(x), before)
    p0 = proj.params["h0:bkg:0"]
    assert p0.vary and p0.value == pytest.approx(h.background.coefficients[0])
    with pytest.raises(ValueError):
        proj.set_range(0, 35.0, 25.0)


def test_narrow_range_background_is_well_conditioned(data_dir: Path) -> None:
    """定義域をデータ全体(10〜160°)のままにして 20〜40° で合わせると係数の相関が 1.000 になる。"""
    proj = _project(data_dir, x_min=20.0, x_max=40.0)
    proj.vary("h0:bkg:*", "PbSO4:scale")
    res = proj.refine(max_cycles=2)
    ib = [i for i, n in enumerate(res.names) if "bkg" in n]
    off = np.abs(res.correlation[np.ix_(ib, ib)] - np.eye(len(ib)))
    assert off.max() < 0.999
    # 旧挙動(定義域 = データ全体)では 1.000 になることを確かめておく
