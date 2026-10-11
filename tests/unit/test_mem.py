"""MEM の単体テスト: 合成 F からの密度回収、対称性、2 チャンネル、ZSPA、書き出し。

32³ 格子の NaCl など重いものは slow。速いテストは 16³ 格子で基本動作を見る。
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.mem import (
    MEMInput,
    density_slice,
    expand_reflections,
    line_profile,
    run_mem,
    write_ccp4,
    write_grd,
    write_xplor,
)
from kaisetsu.reflections.hkl import generate_reflections
from kaisetsu.reflections.structure_factor import structure_factors
from kaisetsu.scattering import Radiation


def _nacl() -> Structure:
    return Structure(
        Lattice.cubic(5.64),
        SpaceGroup.from_symbol("F m -3 m"),
        [
            AtomSite("Na", "Na1+", [0, 0, 0], u_iso=0.02),
            AtomSite("Cl", "Cl1-", [0.5, 0.5, 0.5], u_iso=0.015),
        ],
        name="NaCl",
    )


def _rutile() -> Structure:
    return Structure(
        Lattice.tetragonal(4.594, 2.959),
        SpaceGroup.from_symbol("P42/mnm"),
        [
            AtomSite("Ti", "Ti", [0, 0, 0], u_iso=0.006),
            AtomSite("O", "O", [0.305, 0.305, 0], u_iso=0.008),
        ],
    )


def _synthetic_input(
    st: Structure, d_min: float, grid: int, noise: float, seed: int = 0, neutron: bool = False
) -> tuple[MEMInput, np.ndarray]:
    rad = Radiation.neutron(1.5) if neutron else Radiation.xray(0.5)
    refl = generate_reflections(st.lattice, st.spacegroup, d_min)
    no_anom = None if neutron else dict.fromkeys(st.species_list(), 0j)
    f = structure_factors(st, refl.hkl, rad, anomalous=no_anom)
    rng = np.random.default_rng(seed)
    sigma = np.maximum(noise * np.abs(f), noise * np.abs(f).max() * 0.05)
    f_obs = f + rng.normal(0, 1, len(f)) * sigma  # 位相は正しいまま振幅にノイズ
    ex = st.expand()
    if neutron:
        from kaisetsu.scattering.neutron import NeutronScatteringLengths

        tbl = NeutronScatteringLengths()
        b = np.array([tbl.scattering_length(sp).real for sp in ex.species])[ex.species_index]
    else:
        from kaisetsu.scattering.xray import XrayScatteringFactors

        xt = XrayScatteringFactors()
        b = np.array([float(xt.f0(sp, 0.0)) for sp in ex.species])[ex.species_index]
    tot = b * ex.occupancy
    pos, neg = float(np.sum(tot[tot > 0])), float(-np.sum(tot[tot < 0]))
    inp = MEMInput(
        st.lattice, st.spacegroup, refl.hkl, f_obs, sigma, pos - neg, (grid, grid, grid),
        two_channel=neg > 0, f000_negative=neg,
    )  # fmt: skip
    return inp, f


def test_expand_reflections_consistent_phases() -> None:
    st = _nacl()
    refl = generate_reflections(st.lattice, st.spacegroup, 1.0)
    no_anom = dict.fromkeys(st.species_list(), 0j)
    f = structure_factors(st, refl.hkl, Radiation.xray(0.5), anomalous=no_anom)
    idx, vals, owner = expand_reflections(st.spacegroup, refl.hkl, f, (32, 32, 32))
    assert len(idx) == len(np.unique(idx))  # 重複なし
    counts = np.bincount(owner)
    # 展開数 = 多重度(Friedel 込みでラウエ群 m-3m の多重度と同じ)
    assert np.all(counts == refl.multiplicity)
    # 展開した F は元の構造の F と一致する(位相の変換が正しい)
    hkl_eq = np.stack(np.unravel_index(idx, (32, 32, 32)), axis=1)
    hkl_eq = np.where(hkl_eq > 16, hkl_eq - 32, hkl_eq)
    f_eq = structure_factors(st, hkl_eq, Radiation.xray(0.5), anomalous=no_anom)
    assert np.allclose(f_eq, vals, atol=1e-6)


def test_mem_small_grid_basics() -> None:
    """16³ 格子・d ≥ 1.0 Å の NaCl: C → 1、正値、対称、極大は原子位置、総電子数が保たれる。"""
    st = _nacl()
    inp, _ = _synthetic_input(st, 1.0, 16, 0.02)
    res = run_mem(inp, max_outer=15)
    assert res.chi2 == pytest.approx(1.0, abs=0.05)
    assert np.all(res.density >= 0)
    n = 16
    assert res.density[0, 0, 0] == pytest.approx(res.density[n // 2, n // 2, 0], rel=1e-6)
    assert res.density[n // 2, n // 2, n // 2] > res.density[0, 0, 0]
    voxel = st.lattice.volume / n**3
    assert float(res.density.sum() * voxel) == pytest.approx(4 * 28, rel=1e-3)
    assert res.r_factor() < 0.1  # 反射 15 本・2% ノイズ。弱い反射の σ が相対的に大きい
    assert "収束" in res.message


@pytest.mark.slow
def test_mem_recovers_nacl_density() -> None:
    st = _nacl()
    inp, _ = _synthetic_input(st, 0.6, 32, 0.01)
    res = run_mem(inp, max_outer=20)
    assert res.chi2 == pytest.approx(1.0, abs=0.05)
    assert np.all(res.density >= 0)
    # 極大は Na と Cl の位置(Cl 4 個 + Na 4 個)
    peaks = res.maxima(8, min_separation=1.5)
    frac = [p for p, _ in peaks]
    assert any(np.allclose(p % 1.0, [0.5, 0.5, 0.5], atol=1e-6) for p in frac)
    assert any(np.allclose(p % 1.0, [0, 0, 0], atol=1e-6) for p in frac)
    # 球内の積分電子数がイオンの電子数に近い(分解能 0.6 Å と MEM の平滑化で 1 割ほど少なめに出る)
    assert res.integrate_sphere([0.5, 0.5, 0.5], 1.4) == pytest.approx(18.0, rel=0.15)
    assert res.integrate_sphere([0, 0, 0], 1.2) == pytest.approx(10.0, rel=0.1)
    assert res.r_factor() < 0.03


@pytest.mark.slow
def test_mem_zspa_agrees_with_lbfgs() -> None:
    st = _nacl()
    inp, _ = _synthetic_input(st, 0.8, 24, 0.02)
    a = run_mem(inp, method="lbfgs", max_outer=20)
    b = run_mem(inp, method="zspa", zspa_lambda=0.1, zspa_max_iter=4000)
    assert b.chi2 <= 1.05
    # 事前分布を前の反復の密度に置き換える ZSPA は同じ C でも少し鋭い密度に収束する。
    # 分布の形(相関)と極大の位置が一致することを確認する
    assert np.corrcoef(a.density.ravel(), b.density.ravel())[0, 1] > 0.97
    pa = {tuple(np.round(p % 1.0, 3)) for p, _ in a.maxima(8, 1.5)}
    pb = {tuple(np.round(p % 1.0, 3)) for p, _ in b.maxima(8, 1.5)}
    assert pa == pb
    assert b.r_factor() < 0.05


def test_two_channel_neutron() -> None:
    """Ti(b < 0)を含む TiO2 ルチルの核密度: 負の散乱長の位置が負の密度になる。"""
    inp, _ = _synthetic_input(_rutile(), 0.8, 16, 0.01, neutron=True)
    assert inp.two_channel
    res = run_mem(inp, max_outer=15)
    assert res.chi2 == pytest.approx(1.0, abs=0.05)
    assert res.density_negative is not None and np.all(res.density_negative >= 0)
    n = 16
    assert res.density[0, 0, 0] < 0  # Ti: b = -3.37 fm
    i = round(0.305 * n)
    assert res.density[i, i, 0] > 0  # O: b = +5.8 fm
    # 密度の総和 = F000(正 − 負)
    voxel = inp.lattice.volume / n**3
    assert float(res.density.sum() * voxel) == pytest.approx(inp.f000, rel=1e-3)
    with pytest.raises(ValueError):
        run_mem(inp, method="zspa")


def test_grid_too_coarse_raises() -> None:
    st = _nacl()
    refl = generate_reflections(st.lattice, st.spacegroup, 0.6)
    with pytest.raises(ValueError):
        MEMInput(st.lattice, st.spacegroup, refl.hkl, np.ones(len(refl)), np.ones(len(refl)),
                 100.0, (8, 8, 8))  # fmt: skip


def test_write_formats(tmp_path: Path) -> None:
    st = _nacl()
    inp, _ = _synthetic_input(st, 1.0, 16, 0.02)
    res = run_mem(inp, max_outer=10)
    p = write_ccp4(res.density, st.lattice, st.spacegroup, tmp_path / "d.ccp4")
    m = gemmi.read_ccp4_map(str(p))
    arr = np.array(m.grid, copy=False)
    assert arr.shape == (16, 16, 16)
    assert np.allclose(arr, res.density.astype(np.float32), rtol=1e-5)
    assert m.grid.unit_cell.a == pytest.approx(5.64)
    x = write_xplor(res.density, st.lattice, tmp_path / "d.xplor")
    text = x.read_text(encoding="ascii").splitlines()
    assert text[1].strip().startswith("1 !NTITLE") and "ZYX" in text[5]
    vals = [float(v) for line in text[7:] for v in line.split() if "E" in v]
    assert len(vals) >= 16**3
    g = write_grd(res.density, st.lattice, tmp_path / "d.grd")
    lines = g.read_text(encoding="ascii").splitlines()
    assert lines[2].split() == ["16", "16", "16"]
    # 断面と線分
    sl, axes = density_slice(res.density, 2, 0.5)
    assert sl.shape == (16, 16) and axes == (0, 1)
    d, v = line_profile(res.density, st.lattice, [0, 0, 0], [0.5, 0.5, 0.5], 50)
    assert d[-1] == pytest.approx(5.64 * np.sqrt(3) / 2)
    assert v[0] == pytest.approx(res.density[0, 0, 0])
