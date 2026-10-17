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
    # 探索回数は既定(25)。15 回だと GitHub Actions の Windows で C = 0.93 で上限に達した
    res = run_mem(inp)
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


def test_grid_check_uses_equivalent_reflections() -> None:
    """格子点数の検査は全等価反射で行う(レビュー 2026-10-17 精密化 #12)。

    石英 P3₁21 では独立反射の |h| は (9, 5, 11) でも等価反射に (−h−k) 成分の 14 が出る。
    修正前は (19, 11, 23) が通り、96 の等価反射が同じ格子点に折り返されて C = 371 で上限に達した。
    """
    st = Structure(
        Lattice.hexagonal(4.9, 5.4),
        SpaceGroup.from_symbol("P 31 2 1"),
        [
            AtomSite("Si", "Si", [0.47, 0.0, 0.0], u_iso=0.01),
            AtomSite("O", "O", [0.41, 0.27, 0.12], u_iso=0.01),
        ],
    )
    refl = generate_reflections(st.lattice, st.spacegroup, 1.0)
    hmax = np.abs(refl.hkl).max(axis=0)
    grid = tuple(int(2 * h + 1) for h in hmax)
    f = np.ones(len(refl), dtype=complex)
    with pytest.raises(ValueError, match="粗すぎます"):
        MEMInput(st.lattice, st.spacegroup, refl.hkl, f, np.ones(len(refl)), 100.0, grid)
    # 等価反射まで入る格子なら通る
    MEMInput(st.lattice, st.spacegroup, refl.hkl, f, np.ones(len(refl)), 100.0, (30, 30, 23))


def test_sigma_inf_excluded_from_constraint_and_count() -> None:
    """σ = inf の反射は F 拘束に入れず N_F にも数えない(測定範囲外の反射)。"""
    from kaisetsu.mem.core import _Model

    inp, _ = _synthetic_input(_nacl(), 1.0, 16, 0.02)
    n = len(inp.hkl)
    base = _Model(inp)
    sig = inp.sigma.copy()
    sig[-3:] = np.inf
    fobs = inp.f_obs.copy()
    fobs[-3:] *= 10.0  # 値がどうでも C に効かない
    inp2 = MEMInput(inp.lattice, inp.spacegroup, inp.hkl, fobs, sig, inp.f000, inp.grid)
    model = _Model(inp2)
    assert base.n_f == n + 1 and model.n_f == n + 1 - 3
    assert np.all(model.w_eq[np.isin(model.owner, np.arange(n - 2, n + 1))] == 0.0)
    m = np.full(model.n, inp.f000 / model.n)
    c_full = base.chi2(base.f_calc_eq(m))
    c_part = model.chi2(model.f_calc_eq(m))
    # 残り n − 2 本(F000 込み)の寄与だけを N_F で割った値になる
    d = base.f_obs_eq - base.f_calc_eq(m)
    keep = ~np.isin(base.owner, np.arange(n - 2, n + 1))
    expect = float(np.sum((base.w_eq * (d.real**2 + d.imag**2))[keep])) / (n + 1 - 3)
    assert c_part == pytest.approx(expect, rel=1e-12)
    assert c_part != pytest.approx(c_full, rel=1e-6)
    with pytest.raises(ValueError):
        MEMInput(inp.lattice, inp.spacegroup, inp.hkl, fobs, np.where(sig > 1, np.nan, sig),
                 inp.f000, inp.grid)  # fmt: skip


def test_two_channel_auto_from_negative_scattering() -> None:
    """two_channel を省略(None)すると負の散乱長があるときだけ 2 チャンネルになる。"""
    inp, _ = _synthetic_input(_rutile(), 0.8, 16, 0.01, neutron=True)
    assert inp.f000_negative > 0
    auto = MEMInput(inp.lattice, inp.spacegroup, inp.hkl, inp.f_obs, inp.sigma, inp.f000, inp.grid,
                    f000_negative=inp.f000_negative)  # fmt: skip
    assert auto.two_channel is True
    xray, _ = _synthetic_input(_nacl(), 1.0, 16, 0.02)
    assert xray.two_channel is False
    assert MEMInput(xray.lattice, xray.spacegroup, xray.hkl, xray.f_obs, xray.sigma, xray.f000,
                    xray.grid).two_channel is False  # fmt: skip


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


def test_mem_returns_closest_to_target_when_search_exhausted() -> None:
    """λ の探索が上限に達したら、最後の λ ではなく目標 χ² に最も近かった収束した解を返す。"""
    inp, _ = _synthetic_input(_nacl(), 1.0, 16, 0.02)
    res = run_mem(inp, max_outer=3)
    assert "上限" in res.message
    assert len(res.history_converged) == len(res.history) == 3
    conv = [lc for lc, ok in zip(res.history, res.history_converged, strict=True) if ok]
    pool = conv if conv else res.history
    best = min(pool, key=lambda lc: abs(np.log(lc[1])))
    assert res.lam == best[0] and res.chi2 == best[1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lambda_search_with_small_inner_budget(seed: int) -> None:
    """内側の予算が小さくても(400)λ の探索は収束する(レビュー 2026-10-17 差分 #1)。

    修正前はルチル X 線の seed 0〜2 で、遠い λ の打ち切った C(1.48 など)が lo に入って探索が
    そこに縮み、上限 25 回に達して打ち切り途中の密度を返した。
    """
    inp, _ = _synthetic_input(_rutile(), 0.8, 16, 0.01, seed=seed)
    res = run_mem(inp, max_inner=400)
    assert "収束" in res.message
    assert res.chi2 == pytest.approx(1.0, abs=0.02)
    assert len(res.history) < 15


def test_lambda_search_ignores_truncated_chi2_for_bracket(monkeypatch: pytest.MonkeyPatch) -> None:
    """打ち切った内側の C は lo / hi に使わない(模擬の内側解法で探索の論理だけを試す)。

    収束した C(λ) = (λ/λ*)^-1.7、打ち切り(遠い λ、小さな予算)の C は前の λ からの温かい
    初期値の名残で「目標を少し超えた」値 1.5 を返す。修正前はこの λ が lo に入って探索が縮み、
    上限に達した。
    """
    import kaisetsu.mem.core as core

    lam_star = 0.1
    calls: list[tuple[float, int]] = []

    def fake_minimize(
        objective, x0, lam, bounds, *, max_iter, window, settle_tol, settle_relative, chi2_of
    ):  # type: ignore[no-untyped-def]
        calls.append((lam, max_iter))
        c_true = (lam / lam_star) ** -1.7
        if max_iter < 1000:  # 遠い λ の小さな予算 = 打ち切り。C は当てにならない
            c_true = 1.5 if c_true < 3.0 else c_true
        # 変数は v = 2 sqrt(m) で m = v²/4。模擬の chi2 は m[0] を返すので v = 2 sqrt(C) にする
        x = np.full_like(x0, 2.0 * np.sqrt(c_true))
        if max_iter < 1000:
            return x, max_iter, False, False
        return x, max_iter // 4, False, True

    inp, _ = _synthetic_input(_nacl(), 1.0, 16, 0.02)
    monkeypatch.setattr(core, "_minimize_at_lambda", fake_minimize)
    monkeypatch.setattr(core._Model, "chi2", lambda self, fc: float(np.real(fc[0])))
    monkeypatch.setattr(core._Model, "f_calc_eq", lambda self, m: m.astype(np.complex128))
    res = run_mem(inp, max_inner=2000, lam0=lam_star * 1e-3)
    assert "収束" in res.message
    assert res.lam == pytest.approx(lam_star, rel=0.02)
    assert len(res.history) <= 10
    # 打ち切りの点(C = 1.5)は収束扱いにならない
    assert not all(res.history_converged)
