"""MEM の G 拘束(重なり反射の群)。"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.mem import MEMInput, run_mem
from kaisetsu.mem.core import _Model
from kaisetsu.reflections.hkl import generate_reflections
from kaisetsu.reflections.structure_factor import structure_factors
from kaisetsu.scattering import Radiation


def _nacl_input(grid: int = 16, d_min: float = 1.0) -> tuple[MEMInput, np.ndarray]:
    st = Structure(
        Lattice.cubic(5.64),
        SpaceGroup.from_symbol("F m -3 m"),
        [
            AtomSite("Na", "Na1+", [0, 0, 0], u_iso=0.02),
            AtomSite("Cl", "Cl1-", [0.5, 0.5, 0.5], u_iso=0.015),
        ],
    )
    refl = generate_reflections(st.lattice, st.spacegroup, d_min)
    no_anom = dict.fromkeys(st.species_list(), 0j)
    f = structure_factors(st, refl.hkl, Radiation.xray(0.5), anomalous=no_anom)
    sigma = np.maximum(0.02 * np.abs(f), 0.5)
    inp = MEMInput(st.lattice, st.spacegroup, refl.hkl, f, sigma, 4 * 28.0, (grid, grid, grid))
    return inp, refl.multiplicity.astype(np.int64)


def test_group_gradient_matches_numerical() -> None:
    inp, mult = _nacl_input()
    n = len(inp.hkl)
    # 最後の 2 本を群にする
    idx = np.array([n - 2, n - 1])
    g_obs = float(np.sum(mult[idx] * np.abs(inp.f_obs[idx]) ** 2)) * 0.9
    inp.overlap_groups = [(idx, g_obs, 0.05 * g_obs, mult[idx])]
    model = _Model(inp)
    assert model.n_groups == 1 and model.n_f == n + 1 - 2
    rng = np.random.default_rng(0)
    m = np.abs(rng.normal(1.0, 0.3, model.n)) * (112.0 / model.n)
    fc = model.f_calc_eq(m)
    _val, resid = model.group_terms(fc)
    grad = model.group_gradient(fc, resid)
    # 数値微分と比較(数点)
    for i in rng.choice(model.n, 5, replace=False):
        h = 1e-4 * m[i]
        mp = m.copy()
        mp[i] += h
        mm = m.copy()
        mm[i] -= h
        vp, _ = model.group_terms(model.f_calc_eq(mp))
        vm, _ = model.group_terms(model.f_calc_eq(mm))
        num = (vp - vm) / (2 * h)
        assert grad[i] == pytest.approx(num, rel=2e-3, abs=1e-6 * abs(num) + 1e-9)


def test_mem_with_groups_converges_and_matches_group_sum() -> None:
    inp, mult = _nacl_input()
    n = len(inp.hkl)
    idx = np.array([n - 3, n - 2, n - 1])
    g_obs = float(np.sum(mult[idx] * np.abs(inp.f_obs[idx]) ** 2))
    inp.overlap_groups = [(idx, g_obs, 0.02 * g_obs, mult[idx])]
    res = run_mem(inp, max_outer=15)
    assert res.chi2 == pytest.approx(1.0, abs=0.05)
    assert np.all(res.density >= 0)
    # 群の強度和が観測に合い、個々の |F| も真値に近い(密度の正値性と他の反射が決める)
    f2 = np.abs(res.f_calc[idx]) ** 2
    assert float(np.sum(mult[idx] * f2)) == pytest.approx(g_obs, rel=0.05)
    assert np.allclose(np.abs(res.f_calc[idx]), np.abs(inp.f_obs[idx]), rtol=0.3)
    # 群なしの結果と密度がほぼ同じ
    inp2, _ = _nacl_input()
    res2 = run_mem(inp2, max_outer=15)
    assert np.corrcoef(res.density.ravel(), res2.density.ravel())[0, 1] > 0.99
