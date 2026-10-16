"""粉末強度の |F|² は Friedel 対の平均(外部レビュー 2026-10-17 #2 の回帰テスト)。

非中心対称 + 異常分散では |F(hkl)|² ≠ |F(-h-k-l)|² なので、代表 1 本の |F|² を使うと
全原子座標を反転した(物理的に同じ粉末の)構造でパターンが変わってしまう。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    CWInstrument,
    Lattice,
    PhaseContribution,
    Radiation,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.reflections.structure_factor import (
    powder_structure_factor_squared,
    structure_factors,
)

HKL = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [3, 3, 1]])


def _zns() -> Structure:
    return Structure(
        Lattice.cubic(5.41),
        SpaceGroup.from_symbol("F -4 3 m"),
        [AtomSite("Zn1", "Zn", [0, 0, 0]), AtomSite("S1", "S", [0.25, 0.25, 0.25])],
        name="ZnS",
    )


def _inverted(st: Structure) -> Structure:
    inv = st.copy()
    for site in inv.sites:
        site.position[:] = (-site.position) % 1.0
    return inv


def test_powder_f2_is_friedel_average_for_noncentrosymmetric() -> None:
    st = _zns()
    rad = Radiation.characteristic("Cu", kalpha2=False)
    f2 = powder_structure_factor_squared(st, HKL, rad)
    f_plus = np.abs(structure_factors(st, HKL, rad)) ** 2
    f_minus = np.abs(structure_factors(st, -HKL, rad)) ** 2
    # 111 では Friedel 則が破れている(差が 1% 以上)ことを確かめた上で、平均に一致する
    assert abs(f_plus[0] - f_minus[0]) / f_plus[0] > 0.01
    assert np.allclose(f2, 0.5 * (f_plus + f_minus), rtol=1e-12)


def test_centrosymmetric_and_neutron_unchanged() -> None:
    rutile = Structure(
        Lattice.tetragonal(4.594, 2.959),
        SpaceGroup.from_symbol("P42/mnm"),
        [AtomSite("Ti", "Ti", [0, 0, 0], u_iso=0.006), AtomSite("O", "O", [0.305, 0.305, 0])],
        name="rutile",
    )
    rad = Radiation.characteristic("Cu", kalpha2=False)
    f2 = powder_structure_factor_squared(rutile, HKL, rad)
    assert np.allclose(f2, np.abs(structure_factors(rutile, HKL, rad)) ** 2, rtol=1e-12)
    # 中性子(f'' 無し)では非中心対称でも |F|² と同じ
    n = Radiation.neutron(1.5)
    assert np.allclose(
        powder_structure_factor_squared(_zns(), HKL, n),
        np.abs(structure_factors(_zns(), HKL, n)) ** 2,
        rtol=1e-12,
    )


def test_inverted_structure_gives_same_powder_pattern() -> None:
    st = _zns()
    x = np.linspace(20.0, 80.0, 1201)
    inst = CWInstrument(Radiation.characteristic("Cu", kalpha2=False), w=0.01)
    r = calculate_pattern(x, [PhaseContribution(st)], inst)
    ri = calculate_pattern(x, [PhaseContribution(_inverted(st))], inst)
    assert np.allclose(r.y_calc, ri.y_calc, rtol=1e-10, atol=1e-10 * r.y_calc.max())
    tbl = r.peaks[0]
    j = int(np.flatnonzero(np.all(tbl.hkl == [1, 1, 1], axis=1))[0])
    rad = inst.radiation
    pair = np.abs(structure_factors(st, [[1, 1, 1], [-1, -1, -1]], rad)) ** 2
    assert tbl.f2[j] == pytest.approx(float(pair.mean()), rel=1e-12)
