"""Brindley 微小吸収補正: τ(x) の性質、線吸収係数、重量分率への適用。"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.api import CWInstrument, PowderData, Project, Radiation
from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.profile.corrections import brindley_factor


def test_brindley_factor_limits() -> None:
    assert brindley_factor(0.0) == pytest.approx(1.0, abs=1e-6)
    x = np.array([0.0, 0.05, 0.2, 0.5])
    tau = brindley_factor(x)
    assert np.all(np.diff(tau) < 0)  # 吸収が強いほど小さい
    assert brindley_factor(-0.1) > 1.0  # 周囲より吸収が弱い相は大きく見える
    # 小さい x では τ ≈ 1 - 2 x <sqrt(1-u²)>(重み 3 u sqrt(1-u²))= 1 - 1.5 x
    assert brindley_factor(0.01) == pytest.approx(1.0 - 1.5 * 0.01, abs=2e-4)
    # Brindley の表(2θ = 90°、μR = 0.1 で τ ≈ 0.86)と数 % 以内
    assert brindley_factor(0.1) == pytest.approx(0.86, abs=0.03)


def test_linear_absorption_coefficient() -> None:
    """Si の Cu Kα 線吸収係数は約 140〜150 /cm(光電吸収のみなので少し小さめ)。

    gemmi の 'F d -3 m' は原点選択 1 なので 8a は (0, 0, 0)。
    """
    si = Structure(
        Lattice.cubic(5.431), SpaceGroup.from_symbol("F d -3 m"),
        [AtomSite("Si", "Si", [0.0, 0.0, 0.0])],
    )  # fmt: skip
    mu = si.linear_absorption_coefficient(1.5406)
    assert 120 < mu < 160
    pb = Structure(
        Lattice.cubic(4.95), SpaceGroup.from_symbol("F m -3 m"), [AtomSite("Pb", "Pb", [0, 0, 0])]
    )
    assert pb.linear_absorption_coefficient(1.5406) > 2000


def test_phase_fractions_with_brindley() -> None:
    """重い相(Pb)の粒子半径を与えると、その相の重量分率が上がる(見かけの強度が吸収で落ちていた分)。"""
    si = Structure(
        Lattice.cubic(5.431), SpaceGroup.from_symbol("F d -3 m"),
        [AtomSite("Si", "Si", [0.0, 0.0, 0.0])], name="Si",
    )  # fmt: skip
    pb = Structure(
        Lattice.cubic(4.95),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Pb", "Pb", [0, 0, 0])],
        name="Pb",
    )
    proj = Project()
    proj.add_phase(si, name="Si")
    proj.add_phase(pb, name="Pb")
    x = np.arange(20.0, 80.0, 0.05)
    proj.add_histogram(
        PowderData(x, np.full_like(x, 100.0)), CWInstrument(Radiation.xray(1.5406)), n_background=2
    )
    proj.params["h0:Si:scale"].value = 1.0
    proj.params["h0:Pb:scale"].value = 1.0
    w0 = proj.phase_fractions(0)
    proj.phases["Pb"].brindley_radius_um = 5.0
    proj.phases["Si"].brindley_radius_um = 5.0
    w1 = proj.phase_fractions(0)
    assert w1["Pb"][0] > w0["Pb"][0]
    assert w1["Pb"][0] + w1["Si"][0] == pytest.approx(1.0)
    # 中性子では補正しない
    proj.histograms[0].instrument.radiation = Radiation.neutron(1.5)
    w2 = proj.phase_fractions(0)
    assert w2["Pb"][0] == pytest.approx(w0["Pb"][0])
