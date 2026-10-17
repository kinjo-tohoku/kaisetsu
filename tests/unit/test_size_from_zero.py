"""結晶子サイズ h{i}:相:size を 0(= 広がりなし)から動かしても真値を回収する。

レビュー 2026-10-17 精密化 #5: 0 から前進差分を取ると 1 歩目が 1 nm の結晶子になって微分が
意味を失い、0.0001 μm に潰れたまま success=True で終わっていた。
"""

from __future__ import annotations

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    Lattice,
    PhaseContribution,
    PowderData,
    Project,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.refine.project import SIZE_START_UM

ST = Structure(
    Lattice.cubic(5.4112),
    SpaceGroup.from_symbol("Fm-3m"),
    [AtomSite("Ce", "Ce", [0, 0, 0], u_iso=0.004), AtomSite("O", "O", [0.25] * 3, u_iso=0.008)],
    name="CeO2",
)


def _inst() -> CWInstrument:
    return CWInstrument(
        Radiation.characteristic("Cu"), u=0.004, v=-0.003, w=0.003, x=0.0, y=0.0,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip


def _project(true_size_um: float) -> Project:
    x = np.arange(25.0, 100.0, 0.03)
    bkg = ChebyshevBackground([100.0, 0, 0, 0], 25.0, 100.0)
    pc = PhaseContribution(ST, scale=0.5)
    pc.size_um = true_size_um
    tr = calculate_pattern(x, [pc], _inst(), SampleGeometry(), bkg)
    y = np.random.default_rng(5).poisson(tr.y_calc).astype(float)
    p = Project("t")
    p.add_phase(ST.copy(), name="CeO2")
    p.add_histogram(
        PowderData(x, y, name="s"),
        _inst(),
        SampleGeometry(),
        background=ChebyshevBackground([100.0, 0, 0, 0], 25.0, 100.0),
    )
    p.phase_contribution(0, "CeO2").scale = 0.5
    p.build_parameters()
    return p


@pytest.mark.parametrize("true_size", [0.03, 0.2])
def test_size_varied_from_zero_recovers_truth(true_size: float) -> None:
    p = _project(true_size)
    size = p.params["h0:CeO2:size"]
    assert size.value == 0.0
    p.vary("h0:CeO2:size", "h0:CeO2:scale", "h0:bkg:0")
    # vary を立てた時点で 0 → 1 μm に直し、ログに残す
    assert size.value == SIZE_START_UM
    assert p.phase_contribution(0, "CeO2").size_um == pytest.approx(SIZE_START_UM)
    assert any("h0:CeO2:size は 0(広がりなし)から動かせない" in line for line in p.log)
    fit = p.refine(max_cycles=20)
    assert fit.success
    assert size.value == pytest.approx(true_size, rel=0.1)
    assert p.rfactors()["total"].rwp < 0.1
    assert not any("まで小さくなりました" in n for n in p.diagnostics())


def test_size_flag_set_directly_is_fixed_at_refine() -> None:
    """GUI の表のようにパラメータの vary を直接立てても、refine の最初で 1 μm から始める。"""
    p = _project(0.05)
    size = p.params["h0:CeO2:size"]
    size.vary = True
    p.params["h0:CeO2:scale"].vary = True
    assert size.value == 0.0
    fit = p.refine(max_cycles=20)
    assert fit.success
    assert any("から始めます" in line for line in p.log)
    assert size.value == pytest.approx(0.05, rel=0.1)


def test_collapsed_size_is_reported() -> None:
    p = _project(0.05)
    size = p.params["h0:CeO2:size"]
    size.value = 0.0005  # 0.5 nm: 幅のモデルが合っていない印
    size.vary = True
    size.apply()
    notes = p.diagnostics()
    assert any("h0:CeO2:size が 0.50 nm まで小さくなりました" in n for n in notes)
