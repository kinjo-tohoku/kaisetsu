"""Kβ 線(フィルタ漏れ)の扱い: 波長表、パターンへの寄与、パラメータ、保存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
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
from kaisetsu.io.project import load_project, save_project
from kaisetsu.scattering.radiation import KBETA_WAVELENGTHS


def _si() -> Structure:
    return Structure(
        Lattice.cubic(5.431), SpaceGroup.from_symbol("F d -3 m"),
        [AtomSite("Si", "Si", [0, 0, 0], u_iso=0.005)],
    )  # fmt: skip


def test_radiation_with_kbeta() -> None:
    rad = Radiation.characteristic("Cu", kbeta=True, ratio3=0.02)
    assert rad.wavelength3 == pytest.approx(KBETA_WAVELENGTHS["Cu"])
    assert rad.wavelength3 == pytest.approx(1.39225, abs=1e-4)
    assert len(rad.wavelengths) == 3 and rad.wavelengths[2][1] == pytest.approx(0.02)
    assert len(Radiation.characteristic("Cu").wavelengths) == 2
    assert len(Radiation.xray(1.54, wavelength3=1.39, ratio3=0.0).wavelengths) == 1


def test_kbeta_adds_peak_at_lower_angle() -> None:
    st = _si()
    inst0 = CWInstrument(Radiation.characteristic("Cu", kalpha2=False), w=0.004)
    inst1 = CWInstrument(
        Radiation.characteristic("Cu", kalpha2=False, kbeta=True, ratio3=0.05), w=0.004
    )
    x = np.arange(20.0, 40.0, 0.01)
    y0 = calculate_pattern(x, [PhaseContribution(st, scale=100.0)], inst0).y_calc
    y1 = calculate_pattern(x, [PhaseContribution(st, scale=100.0)], inst1).y_calc
    # Si 111: Kα1 で 28.44°、Kβ で 25.66°
    k_beta = (x > 25.3) & (x < 26.0)
    assert y0[k_beta].max() < 1.0 and y1[k_beta].max() > 10.0
    assert y1[k_beta].sum() / y1[(x > 28.0) & (x < 29.0)].sum() == pytest.approx(0.05, rel=0.25)


def test_kbeta_parameter_and_roundtrip(tmp_path: Path) -> None:
    st = _si()
    x = np.arange(20.0, 60.0, 0.02)
    proj = Project()
    proj.add_phase(st, name="Si")
    proj.add_histogram(
        PowderData(x, np.full_like(x, 10.0)),
        CWInstrument(Radiation.characteristic("Cu", kbeta=True, ratio3=0.01), w=0.004),
        SampleGeometry(),
        n_background=2,
    )
    p = proj.params["h0:inst:kb_ratio"]
    assert p.value == pytest.approx(0.01)
    p.value = 0.03
    proj.params.apply_all()
    assert proj.histograms[0].instrument.radiation.ratio3 == pytest.approx(0.03)
    path = save_project(proj, tmp_path / "k.json")
    p2 = load_project(path)
    rad = p2.histograms[0].instrument.radiation
    assert rad.wavelength3 == pytest.approx(
        KBETA_WAVELENGTHS["Cu"]
    ) and rad.ratio3 == pytest.approx(0.03)
