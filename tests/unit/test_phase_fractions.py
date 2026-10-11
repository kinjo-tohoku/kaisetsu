"""単位胞質量・密度と、多相の重量分率。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import CWInstrument, PowderData, Project, Radiation
from kaisetsu.crystal.cif import read_structure


def test_cell_mass_and_density(diamond_cif: Path, pbso4_cif: Path) -> None:
    dia = read_structure(diamond_cif)
    assert dia.cell_mass() == pytest.approx(8 * 12.011, rel=1e-3)
    assert dia.density() == pytest.approx(3.52, abs=0.02)
    pbso4 = read_structure(pbso4_cif)
    assert pbso4.cell_mass() == pytest.approx(4 * 303.26, rel=1e-3)
    assert pbso4.density() == pytest.approx(6.3, abs=0.2)


def test_phase_fractions(diamond_cif: Path, pbso4_cif: Path) -> None:
    proj = Project()
    proj.add_phase(read_structure(diamond_cif), name="C")
    proj.add_phase(read_structure(pbso4_cif), name="PbSO4")
    x = np.arange(10.0, 60.0, 0.05)
    y = np.full_like(x, 100.0)
    proj.add_histogram(
        PowderData(x, y, 1 / y), CWInstrument(Radiation.xray(1.5406)), n_background=2
    )
    proj.params["h0:C:scale"].value = 2.0
    proj.params["h0:PbSO4:scale"].value = 1.0
    proj.params["h0:C:scale"].esd = 0.1
    proj.params["h0:PbSO4:scale"].esd = 0.0
    fr = proj.phase_fractions(0)
    m_c = 8 * 12.011
    m_p = 4 * 303.26
    w_c = 2.0 * m_c / (2.0 * m_c + m_p)
    assert fr["C"][0] == pytest.approx(w_c, rel=1e-3)
    assert fr["PbSO4"][0] == pytest.approx(1 - w_c, rel=1e-3)
    assert sum(w for w, _ in fr.values()) == pytest.approx(1.0)
    # esd: C のスケールだけ不確かなら両相の σ は等しい(W_C + W_P = 1)
    assert fr["C"][1] == pytest.approx(fr["PbSO4"][1])
    assert fr["C"][1] > 0
    assert "重量分率" in proj.summary()
