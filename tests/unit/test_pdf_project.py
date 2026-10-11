"""Project への PDF データセット組み込み: パラメータ、残差、合成データの回収、保存と読み込み。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    CWInstrument,
    PDFData,
    PDFPhase,
    PowderData,
    Project,
    Radiation,
    Structure,
    calculate_pdf,
    pdf_recipe,
)
from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup
from kaisetsu.io.project import load_project, save_project


def _ni(a: float = 3.524, uiso: float = 0.005) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni1", "Ni", [0, 0, 0], u_iso=uiso)],
        name="Ni",
    )


def _synthetic(seed: int = 1) -> PDFData:
    r = np.arange(0.01, 15.0, 0.01)
    truth = calculate_pdf(
        r, [PDFPhase(_ni(3.530, 0.006), delta2=2.0)], "xray", qmax=25.0, qdamp=0.05, scale=0.8
    )
    rng = np.random.default_rng(seed)
    dg = np.full_like(r, 0.05)
    return PDFData(r, truth.g_calc + rng.normal(0, 0.05, len(r)), dg, name="synthetic")


def test_parameters_and_residuals() -> None:
    proj = Project()
    proj.add_phase(_ni(), name="Ni")
    g = proj.add_pdf(_synthetic(), "xray", qmax=25.0, qdamp=0.05, r_min=1.5, r_max=12.0)
    names = proj.params.names()
    for n in ("g0:pdf:scale", "g0:pdf:qdamp", "g0:pdf:qbroad", "g0:Ni:scale", "g0:Ni:delta2"):
        assert n in names
    assert g.qmax == 25.0
    res = proj.residuals()
    assert len(res) == int(np.count_nonzero(g.mask()))
    rf = proj.rfactors()
    assert "g0" in rf and "total" in rf
    assert "PDF g0" in proj.summary()
    # 原子対リストは構造が変わらなければ使い回され、変わると作り直される
    proj.calculate_pdf(0)
    cached = g._pair_cache["Ni"][2]
    proj.calculate_pdf(0)
    assert g._pair_cache["Ni"][2] is cached
    proj.params["Ni:cell:a"].value = 3.53
    proj.params.apply_all()
    proj.calculate_pdf(0)
    assert g._pair_cache["Ni"][2] is not cached


def test_recovers_truth_from_synthetic() -> None:
    proj = Project()
    proj.add_phase(_ni(3.524, 0.005), name="Ni")
    proj.add_pdf(_synthetic(), "xray", qmax=25.0, qdamp=0.03, r_min=1.5, r_max=14.0)
    proj.pdf_phase(0, "Ni").delta2 = 1.0
    proj.build_parameters()
    pdf_recipe("Ni", "g0").run(proj)
    ps = proj.params
    assert ps["Ni:cell:a"].value == pytest.approx(3.530, abs=3 * (ps["Ni:cell:a"].esd or 1e-3))
    assert ps["Ni:Ni1:uiso"].value == pytest.approx(0.006, abs=0.0005)
    assert ps["g0:Ni:delta2"].value == pytest.approx(2.0, abs=0.4)
    assert ps["g0:pdf:qdamp"].value == pytest.approx(0.05, abs=0.005)
    assert ps["g0:pdf:scale"].value == pytest.approx(0.8, abs=0.02)
    assert proj.rfactors()["g0"].chi2 < 1.5


def test_combined_powder_and_pdf() -> None:
    """粉末パターンと PDF を同時に持てる(残差は連結される)。"""
    proj = Project()
    proj.add_phase(_ni(), name="Ni")
    x = np.arange(20.0, 80.0, 0.05)
    proj.add_histogram(
        PowderData(x, np.full_like(x, 50.0)), CWInstrument(Radiation.xray(1.5406)), n_background=2
    )
    proj.add_pdf(_synthetic(), "xray", qmax=25.0, r_min=1.5, r_max=10.0)
    n_h = int(np.count_nonzero(proj.histograms[0].mask()))
    n_g = int(np.count_nonzero(proj.pdf_datasets[0].mask()))
    assert len(proj.residuals()) == n_h + n_g
    rf = proj.rfactors()
    assert set(rf) >= {"g0", "total", proj.histograms[0].name}


def test_pdf_roundtrip(tmp_path: Path) -> None:
    proj = Project("pdf io")
    proj.add_phase(_ni(), name="Ni")
    proj.add_pdf(_synthetic(), "neutron", qmax=25.0, qdamp=0.04, r_min=1.2, r_max=11.0)
    proj.pdf_phase(0, "Ni").delta2 = 1.7
    proj.pdf_phase(0, "Ni").spdiameter = 50.0
    proj.build_parameters()
    proj.vary("g0:pdf:scale")
    path = save_project(proj, tmp_path / "p.kaisetsu.json")
    p2 = load_project(path)
    g = p2.pdf_datasets[0]
    assert g.radiation == "neutron" and g.qmax == 25.0 and g.qdamp == 0.04
    assert (g.r_min, g.r_max) == (1.2, 11.0)
    assert g.phases["Ni"].delta2 == 1.7 and g.phases["Ni"].spdiameter == 50.0
    assert g.data.dg is not None
    assert p2.params["g0:pdf:scale"].vary
    assert p2.rfactors()["g0"].rwp == pytest.approx(proj.rfactors()["g0"].rwp)
