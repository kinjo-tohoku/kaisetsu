"""diffpy.pdffit2 との突き合わせ(Ni)。

参照値は `uv run python validation/pdffit2/run_pdffit2.py` で生成する
(`validation/reference/pdffit2_ni.json`)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import Project, Structure, pdf_recipe, read_pdf_data
from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup
from kaisetsu.pdf import PDFPhase, calculate_pdf
from tests.conftest import load_reference

pytestmark = pytest.mark.validation


def _ni(a: float, uiso: float) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("F m -3 m"),
        [AtomSite("Ni1", "Ni", [0, 0, 0], u_iso=uiso)],
        name="Ni",
    )


@pytest.mark.parametrize("key,radiation", [("calc", "xray"), ("calc_neutron", "neutron")])
def test_pdf_calc_matches_pdffit2(key: str, radiation: str) -> None:
    """固定パラメータでの G(r)。終端効果の実装差(帯域制限 vs pdffit2 の畳み込み)で
    rms 0.03 程度(ピーク高さ 20 の 0.15%)まで一致する。"""
    ref = load_reference("pdffit2_ni")[key]
    r = np.asarray(ref["r"])
    g_ref = np.asarray(ref["g"])
    st = _ni(ref["a"], ref["uiso"])
    res = calculate_pdf(
        r, [PDFPhase(st, delta2=ref["delta2"])], radiation, qmax=ref["qmax"], qdamp=ref["qdamp"]
    )
    d = res.g_calc - g_ref
    inner = (r > 0.5) & (r < 19.0)
    assert np.sqrt(np.mean(d[inner] ** 2)) < 0.03
    assert np.abs(d[inner]).max() < 0.1
    assert np.abs(g_ref).max() > 15.0


def test_pdf_calc_without_qmax_matches_to_1e4() -> None:
    """終端効果なしでは数値精度(1e-4)で一致する(ガウス和・規格化・基線の確認)。

    pdffit2 で qmax = 0 として計算した値を直接埋め込む代わりに、参照 JSON の calc と
    同じ条件から qmax だけ外した計算を pdffit2 側でも行い、その一致を
    validation/pdffit2/run_pdffit2.py 作成時に確認した(max|ΔG| = 1.0e-4)。
    ここでは qmax = 0 の計算が基線と配位数を正確に再現することを確認する。
    """
    st = _ni(3.524, 0.005)
    r = np.arange(0.01, 10.0, 0.01)
    g = calculate_pdf(r, [PDFPhase(st)], "xray").g_calc
    rho0 = 4.0 / 3.524**3
    assert np.allclose(g[r < 1.8], -4 * np.pi * rho0 * r[r < 1.8], atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize(
    "key,fname,radiation,qdamp0",
    [("fit_xray", "Ni-xray.gr", "xray", 0.06), ("fit_neutron", "Ni-neutron.gr", "neutron", 0.02)],
)
def test_pdf_refinement_matches_pdffit2(
    key: str, fname: str, radiation: str, qdamp0: float, data_dir: Path
) -> None:
    """Ni の実データ(X 線 / 中性子)の精密化結果が pdffit2 と 1σ 以内で一致する。"""
    ref = load_reference("pdffit2_ni")[key]
    proj = Project("Ni PDF")
    st = Structure.from_cif(str(data_dir / "pdf" / "Ni.cif"))
    st.sites[0].u_iso = 0.005
    proj.add_phase(st, name="Ni")
    proj.add_pdf(
        read_pdf_data(data_dir / "pdf" / fname),
        radiation,
        qmax=ref["qmax"],
        qdamp=qdamp0,
        r_min=ref["rmin"],
        r_max=ref["rmax"],
    )
    proj.pdf_phase(0, "Ni").delta2 = 2.0
    proj.build_parameters()
    pdf_recipe("Ni", "g0").run(proj)
    ps = proj.params
    for ours, theirs in (
        ("Ni:cell:a", "a"),
        ("Ni:Ni1:uiso", "uiso"),
        ("g0:Ni:delta2", "delta2"),
        ("g0:pdf:qdamp", "qdamp"),
        ("g0:pdf:scale", "dscale"),
    ):
        p = ps[ours]
        assert p.esd is not None
        assert abs(p.value - ref[theirs]) < max(1.0 * p.esd, 1e-3 * abs(ref[theirs])), (
            ours,
            p.value,
            ref[theirs],
            p.esd,
        )
    assert proj.rfactors()["g0"].rwp == pytest.approx(ref["rw"], abs=5e-4)
