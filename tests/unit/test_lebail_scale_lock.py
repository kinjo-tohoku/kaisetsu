"""Le Bail / Pawley では相のスケールを固定し、標準手順が構造パラメータを動かさない。

スクロース(GSAS-II チュートリアル、11-BM)でスケールが 1 → 1e-12 に流れて発散した件。
"""

from __future__ import annotations

from pathlib import Path

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.refine.recipe import standard_recipe


def _project(data_dir: Path) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
    )
    proj = Project("lebail-lock")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def _chosen(proj: Project, recipe_step: str) -> set[str]:
    r = standard_recipe("*", "h0")
    step = next(s for s in r.steps if s.name == recipe_step)
    return {
        p.name
        for pat in step.patterns
        for p in proj.params.select(pat)
        if step.filter is None or step.filter(proj, p)
    }


def test_scale_locked_in_le_bail_and_pawley_and_released_after(data_dir: Path) -> None:
    proj = _project(data_dir)
    scale = proj.params["h0:PbSO4:scale"]
    assert not scale.locked
    proj.vary("h0:PbSO4:scale")
    proj.start_le_bail("PbSO4")
    assert scale.locked and not scale.vary and "固定" in scale.description
    # vary を指定しても動かない
    proj.vary("h0:PbSO4:scale")
    assert not scale.is_free
    # 再構築しても固定のまま
    proj.build_parameters()
    assert proj.params["h0:PbSO4:scale"].locked
    proj.start_pawley("PbSO4")
    assert proj.params["h0:PbSO4:scale"].locked
    proj.stop_extraction("PbSO4")
    assert not proj.params["h0:PbSO4:scale"].locked
    assert "固定" not in proj.params["h0:PbSO4:scale"].description


def test_standard_recipe_skips_structure_params_of_extracted_phase(data_dir: Path) -> None:
    proj = _project(data_dir)
    assert "PbSO4:O3:x" in _chosen(proj, "原子座標")
    assert "PbSO4:Pb:uiso" in _chosen(proj, "変位パラメータ")
    proj.start_le_bail("PbSO4")
    assert _chosen(proj, "原子座標") == set()
    assert _chosen(proj, "変位パラメータ") == set()
    # 格子定数は Le Bail でも動かす
    assert "PbSO4:cell:a" in _chosen(proj, "格子定数とピーク位置")


def test_le_bail_refinement_keeps_scale_and_fits(data_dir: Path) -> None:
    """スケールを固定したまま、Le Bail は構造から計算した Rietveld より良く合う。"""
    proj = _project(data_dir)
    pc = proj.phase_contribution(0, "PbSO4")
    res = proj.calculate(0)
    h = proj.histograms[0]
    pc.scale *= float((h.data.y - res.y_background).sum() / (res.y_calc - res.y_background).sum())
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:PbSO4:scale", "PbSO4:cell:*", "h0:inst:zero")
    proj.refine(max_cycles=6)
    proj.vary("h0:inst:w", "h0:inst:u", "h0:inst:v", "h0:inst:x", "h0:inst:y")
    proj.refine(max_cycles=8)
    rietveld = proj.rfactors()["total"].rwp
    proj.start_le_bail("PbSO4")
    before = proj.params["h0:PbSO4:scale"].value
    proj.vary("h0:PbSO4:scale")  # 固定なので効かない
    proj.refine(max_cycles=8, le_bail_cycles=3)
    assert proj.params["h0:PbSO4:scale"].value == before
    lebail = proj.rfactors()["total"].rwp
    # 強度を自由にした分、構造からの Rietveld より悪くはならない(切り出し範囲なので差は小さい)
    assert lebail < rietveld * 1.02
    assert lebail < 0.25
