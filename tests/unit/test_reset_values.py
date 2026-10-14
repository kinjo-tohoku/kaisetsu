"""初期値の記録と「戻す」操作(Project.snapshot / restore / reset_to_initial、Parameter.initial)。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.io.project import load_project, save_project


def _project(data_dir: Path) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(20.0, 40.0)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
    )
    proj = Project("reset")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=3)
    proj.build_parameters()
    return proj


def test_initial_is_recorded_and_reset_returns_to_it(data_dir: Path) -> None:
    proj = _project(data_dir)
    assert all(p.initial == p.value for p in proj.params)
    init = proj.snapshot()
    proj.vary("h0:bkg:*", "h0:PbSO4:scale")
    proj.refine(max_cycles=2)
    scale = proj.params["h0:PbSO4:scale"]
    assert scale.value != init["h0:PbSO4:scale"] and scale.esd is not None
    # 再構築しても初期値は残る(値は精密化後のまま)
    proj.build_parameters()
    assert proj.params["h0:PbSO4:scale"].initial == init["h0:PbSO4:scale"]
    assert proj.params["h0:PbSO4:scale"].value == scale.value
    names = proj.reset_to_initial()
    assert set(names) >= {"h0:PbSO4:scale", "h0:bkg:0", "PbSO4:cell:a"}
    assert proj.snapshot() == init
    assert proj.params["h0:PbSO4:scale"].esd is None
    assert proj.last_fit is None


def test_setup_changes_become_the_new_initial(data_dir: Path) -> None:
    proj = _project(data_dir)
    h = proj.histograms[0]
    # 装置ダイアログ相当: パラメータ系の外で値が変わった
    h.instrument = replace(h.instrument, w=0.009)
    proj.build_parameters()
    assert proj.params["h0:inst:w"].initial == 0.009
    # 範囲を変えると背景の係数が書き直され、その値が初期値になる
    proj.set_range(0, 22.0, 38.0)
    p = proj.params["h0:bkg:0"]
    assert p.initial == p.value
    # 背景の推定し直しも同じ
    proj.reset_background(0, n_terms=4, estimate=True)
    assert all(q.initial == q.value for q in proj.params.select("h0:bkg:*"))


def test_restore_and_partial_reset(data_dir: Path) -> None:
    proj = _project(data_dir)
    snap = proj.snapshot()
    proj.params["h0:inst:zero"].value = 0.05
    proj.params["PbSO4:cell:a"].value = snap["PbSO4:cell:a"] + 0.01
    proj.params.apply_all()
    proj.mark_initial(["PbSO4:cell:a"])  # 手で入れた値を初期値にする
    assert proj.reset_to_initial(["h0:inst:zero"]) == ["h0:inst:zero"]
    assert proj.params["h0:inst:zero"].value == snap["h0:inst:zero"]
    assert proj.params["PbSO4:cell:a"].value == snap["PbSO4:cell:a"] + 0.01
    assert (
        proj.reset_to_initial() and proj.params["PbSO4:cell:a"].value == snap["PbSO4:cell:a"] + 0.01
    )
    proj.restore(snap)
    assert proj.snapshot() == snap
    # 拘束されたパラメータは restore の対象外(拘束から計算される)
    proj.tie("h0:inst:v", "h0:inst:u")
    assert proj.restore({"h0:inst:v": 1.0}) == []
    assert proj.params["h0:inst:v"].value == proj.params["h0:inst:u"].value


def test_initial_survives_save_and_load(data_dir: Path, tmp_path: Path) -> None:
    proj = _project(data_dir)
    init = proj.snapshot()
    proj.vary("h0:bkg:*", "h0:PbSO4:scale")
    proj.refine(max_cycles=2)
    path = save_project(proj, tmp_path / "reset.kaisetsu.json")
    q = load_project(path)
    assert q.params["h0:PbSO4:scale"].value == proj.params["h0:PbSO4:scale"].value
    assert q.params["h0:PbSO4:scale"].initial == init["h0:PbSO4:scale"]
    q.reset_to_initial()
    assert q.params["h0:PbSO4:scale"].value == init["h0:PbSO4:scale"]
