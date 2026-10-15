"""標準手順の「変位パラメータ」の後処理。

個別に動かした軽原子の U_iso が負か下限なら共通化してやり直す。

COD 2311093(Na2HfSi2O7)で軽原子の割合が閾値をわずかに超えて個別に動かし、O 7 サイトのうち 4 つが
負になった例から(著者は O の U を 1 つに拘束)。
"""

from __future__ import annotations

from pathlib import Path

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.io import read_powder_data
from kaisetsu.refine.recipe import (
    Recipe,
    Step,
    _retie_negative_light_uiso,
    _tie_uiso_group,
    light_atom_uiso_groups,
)

ROOT = Path(__file__).resolve().parents[2]


def _project() -> Project:
    proj = Project("t")
    proj.add_phase(
        Structure.from_cif(str(ROOT / "data/public/pbso4/PbSO4-Wyckoff.cif")), name="PbSO4"
    )
    proj.add_histogram(
        read_powder_data(ROOT / "data/public/pbso4/PBSO4.XRA"),
        CWInstrument(Radiation.xray(1.5405, 1.5443, 0.5), u=0.002, v=-0.002, w=0.004),
        SampleGeometry(),
        n_background=3,
        x_max=60.0,
    )
    proj.build_parameters()
    return proj


def _o_names(proj: Project) -> list[str]:
    return [
        p.name for p in proj.params if p.name.startswith("PbSO4:O") and p.name.endswith(":uiso")
    ]


def test_retie_when_a_light_uiso_is_negative() -> None:
    proj = _project()
    names = _o_names(proj)
    assert len(names) == 3
    proj.vary(*names)
    proj.params[names[0]].value = -0.004
    proj.params[names[1]].value = 0.012
    proj.params[names[2]].value = 0.020
    msgs, again = _retie_negative_light_uiso(proj, Step("adp", ()))
    assert again and len(msgs) == 1 and "共通にしてやり直します" in msgs[0]
    master = next(n for n in names if proj.params[n].tie is None)
    assert all(proj.params[n].tie is not None for n in names if n != master)
    # 共通の初期値は正の値の平均(負は入れない)。共通の U は 0 より下に行かない
    assert abs(proj.params[master].value - 0.016) < 1e-9
    assert proj.params[master].min == 0.0


def test_no_retie_when_all_positive_or_already_tied() -> None:
    proj = _project()
    names = _o_names(proj)
    proj.vary(*names)
    for n in names:
        proj.params[n].value = 0.01
    assert _retie_negative_light_uiso(proj, Step("adp", ())) == ([], False)
    # 既に共通化されていれば何もしない
    tied = _tie_uiso_group(proj, "PbSO4", [n.split(":")[1] for n in names])
    assert tied is not None
    proj.params[tied[0]].value = -0.01
    assert _retie_negative_light_uiso(proj, Step("adp", ())) == ([], False)


def test_light_atom_groups_report_fraction() -> None:
    proj = _project()
    groups = light_atom_uiso_groups(proj)
    assert groups and groups[0][0] == "PbSO4" and set(groups[0][1]) == {"O1", "O2", "O3"}
    assert 0.0 < groups[0][2] < 0.3  # Pb が支配


def test_recipe_runs_after_hook_and_refines_again() -> None:
    proj = _project()
    calls: list[int] = []

    def _after(p: Project, s: Step) -> tuple[list[str], bool]:
        calls.append(len(p.log))
        return (["後処理で見直し"], len(calls) == 1)

    r = Recipe("t").add("bg", ["h0:bkg:*", "h0:PbSO4:scale"], 3, after=_after)
    r.run(proj)
    assert len(calls) == 1
    assert any("後処理で見直し" in line for line in proj.log)
    # 1 回目の精密化のあと after が「やり直し」を返したので、精密化のログが 2 回出る
    assert sum(1 for line in proj.log if line.startswith("精密化:")) == 2
