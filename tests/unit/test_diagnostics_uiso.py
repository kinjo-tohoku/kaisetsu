"""診断: 同じ相の他のサイトに比べて U_iso が目立って大きいサイトを知らせる。

相見(電気化学 94, 264 (2026))の PbNiO3 の例(Pnma で解くと O の B が 3.9 Å² と異常)に倣い、
最小の U_iso の 4 倍を超え 0.03 Å² より大きいサイトだけを出す。
"""

from __future__ import annotations

from pathlib import Path

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.io import read_powder_data

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
    )
    proj.build_parameters()
    return proj


def _uiso_names(proj: Project) -> list[str]:
    return [p.name for p in proj.params if p.name.endswith(":uiso")]


def test_uiso_outlier_is_reported() -> None:
    proj = _project()
    names = _uiso_names(proj)
    for n in names:
        proj.params[n].value = 0.012
    proj.params[names[0]].value = 0.06  # B ≈ 4.7 Å²
    notes = proj._uiso_outliers()
    assert len(notes) == 1 and names[0] in notes[0]
    assert "空間群" in notes[0] and "B ≈ 4.7" in notes[0]
    assert any(names[0] in line and "目立って大きい" in line for line in proj.diagnostics())


def test_normal_spread_is_quiet() -> None:
    """軽原子の U が重原子の 2〜3 倍なのは普通なので何も言わない。"""
    proj = _project()
    for n in _uiso_names(proj):
        proj.params[n].value = 0.03 if n.split(":")[1].startswith("O") else 0.012
    assert proj._uiso_outliers() == []
    # 全部大きい(有機物など)ときも相対的には揃っているので何も言わない
    for n in _uiso_names(proj):
        proj.params[n].value = 0.05
    assert proj._uiso_outliers() == []
