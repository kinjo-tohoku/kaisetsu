"""プロジェクトの保存は途中で失敗しても元のファイルを壊さない。古い版の注意。独立なコピー。

(レビュー 2026-10-17 GUI #2、差分 #3、GUI #4)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    CWInstrument,
    Lattice,
    PowderData,
    Project,
    Radiation,
    SpaceGroup,
    Structure,
)
from kaisetsu.io import project as project_io
from kaisetsu.io.project import (
    clone_project,
    compatibility_notes,
    load_project,
    load_project_with_notes,
    project_to_dict,
    save_project,
)


def _project() -> Project:
    st = Structure(
        Lattice.cubic(4.0), SpaceGroup.from_symbol("P m -3 m"), [AtomSite("Ni", "Ni", [0, 0, 0])]
    )
    proj = Project("atomic")
    proj.add_phase(st, name="Ni")
    x = np.arange(10.0, 80.0, 0.05)
    proj.add_histogram(
        PowderData(x, np.full_like(x, 5.0)), CWInstrument(Radiation.xray(1.54)), n_background=2
    )
    return proj


def test_failed_save_keeps_original_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = _project()
    path = tmp_path / "a.kaisetsu.json"
    save_project(proj, path, external_arrays=True)
    npz = tmp_path / "a.npz"
    before = path.read_bytes()
    before_npz = npz.read_bytes()

    def boom(*_a: object, **_k: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(project_io.json, "dumps", boom)
    with pytest.raises(OSError, match="disk full"):
        save_project(proj, path)
    assert path.read_bytes() == before, "失敗した保存が元のファイルを切り詰めた"
    monkeypatch.undo()

    def boom_npz(*_a: object, **_k: object) -> None:
        raise OSError("disk full npz")

    monkeypatch.setattr(project_io.np, "savez_compressed", boom_npz)
    with pytest.raises(OSError, match="disk full npz"):
        save_project(proj, path, external_arrays=True)
    assert npz.read_bytes() == before_npz and path.read_bytes() == before
    monkeypatch.undo()
    # 一時ファイルが残らない
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.kaisetsu.json", "a.npz"]
    # 成功すれば置き換わる
    proj.name = "changed"
    save_project(proj, path)
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "changed"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.kaisetsu.json", "a.npz"]


def _tilted_perovskite() -> Project:
    """I4/mcm の 8h (x, x+1/2, 0): y が x に並進付きで拘束されるサイトを持つ。"""
    st = Structure(
        Lattice.tetragonal(5.5, 7.8),
        SpaceGroup.from_symbol("I4/mcm"),
        [AtomSite("Sr", "Sr", [0, 0.5, 0.25]), AtomSite("O2", "O", [0.22, 0.72, 0.0])],
    )
    proj = Project("sto")
    proj.add_phase(st, name="STO")
    tie = proj.params["STO:O2:y"].tie
    assert tie is not None and tie.offset == pytest.approx(0.5)
    return proj


def test_old_project_with_translation_tie_is_flagged(tmp_path: Path) -> None:
    proj = _tilted_perovskite()
    d = project_to_dict(proj)
    assert compatibility_notes(dict(d, version="0.1.13"), proj) == []
    assert compatibility_notes(dict(d, version="0.2.0"), proj) == []
    notes = compatibility_notes(dict(d, version="0.1.12"), proj)
    assert len(notes) == 1 and "STO:O2" in notes[0] and "CIF から読み直して" in notes[0]
    assert "0.1.12" in notes[0]
    # 並進の無い拘束(例: P m -3 m の一般位置や y = x)だけなら注意しない
    plain = _project()
    assert compatibility_notes(dict(project_to_dict(plain), version="0.1.5"), plain) == []
    # ファイル経由: load_project_with_notes が注意を返し、Project.log にも残す
    path = tmp_path / "old.kaisetsu.json"
    path.write_text(json.dumps(dict(d, version="0.1.12"), ensure_ascii=False), encoding="utf-8")
    loaded, got = load_project_with_notes(path)
    assert got == notes and loaded.log[-1] == "注意: " + notes[0]
    assert load_project(path).log[-1].startswith("注意:")
    new = tmp_path / "new.kaisetsu.json"
    save_project(proj, new)
    assert load_project_with_notes(new)[1] == []


def test_clone_project_is_independent() -> None:
    proj = _project()
    proj.vary("h0:bkg:*", "Ni:cell:a")
    proj.params["Ni:cell:a"].value = 4.1
    proj.params.apply_all()
    clone = clone_project(proj)
    assert clone is not proj and clone.params.free_names() == proj.params.free_names()
    assert clone.params["Ni:cell:a"].value == pytest.approx(4.1)
    # コピー側を書き換えても元のモデル・データは動かない(setter も共有していない)
    clone.params["Ni:cell:a"].value = 4.5
    clone.params.apply_all()
    clone.phases["Ni"].sync()
    clone.histograms[0].data.y[:] = 0.0
    clone.histograms[0].set_excluded([(20.0, 30.0)])
    assert proj.params["Ni:cell:a"].value == pytest.approx(4.1)
    assert proj.phases["Ni"].structure.lattice.a == pytest.approx(4.1)
    assert np.all(proj.histograms[0].data.y == 5.0) and proj.histograms[0].excluded == []
