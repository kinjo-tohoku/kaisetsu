"""プロジェクトの配列を .npz に分けて保存・読み込みできる。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import AtomSite, CWInstrument, Lattice, PDFData, PowderData, Project, Radiation
from kaisetsu.crystal import SpaceGroup, Structure
from kaisetsu.io.project import load_project, save_project


def test_external_arrays_roundtrip(tmp_path: Path) -> None:
    st = Structure(
        Lattice.cubic(4.0), SpaceGroup.from_symbol("P m -3 m"), [AtomSite("Ni", "Ni", [0, 0, 0])]
    )
    proj = Project("npz")
    proj.add_phase(st, name="Ni")
    x = np.arange(10.0, 80.0, 0.02)
    proj.add_histogram(
        PowderData(x, np.full_like(x, 5.0)), CWInstrument(Radiation.xray(1.54)), n_background=2
    )
    r = np.arange(0.01, 10.0, 0.01)
    proj.add_pdf(PDFData(r, np.sin(r), np.full_like(r, 0.1)), "xray", qmax=20.0)
    p = save_project(proj, tmp_path / "a.kaisetsu.json", external_arrays=True)
    npz = tmp_path / "a.npz"
    assert npz.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["histograms"][0]["data"]["x"] == {"$npz": "h0_x"}
    assert d["pdf_datasets"][0]["data"]["dg"] == {"$npz": "g0_dg"}
    assert p.stat().st_size < 20_000  # 配列が JSON に無い
    p2 = load_project(p)
    assert np.allclose(p2.histograms[0].data.x, x)
    assert p2.pdf_datasets[0].data.dg is not None and np.allclose(p2.pdf_datasets[0].data.dg, 0.1)
    # .npz が無いと分かりやすいエラー
    npz.unlink()
    with pytest.raises(FileNotFoundError):
        load_project(p)
    # 既定(内蔵)は従来どおり
    p3 = save_project(proj, tmp_path / "b.json")
    assert not (tmp_path / "b.npz").exists()
    assert len(load_project(p3).histograms[0].data) == len(x)
