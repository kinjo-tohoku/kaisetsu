"""CSV の書き出し: 相の名前にカンマがあっても列がずれず、BOM 付き UTF-8 で書く。

(レビュー 2026-10-17 GUI #14)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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
from kaisetsu.io.export import write_pattern_csv, write_reflections_csv

PHASE = "Y3(Fe,Al)5O12"  # カンマ入り


def _project() -> Project:
    st = Structure(
        Lattice.cubic(4.05), SpaceGroup.from_symbol("F m -3 m"), [AtomSite("Al", "Al", [0, 0, 0])]
    )
    proj = Project("csv")
    proj.add_phase(st, name=PHASE)
    x = np.arange(30.0, 50.0, 0.05)
    proj.add_histogram(
        PowderData(x, np.full_like(x, 50.0)), CWInstrument(Radiation.xray(1.5406)), n_background=2
    )
    proj.phase_contribution(0, PHASE).scale = 1.0
    proj.build_parameters()
    return proj


def _rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def test_pattern_csv_quotes_phase_name_and_has_bom(tmp_path: Path) -> None:
    proj = _project()
    out = write_pattern_csv(proj, 0, tmp_path / "calc.csv")
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = _rows(out)
    assert rows[0] == ["two_theta", "y_obs", "y_calc", "y_bkg", "diff", PHASE]
    assert all(len(r) == len(rows[0]) for r in rows[1:])
    assert len(rows) == 1 + len(proj.histograms[0].data)
    # 値は往復できる精度で書かれている
    assert float(rows[1][0]) == proj.histograms[0].data.x[0]
    assert float(rows[1][1]) == 50.0
    raw = out.read_text(encoding="utf-8-sig").splitlines()[0]
    assert raw == 'two_theta,y_obs,y_calc,y_bkg,diff,"Y3(Fe,Al)5O12"'


def test_reflections_csv_quotes_phase_name(tmp_path: Path) -> None:
    proj = _project()
    out = write_reflections_csv(proj, 0, tmp_path / "refl.csv")
    rows = _rows(out)
    assert rows[0][:4] == ["phase", "h", "k", "l"] and len(rows) > 1
    assert all(len(r) == len(rows[0]) for r in rows[1:])
    assert rows[1][0] == PHASE
