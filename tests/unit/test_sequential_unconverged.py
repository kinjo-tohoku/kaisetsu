"""逐次精密化の行に未収束(評価回数の上限)を備考として出すこと(外部レビュー 2026-10-17 #5)。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    Lattice,
    PhaseContribution,
    PowderData,
    Project,
    Radiation,
    SampleGeometry,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.refine.lsq import FitResult, convergence_note
from kaisetsu.refine.sequential import refine_sequence


def _ni(a: float) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("P m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)],
    )


def _make(tmp_path: Path) -> tuple[Project, list[Path]]:
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7)
    x = np.arange(20.0, 80.0, 0.05)
    bkg = ChebyshevBackground([60.0, -5.0], 20.0, 80.0)
    rng = np.random.default_rng(1)
    paths = []
    for i, a in enumerate([4.000, 4.004]):
        y = calculate_pattern(
            x, [PhaseContribution(_ni(a), scale=40.0)], inst, SampleGeometry(), bkg
        ).y_calc
        p = tmp_path / f"T{i}.xy"
        np.savetxt(p, np.column_stack([x, rng.poisson(y).astype(float)]))
        paths.append(p)
    proj = Project("seq")
    proj.add_phase(_ni(4.0), name="Ni")
    proj.add_histogram(PowderData(*np.loadtxt(paths[0]).T), inst, SampleGeometry(), n_background=2)
    proj.phase_contribution(0, "Ni").scale = 40.0
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:Ni:scale", "Ni:cell:a")
    return proj, paths


def test_convergence_note_texts() -> None:
    assert convergence_note(None) == ""
    fit = FitResult(
        names=[], values=np.zeros(0), esd=np.zeros(0), covariance=np.zeros((0, 0)),
        correlation=np.zeros((0, 0)), chi2=1.0, n_data=1, n_params=0, n_evaluations=1,
        success=True, message="`ftol` termination condition is satisfied.", elapsed=0.0,
    )  # fmt: skip
    assert convergence_note(fit) == ""
    # 上限で止まったかは scipy のメッセージ文字列ではなく status(0 = max_nfev)で判定する
    capped = dataclasses.replace(fit, success=False, status=0, message="(文言は問わない)")
    assert capped.hit_evaluation_limit
    assert "評価回数の上限" in convergence_note(capped)
    assert convergence_note(capped, short=True) == "未収束(評価回数の上限で停止)"
    other = dataclasses.replace(fit, success=False, status=-1, message="xyz")
    assert not other.hit_evaluation_limit
    assert convergence_note(other, short=True) == "未収束(xyz)"


def test_sequence_row_notes_unconverged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj, paths = _make(tmp_path)
    real_refine = proj.refine

    def capped_refine(*args: object, **kwargs: object) -> FitResult:
        fit = real_refine(*args, **kwargs)  # type: ignore[arg-type]
        proj.last_fit = dataclasses.replace(fit, success=False, status=0)
        return proj.last_fit

    monkeypatch.setattr(proj, "refine", capped_refine)
    res = refine_sequence(proj, paths, params=["Ni:cell:a"], cell_scan=False)
    assert len(res.rows) == 2 and all(r.ok for r in res.rows)
    assert all("未収束(評価回数の上限で停止)" in r.message for r in res.rows)
    # 備考は CSV の note 列と summary にも出る(レビュー 2026-10-17 差分 #2)
    csv_path = res.to_csv(tmp_path / "seq.csv")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith(",note")
    assert all("未収束(評価回数の上限で停止)" in line for line in lines[1:])
    assert "未収束(評価回数の上限で停止)" in res.summary()
    # Project.diagnostics() にも入る(結果の要約・CLI がこれを出す)
    assert any("評価回数の上限" in n for n in proj.diagnostics())


def test_sequence_row_quiet_when_converged(tmp_path: Path) -> None:
    proj, paths = _make(tmp_path)
    res = refine_sequence(proj, paths, params=["Ni:cell:a"], cell_scan=False)
    assert len(res.rows) == 2 and all(r.ok for r in res.rows)
    assert all("未収束" not in r.message for r in res.rows)
