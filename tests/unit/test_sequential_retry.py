"""逐次精密化の振り直し(格子定数の走査 → 再精密化)の失敗の扱い。

レビュー 2026-10-17 差分 #5: 1 回目が成功していても振り直しが例外を出すと行ごと失敗になっていた。
精密化 #13: 2 回とも失敗・悪化したときに発散した値が次のデータの初期値になっていた。
"""

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
from kaisetsu.refine import sequential
from kaisetsu.refine.lsq import FitResult
from kaisetsu.refine.sequential import refine_sequence


def _ni(a: float) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("P m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)],
    )


def _make(
    tmp_path: Path, a_values: tuple[float, ...] = (4.000, 4.004)
) -> tuple[Project, list[Path]]:
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7)
    x = np.arange(20.0, 80.0, 0.05)
    bkg = ChebyshevBackground([60.0, -5.0], 20.0, 80.0)
    rng = np.random.default_rng(1)
    paths = []
    for i, a in enumerate(a_values):
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


def test_retry_exception_keeps_first_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj, paths = _make(tmp_path)
    calls = {"n": 0}
    real_refine = proj.refine
    real_rf = proj.rfactors

    def refine(*args: object, **kwargs: object) -> FitResult:
        calls["n"] += 1
        if calls["n"] == 3:  # 2 つ目のデータの振り直し
            raise RuntimeError("retry blew up")
        return real_refine(*args, **kwargs)  # type: ignore[arg-type]

    def rfactors() -> dict[str, object]:
        d = real_rf()
        if calls["n"] == 2:  # 2 つ目のデータの 1 回目を「大きく悪化」に見せる
            d = {k: dataclasses.replace(v, rwp=v.rwp * 10 + 0.5) for k, v in d.items()}
        return d  # type: ignore[return-value]

    monkeypatch.setattr(proj, "refine", refine)
    monkeypatch.setattr(proj, "rfactors", rfactors)
    monkeypatch.setattr(sequential, "scan_all_cells", lambda project: [])
    res = refine_sequence(proj, paths, params=["Ni:cell:a"], cell_scan=True)
    assert [r.ok for r in res.rows] == [True, True]
    assert "振り直しの精密化でエラー(retry blew up)" in res.rows[1].message
    assert "1 回目の結果を採用" in res.rows[1].message
    assert res.rows[1].values["Ni:cell:a"] == pytest.approx(4.004, abs=5e-4)
    assert res.rows[1].esds["Ni:cell:a"] is not None
    assert "note" in res.to_csv(tmp_path / "s.csv").read_text(encoding="utf-8").splitlines()[0]


def test_both_attempts_failing_restores_previous_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj, paths = _make(tmp_path, (4.000, 4.004, 4.008))
    calls = {"n": 0}
    real_refine = proj.refine

    def refine(*args: object, **kwargs: object) -> FitResult:
        calls["n"] += 1
        if calls["n"] in (2, 3):  # 2 つ目のデータは 1 回目も振り直しも失敗。値は発散させておく
            proj.params["Ni:cell:a"].value = 9.9
            raise RuntimeError("diverged")
        return real_refine(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(proj, "refine", refine)
    monkeypatch.setattr(sequential, "scan_all_cells", lambda project: [])
    seen: list[float] = []
    real_read = sequential.read_powder_data

    def read(path: object) -> PowderData:
        seen.append(proj.params["Ni:cell:a"].value)  # 次のデータを読む時点の初期値
        return real_read(path)  # type: ignore[arg-type]

    monkeypatch.setattr(sequential, "read_powder_data", read)
    res = refine_sequence(proj, paths, params=["Ni:cell:a"], cell_scan=True)
    assert [r.ok for r in res.rows] == [True, False, True]
    assert "diverged" in res.rows[1].message
    # 3 つ目のデータは発散した 9.9 ではなく、1 つ目の結果(≈ 4.000)から始まる
    assert seen[2] == pytest.approx(4.000, abs=2e-3)
    assert res.rows[2].values["Ni:cell:a"] == pytest.approx(4.008, abs=5e-4)


def test_persistently_bad_row_does_not_seed_next_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj, paths = _make(tmp_path, (4.000, 4.004, 4.008))
    real_rf = proj.rfactors
    state = {"bad": False}

    def rfactors() -> dict[str, object]:
        d = real_rf()
        if state["bad"]:  # 2 つ目のデータは振り直しても Rwp が悪いまま(相転移でモデルが合わない)
            d = {k: dataclasses.replace(v, rwp=v.rwp * 10 + 0.5) for k, v in d.items()}
        return d  # type: ignore[return-value]

    monkeypatch.setattr(proj, "rfactors", rfactors)
    monkeypatch.setattr(sequential, "scan_all_cells", lambda project: [])
    seen: list[float] = []
    real_read = sequential.read_powder_data

    def read(path: object) -> PowderData:
        seen.append(proj.params["Ni:cell:a"].value)
        state["bad"] = Path(str(path)) == paths[1]
        return real_read(path)  # type: ignore[arg-type]

    monkeypatch.setattr(sequential, "read_powder_data", read)
    res = refine_sequence(proj, paths, params=["Ni:cell:a"], cell_scan=True)
    assert [r.ok for r in res.rows] == [True, True, True]
    assert (
        "大きく悪化" in res.rows[1].message and "前のデータの値から始めます" in res.rows[1].message
    )
    assert res.rows[1].values["Ni:cell:a"] == pytest.approx(
        4.004, abs=5e-4
    )  # 行の値はこのデータの結果
    assert seen[2] == pytest.approx(4.000, abs=2e-3)  # 次のデータは 1 つ目の値から
    assert res.rows[2].values["Ni:cell:a"] == pytest.approx(4.008, abs=5e-4)
    assert "大きく悪化" in res.summary()
