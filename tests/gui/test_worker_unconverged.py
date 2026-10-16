"""精密化ワーカーが評価回数の上限で止まったことを「完了」に添えること(外部レビュー 2026-10-17 #5)。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from kaisetsu.api import (
    AtomSite,
    ChebyshevBackground,
    CWInstrument,
    Lattice,
    PhaseContribution,
    PowderData,
    Project,
    Radiation,
    SpaceGroup,
    Structure,
    calculate_pattern,
)
from kaisetsu.gui.worker import RefineWorker


def _rutile_project() -> Project:
    st = Structure(
        Lattice.tetragonal(4.594, 2.959),
        SpaceGroup.from_symbol("P42/mnm"),
        [
            AtomSite("Ti", "Ti", [0, 0, 0], u_iso=0.006),
            AtomSite("O", "O", [0.305, 0.305, 0], u_iso=0.008, occupancy=0.95),
        ],
        name="rutile",
    )
    inst = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.02, u=0.004, v=-0.003, w=0.004, x=0.02, y=0.01,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    x = np.arange(20.0, 100.0, 0.02)
    truth = calculate_pattern(
        x,
        [PhaseContribution(st, scale=30.0, march_dollase_r=0.9)],
        inst,
        background=ChebyshevBackground([100.0, -20.0, 5.0, 0.0], 20.0, 100.0),
    )
    p = Project()
    p.add_phase(st.copy(), name="rutile")
    p.add_histogram(
        PowderData(x, np.random.default_rng(5).poisson(truth.y_calc).astype(float)),
        inst,
        n_background=4,
    )
    p.phase_contribution(0, "rutile").scale = 28.0
    p.phase_contribution(0, "rutile").march_dollase_r = 0.92
    p.build_parameters()
    p.vary(
        "h0:bkg:*", "h0:rutile:scale", "rutile:O:x", "rutile:*:uiso", "rutile:O:occ",
        "h0:rutile:md_r", "rutile:cell:*", "h0:inst:w",
    )  # fmt: skip
    return p


def _run(worker: RefineWorker) -> tuple[bool, str]:
    got: list[tuple[bool, str]] = []
    worker.finished.connect(lambda ok, msg: got.append((ok, msg)))
    worker.run()
    assert len(got) == 1
    return got[0]


def test_worker_mentions_evaluation_cap(qtbot) -> None:  # type: ignore[no-untyped-def]
    p = _rutile_project()
    ok, msg = _run(RefineWorker(p, max_cycles=1))
    assert p.last_fit is not None and not p.last_fit.success
    assert ok is True  # 途中までの値は入っているので「失敗」ではない
    assert msg.startswith("完了: Rwp")
    assert "評価回数の上限" in msg and "精密化を実行" in msg


def test_worker_is_quiet_when_converged(qtbot) -> None:  # type: ignore[no-untyped-def]
    p = Project()
    x = np.linspace(20, 30, 21)
    p.add_histogram(
        PowderData(x, np.full_like(x, 3.0), np.ones_like(x)),
        CWInstrument(Radiation.xray(1.54)),
        background=ChebyshevBackground([1.0], 20, 30),
    )
    p.vary("h0:bkg:0")
    ok, msg = _run(RefineWorker(p, max_cycles=20))
    assert ok is True and p.last_fit is not None and p.last_fit.success
    assert "注意" not in msg and "\n" not in msg
