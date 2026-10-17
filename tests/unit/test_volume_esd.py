"""`refine_lattice` の体積の esd が格子定数どうしの共分散を含むこと(レビュー 2026-10-17 #9)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    ChebyshevBackground,
    CWInstrument,
    PhaseContribution,
    PowderData,
    Radiation,
    SampleGeometry,
    Structure,
    calculate_pattern,
)
from kaisetsu.refine.lattice import refine_lattice


def test_volume_esd_uses_covariance(pbso4_cif: Path) -> None:
    st = Structure.from_cif(str(pbso4_cif))
    inst = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.03, u=0.004, v=-0.003, w=0.003, x=0.02, y=0.01,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    x = np.arange(15.0, 45.0, 0.03)
    tr = calculate_pattern(
        x, [PhaseContribution(st, scale=0.05)], inst, SampleGeometry(),
        ChebyshevBackground([200.0, -30.0, 0, 0], 15.0, 45.0),
    )  # fmt: skip
    y = np.random.default_rng(4).poisson(tr.y_calc).astype(float)
    inst0 = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.0, u=0.004, v=-0.003, w=0.003, x=0.02, y=0.01,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    res = refine_lattice(
        PowderData(x, y, name="s"), [st], inst0, SampleGeometry(), position="zero", widths="sample"
    )
    assert res.ok, res.message
    proj = res.project
    assert proj is not None and proj.last_fit is not None
    fit = proj.last_fit
    cell = proj.phases[res.cells[0].phase]
    cell.sync()
    a, b, c = (proj.params[f"{cell.name}:cell:{n}"].value for n in ("a", "b", "c"))
    names = [f"{cell.name}:cell:{n}" for n in ("a", "b", "c")]
    idx = [fit.names.index(n) for n in names]
    g = np.array([b * c, a * c, a * b])  # 斜方晶: V = abc
    cov = fit.covariance[np.ix_(idx, idx)]
    full = float(np.sqrt(g @ cov @ g))
    diag = float(np.sqrt(np.sum((g * np.sqrt(np.diag(cov))) ** 2)))
    assert res.cells[0].volume == pytest.approx(a * b * c, rel=1e-9)
    assert res.cells[0].volume_esd == pytest.approx(full, rel=1e-6)
    # ゼロ点と一緒に動かすと a, b, c は正に相関するので、相関を無視した値より大きい
    assert abs(float(fit.correlation[idx[0], idx[1]])) > 0.1
    assert res.cells[0].volume_esd != pytest.approx(diag, rel=1e-3)
