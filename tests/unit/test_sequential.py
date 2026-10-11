"""逐次精密化: 格子定数が少しずつ変わる合成データ列を順に精密化し、表に出す。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
from kaisetsu.refine.sequential import refine_sequence


def test_refine_sequence_tracks_lattice(tmp_path: Path) -> None:
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7)
    x = np.arange(20.0, 80.0, 0.03)
    bkg = ChebyshevBackground([60.0, -5.0], 20.0, 80.0)
    rng = np.random.default_rng(1)
    a_values = [4.000, 4.004, 4.008, 4.012]
    paths = []
    for i, a in enumerate(a_values):
        st = Structure(Lattice.cubic(a), SpaceGroup.from_symbol("P m -3 m"),
                       [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)])  # fmt: skip
        y = calculate_pattern(
            x, [PhaseContribution(st, scale=40.0)], inst, SampleGeometry(), bkg
        ).y_calc
        y = rng.poisson(y).astype(float)
        p = tmp_path / f"T{i}.xy"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(p)
    proj = Project("seq")
    proj.add_phase(
        Structure(Lattice.cubic(4.0), SpaceGroup.from_symbol("P m -3 m"),
                  [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)]),
        name="Ni",
    )  # fmt: skip
    first = PowderData(*np.loadtxt(paths[0]).T)
    proj.add_histogram(first, inst, SampleGeometry(), n_background=2)
    proj.phase_contribution(0, "Ni").scale = 40.0
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:Ni:scale", "Ni:cell:a", "h0:inst:zero")
    msgs: list[str] = []
    res = refine_sequence(proj, paths, params=["Ni:cell:a", "h0:inst:zero"], progress=msgs.append)
    assert len(res.rows) == 4 and all(r.ok for r in res.rows)
    a_fit = res.column("Ni:cell:a")
    assert np.allclose(a_fit, a_values, atol=5e-4)
    assert len(msgs) == 4 and "Rwp" in msgs[0]
    out = res.to_csv(tmp_path / "seq.csv")
    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("name,ok,Rwp,chi2,Ni:cell:a")
    assert len(text) == 5
    assert "T0" in res.summary()
    # 読めないファイルは失敗として記録され、処理は続く
    bad = tmp_path / "broken.xy"
    bad.write_text("garbage\n", encoding="utf-8")
    res2 = refine_sequence(proj, [bad, paths[1]], params=["Ni:cell:a"])
    assert not res2.rows[0].ok and res2.rows[1].ok
