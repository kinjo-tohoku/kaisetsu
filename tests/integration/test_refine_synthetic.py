"""合成データのラウンドトリップ: 既知パラメータで作ったパターンにノイズを加え、
初期値をずらして精密化し、真値を 3σ 以内で回収できることを確認する。"""

from __future__ import annotations

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

pytestmark = pytest.mark.slow


def _truth() -> tuple[Structure, CWInstrument, SampleGeometry, ChebyshevBackground, float]:
    # 正方晶の仮想構造(一般位置を含む)で座標・U・格子・プロファイルを同時に試す
    st = Structure(
        Lattice.tetragonal(4.60, 2.96),  # ルチル TiO2
        SpaceGroup.from_symbol("P42/mnm"),
        [
            AtomSite("Ti", "Ti", [0.0, 0.0, 0.0], u_iso=0.006),
            AtomSite("O", "O", [0.305, 0.305, 0.0], u_iso=0.008),
        ],
        name="rutile",
    )
    inst = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.03, u=0.004, v=-0.003, w=0.003, x=0.02, y=0.01,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    sample = SampleGeometry(displacement_mm=0.05)
    bkg = ChebyshevBackground([120.0, -30.0, 10.0, 0.0, 0.0, 0.0], 20.0, 120.0)
    return st, inst, sample, bkg, 25.0


def test_roundtrip_recovers_truth() -> None:
    st, inst, sample, bkg, scale = _truth()
    x = np.arange(20.0, 120.0, 0.02)
    truth = calculate_pattern(x, [PhaseContribution(st, scale=scale)], inst, sample, bkg)
    rng = np.random.default_rng(42)
    y = rng.poisson(truth.y_calc).astype(float)
    data = PowderData(x, y, name="synthetic")

    # 初期値をずらす
    st0 = st.copy()
    st0.lattice = Lattice.tetragonal(4.605, 2.955)
    st0.sites[1].position[:] = [0.300, 0.300, 0.0]
    st0.sites[0].u_iso = 0.01
    st0.sites[1].u_iso = 0.01
    inst0 = CWInstrument(
        Radiation.characteristic("Cu"), zero=0.0, u=0.0, v=0.0, w=0.005, x=0.0, y=0.0,
        s_l=0.015, h_l=0.015, polarization=0.7,
    )  # fmt: skip
    proj = Project("synthetic")
    proj.add_phase(st0, name="rutile")
    proj.add_histogram(data, inst0, SampleGeometry(), n_background=6)
    proj.phase_contribution(0, "rutile").scale = 10.0
    proj.build_parameters()

    # 推奨手順
    proj.vary("h0:bkg:*", "h0:rutile:scale")
    proj.refine(max_cycles=10)
    proj.vary("rutile:cell:*", "h0:inst:zero", "h0:sample:displacement")
    proj.refine(max_cycles=10)
    proj.vary("h0:inst:w", "h0:inst:u", "h0:inst:v", "h0:inst:x", "h0:inst:y")
    proj.refine(max_cycles=15)
    proj.vary("rutile:O:x", "rutile:*:uiso")
    fit = proj.refine(max_cycles=20)
    assert fit.success

    rf = proj.rfactors()["total"]
    assert rf.chi2 < 1.5  # Poisson ノイズのみなので χ² ≈ 1

    p = proj.params
    # (x, x, 0) の拘束が効いている
    assert p["rutile:O:y"].tie is not None
    assert p["rutile:O:y"].value == pytest.approx(p["rutile:O:x"].value)
    assert p["rutile:O:z"].locked

    def within(name: str, true: float, nsig: float = 3.0, floor: float = 0.0) -> None:
        par = p[name]
        assert par.esd is not None and par.esd > 0, name
        tol = max(nsig * par.esd, floor)
        assert abs(par.value - true) < tol, f"{name}: {par.value} vs {true} (esd {par.esd})"

    within("rutile:cell:a", 4.60, floor=1e-4)
    within("rutile:cell:c", 2.96, floor=1e-4)
    within("rutile:O:x", 0.305, floor=2e-4)
    within("rutile:Ti:uiso", 0.006, floor=5e-4)
    within("rutile:O:uiso", 0.008, floor=1e-3)
    within("h0:rutile:scale", scale, floor=0.05)
    # ゼロ点と試料変位は強く相関するが、両者の合成シフトは再現される
    assert abs(p["h0:inst:zero"].value - 0.03) < 0.05
    assert p["rutile:cell:b"].value == pytest.approx(p["rutile:cell:a"].value)

    rep = proj.reflection_report(0, "rutile")
    assert rep.r_bragg < 0.03
    assert len(rep.hkl) > 20
