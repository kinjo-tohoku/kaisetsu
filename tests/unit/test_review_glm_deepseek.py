"""GLM / DeepSeek のコードレビュー(2026-10-17)で採用した修正の回帰テスト。

記録: docs/validation/review_glm_deepseek_2026-10-17.md
"""

from __future__ import annotations

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
from kaisetsu.crystal.symmetry import find_symmetry
from kaisetsu.profile.background import InterpolatedBackground
from kaisetsu.refine import cell_scan as cell_scan_mod
from kaisetsu.refine.cell_scan import scan_cell
from kaisetsu.refine.lsq import bounded_step, numerical_jacobian
from kaisetsu.refine.sequential import refine_sequence
from kaisetsu.reflections import canonical_hkl, generate_reflections
from kaisetsu.scattering.xray import XrayScatteringFactors


def _ni_structure(a: float = 4.0) -> Structure:
    return Structure(
        Lattice.cubic(a),
        SpaceGroup.from_symbol("P m -3 m"),
        [AtomSite("Ni", "Ni", [0, 0, 0], u_iso=0.006)],
    )


def _ni_project(x: np.ndarray, y: np.ndarray) -> Project:
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7)
    proj = Project("ni")
    proj.add_phase(_ni_structure(), name="Ni")
    proj.add_histogram(PowderData(x, y), inst, SampleGeometry(), n_background=2)
    proj.phase_contribution(0, "Ni").scale = 40.0
    proj.build_parameters()
    return proj


# ---------------------------------------------------------------- GLM #2, #3 / DeepSeek #12
def test_bounded_step_prefers_forward_then_backward_then_shrinks() -> None:
    assert bounded_step(0.5, 0.1, 0.0, 1.0) == 0.1
    assert bounded_step(0.95, 0.1, 0.0, 1.0) == -0.1
    # 両側に余地が無ければ広い側に縮める
    assert bounded_step(0.0, 1.0, 0.0, 0.5) == 0.5
    assert bounded_step(0.4, 1.0, 0.0, 0.5) == -0.4
    # 上下限が一致していれば動かせない
    assert bounded_step(0.0, 1.0, 0.0, 0.0) == 0.0
    assert bounded_step(0.3, 1e-3, -np.inf, np.inf) == 1e-3


def test_numerical_jacobian_at_lower_bound_with_step_wider_than_range() -> None:
    def fun(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * x[0] + x[1], x[1] ** 2])

    x0 = np.array([0.0, 1.0])
    steps = np.array([1.0, 1e-3])
    lower = np.array([0.0, 0.0])
    # 刻み 1.0 が上下限の幅 0.5 より大きく、現在値が下限ちょうど(以前は 0 で割って NaN)
    j = numerical_jacobian(fun, x0, steps, lower, np.array([0.5, 2.0]))
    assert np.all(np.isfinite(j))
    assert j[0, 0] == pytest.approx(2.0)
    assert j[1, 1] == pytest.approx(2.0, rel=1e-2)
    # 上下限が一致した変数の列は 0
    j2 = numerical_jacobian(fun, x0, steps, lower, np.array([0.0, 2.0]))
    assert np.all(j2[:, 0] == 0.0) and np.all(np.isfinite(j2))


# ---------------------------------------------------------------- GLM #6
def test_scan_cell_does_not_repeat_the_same_span(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.arange(20.0, 80.0, 0.05)
    proj = _ni_project(x, np.full_like(x, 100.0))
    calls: list[float] = []

    def fake_scan(project, params, step, half, progress=None):  # type: ignore[no-untyped-def]
        calls.append(step * half)
        f = 1.0 + step * np.arange(-half, half + 1)
        return f, np.full(len(f), 0.3)  # 改善なし

    monkeypatch.setattr(cell_scan_mod, "_scan_1d", fake_scan)
    monkeypatch.setattr(cell_scan_mod, "analytic_rwp", lambda project: 0.3)  # Rwp が悪い → 広げる
    scan_cell(proj, "Ni", span=0.01)
    assert len(calls) == 2  # ±1% の後に ±3%
    calls.clear()
    scan_cell(proj, "Ni", span=cell_scan_mod.CELL_SCAN_WIDE_SPAN)
    assert len(calls) == 1  # 最初から広ければ 1 回だけ(以前は同じ範囲を 2 回走査していた)


# ---------------------------------------------------------------- GLM #7
def test_refine_sequence_keeps_original_range_per_file(tmp_path: Path) -> None:
    inst = CWInstrument(Radiation.xray(1.5406), w=0.004, s_l=0.01, h_l=0.01, polarization=0.7)
    x_full = np.arange(20.0, 80.0, 0.03)
    bkg = ChebyshevBackground([60.0, -5.0], 20.0, 80.0)
    y_full = calculate_pattern(
        x_full, [PhaseContribution(_ni_structure(), scale=40.0)], inst, SampleGeometry(), bkg
    ).y_calc
    narrow = x_full <= 60.0
    paths = []
    for name, m in (("narrow", narrow), ("full", np.ones_like(narrow))):
        p = tmp_path / f"{name}.xy"
        np.savetxt(p, np.column_stack([x_full[m], y_full[m]]))
        paths.append(p)
    proj = _ni_project(x_full, y_full)
    proj.vary("h0:Ni:scale")
    h = proj.histograms[0]
    x_end = float(x_full[-1])
    assert (h.x_min, h.x_max) == (pytest.approx(20.0), pytest.approx(x_end))
    res = refine_sequence(proj, paths, params=["h0:Ni:scale"], cell_scan=False)
    assert all(r.ok for r in res.rows)
    # 狭いデータで 60° に縮めた範囲を、次の広いデータに引きずらない(以前は 60° のままだった)
    assert h.x_max == pytest.approx(x_end)


# ---------------------------------------------------------------- DeepSeek #1
def test_build_parameters_logs_when_sh_coefficients_mismatch() -> None:
    x = np.arange(20.0, 80.0, 0.05)
    proj = _ni_project(x, np.full_like(x, 100.0))
    proj.enable_texture(0, "Ni", order=4)
    pc = proj.phase_contribution(0, "Ni")
    assert pc.sh_coefficients is not None
    n = len(pc.sh_coefficients)
    pc.sh_coefficients = np.ones(n + 3)  # 古いプロジェクトなどで基底の数が合わない
    proj.build_parameters()
    assert pc.sh_coefficients is not None and len(pc.sh_coefficients) == n
    assert any("球面調和関数の係数の数" in m for m in proj.log)


# ---------------------------------------------------------------- DeepSeek #2, #10, #11
@pytest.mark.parametrize(
    ("symbol", "lattice"),
    [
        ("P6/mmm", Lattice.hexagonal(3.0, 5.0)),
        ("P-3m1", Lattice.hexagonal(3.0, 5.0)),
        ("R-3m", Lattice.hexagonal(5.0, 14.0)),
        ("P4/mmm", Lattice.tetragonal(4.0, 6.0)),
    ],
)
def test_canonical_hkl_handles_rotated_images_beyond_input_range(
    symbol: str, lattice: Lattice
) -> None:
    sg = SpaceGroup.from_symbol(symbol)
    refl = generate_reflections(lattice, sg, 0.8)
    for rot in sg.laue_rotations:
        # 六方晶の h + k などは入力の最大成分を超える。それでも代表に戻ること
        assert np.array_equal(canonical_hkl(sg, refl.hkl @ rot), refl.hkl)


def test_generate_reflections_keeps_boundary_reflection_when_ratio_is_integer() -> None:
    a = 5.1
    # a / d_min が 3 を丸め誤差の分(1e-13)だけ下回る。採否の判定は相対 1e-12 の余裕で
    # (3 0 0) を通すのに、候補の上限 floor(a / d_min) が 2 になって落としていた
    d_min = a / 3 * (1.0 + 1e-13)
    assert a / d_min < 3.0
    refl = generate_reflections(Lattice.cubic(a), SpaceGroup.from_symbol("Pm-3m"), d_min)
    assert refl.index_of(3, 0, 0) >= 0
    assert refl.d.min() == pytest.approx(a / 3)


# ---------------------------------------------------------------- DeepSeek #3
def test_find_symmetry_reduces_anisotropic_u_to_u_equiv() -> None:
    st = Structure(
        Lattice.cubic(5.64),
        SpaceGroup.from_symbol("Fm-3m"),
        [
            AtomSite("Na", "Na1+", [0, 0, 0], u_aniso=[0.02, 0.02, 0.02, 0, 0, 0]),
            AtomSite("Cl", "Cl1-", [0.5, 0.5, 0.5], u_iso=0.015),
        ],
    )
    res = find_symmetry(st)
    assert res.number == 225
    by_species = {s.species: s for s in res.structure.sites}
    assert by_species["Na1+"].u_aniso is None
    assert by_species["Na1+"].u_iso == pytest.approx(0.02)
    assert by_species["Cl1-"].u_iso == pytest.approx(0.015)


# ---------------------------------------------------------------- DeepSeek #6, #13
def test_chebyshev_requires_at_least_one_coefficient() -> None:
    with pytest.raises(ValueError, match="係数"):
        ChebyshevBackground(np.zeros(0), 0.0, 1.0)


def test_interpolated_background_domain_defaults_to_points() -> None:
    b = InterpolatedBackground(np.array([10.0, 20.0, 80.0]), np.array([1.0, 2.0, 3.0]))
    assert (b.x_min, b.x_max) == (10.0, 80.0)
    explicit = InterpolatedBackground(
        np.array([10.0, 20.0, 80.0]), np.array([1.0, 2.0, 3.0]), 5.0, 90.0
    )
    assert (explicit.x_min, explicit.x_max) == (5.0, 90.0)


# ---------------------------------------------------------------- DeepSeek #7
def test_table_specific_labels_resolve_directly() -> None:
    xf = XrayScatteringFactors()
    assert xf.resolve("Cval") == "Cval"
    assert xf.resolve(" Cval ") == "Cval"
    assert np.isfinite(xf.f0("Cval", np.array([0.0, 0.3]))).all()
    assert xf.resolve("Pb2+") == "Pb2+"


# ---------------------------------------------------------------- DeepSeek #8, #9
def test_reciprocal_round_trip_for_skewed_lattice() -> None:
    lat = Lattice(5.0, 6.0, 7.0, 70.0, 95.0, 110.0)
    back = lat.reciprocal.reciprocal
    for name in ("a", "b", "c", "alpha", "beta", "gamma"):
        assert getattr(back, name) == pytest.approx(getattr(lat, name), rel=1e-9)


def test_radiation_rejects_bad_third_wavelength_and_ratios() -> None:
    with pytest.raises(ValueError, match="第 3 の波長"):
        Radiation.xray(1.54, wavelength3=-1.0, ratio3=0.1)
    with pytest.raises(ValueError, match="強度比"):
        Radiation.xray(1.54, 1.544, ratio2=-0.1)
    with pytest.raises(ValueError, match="強度比"):
        Radiation.xray(1.54, 1.544, ratio2=float("nan"))
    ok = Radiation.xray(1.5406, 1.5444, 0.5, 1.3922, 0.01)
    assert len(ok.wavelengths) == 3
