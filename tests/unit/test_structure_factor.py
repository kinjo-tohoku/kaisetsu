import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.reflections import generate_reflections, structure_factors
from kaisetsu.scattering import NeutronScatteringLengths, Radiation, XrayScatteringFactors

NEUTRON = Radiation.neutron(1.5)


def _nacl() -> Structure:
    return Structure(
        Lattice.cubic(5.64),
        SpaceGroup.from_symbol("Fm-3m"),
        [AtomSite("Na", "Na", [0, 0, 0]), AtomSite("Cl", "Cl", [0.5, 0.5, 0.5])],
        name="NaCl",
    )


def test_nacl_neutron_analytic() -> None:
    st = _nacl()
    nt = NeutronScatteringLengths()
    b_na, b_cl = nt.scattering_length("Na").real, nt.scattering_length("Cl").real
    f = structure_factors(st, [[2, 0, 0], [1, 1, 1], [2, 2, 0], [3, 1, 1]], NEUTRON)
    assert f[0].real == pytest.approx(4 * (b_na + b_cl))
    assert f[1].real == pytest.approx(4 * (b_na - b_cl))
    assert f[2].real == pytest.approx(4 * (b_na + b_cl))
    assert f[3].real == pytest.approx(4 * (b_na - b_cl))
    assert np.allclose(f.imag, 0.0)


def test_diamond_basis_extinction_and_ratios() -> None:
    st = Structure(
        Lattice.cubic(3.567), SpaceGroup.from_symbol("Fd-3m"), [AtomSite("C", "C", [0, 0, 0])]
    )
    f2 = (
        np.abs(
            structure_factors(st, [[1, 1, 1], [2, 2, 0], [2, 2, 2], [4, 0, 0], [3, 1, 1]], NEUTRON)
        )
        ** 2
    )
    b = NeutronScatteringLengths().scattering_length("C").real
    assert f2[0] == pytest.approx(32 * b**2)
    assert f2[1] == pytest.approx(64 * b**2)
    assert f2[2] == pytest.approx(0.0, abs=1e-9)  # 222 は基底で消える
    assert f2[3] == pytest.approx(64 * b**2)
    assert f2[4] == pytest.approx(32 * b**2)


def test_isotropic_debye_waller() -> None:
    u = 0.02
    st = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("Pm-3m"),
        [AtomSite("Fe", "Fe", [0, 0, 0], u_iso=u)],
    )
    hkl = np.array([[1, 0, 0], [2, 0, 0], [2, 2, 2]])
    f = structure_factors(st, hkl, NEUTRON).real
    b = NeutronScatteringLengths().scattering_length("Fe").real
    d = st.lattice.d_spacing(hkl)
    assert np.allclose(f, b * np.exp(-2 * np.pi**2 * u / d**2))


def test_anisotropic_equals_isotropic_when_equivalent() -> None:
    u = 0.015
    # 立方晶: U11=U22=U33=U, 非対角 0
    iso = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("P1"),
        [AtomSite("Fe", "Fe", [0.1, 0.2, 0.3], u_iso=u)],
    )
    ani = Structure(
        Lattice.cubic(4.0),
        SpaceGroup.from_symbol("P1"),
        [AtomSite("Fe", "Fe", [0.1, 0.2, 0.3], u_aniso=[u, u, u, 0, 0, 0])],
    )
    hkl = generate_reflections(iso.lattice, iso.spacegroup, 1.0).hkl
    assert np.allclose(structure_factors(iso, hkl, NEUTRON), structure_factors(ani, hkl, NEUTRON))
    # 六方晶: 等方性に相当する U^{ij} は U12 = U/2
    lat = Lattice.hexagonal(3.0, 5.0)
    iso_h = Structure(
        lat, SpaceGroup.from_symbol("P1"), [AtomSite("Fe", "Fe", [0.1, 0.2, 0.3], u_iso=u)]
    )
    ani_h = Structure(
        lat,
        SpaceGroup.from_symbol("P1"),
        [AtomSite("Fe", "Fe", [0.1, 0.2, 0.3], u_aniso=[u, u, u, u / 2, 0, 0])],
    )
    hkl = generate_reflections(lat, iso_h.spacegroup, 1.0).hkl
    assert np.allclose(
        structure_factors(iso_h, hkl, NEUTRON), structure_factors(ani_h, hkl, NEUTRON)
    )


def test_laue_equivalents_have_equal_magnitude(pbso4_cif) -> None:
    st = Structure.from_cif(str(pbso4_cif))
    xray = Radiation.characteristic("Cu", kalpha2=False)
    for hkl in ([1, 2, 1], [2, 1, 1], [3, 1, 2]):
        eq = st.spacegroup.laue_equivalents(hkl)
        f = structure_factors(st, eq, xray)
        assert np.allclose(np.abs(f), np.abs(f[0]))
    # 占有率は線形に効く
    st2 = st.copy()
    st2.sites[0].occupancy = 0.5
    f_full = structure_factors(st, [[2, 0, 0]], NEUTRON)
    f_half = structure_factors(st2, [[2, 0, 0]], NEUTRON)
    st_no_pb = st.copy()
    st_no_pb.sites[0].occupancy = 0.0
    f_none = structure_factors(st_no_pb, [[2, 0, 0]], NEUTRON)
    assert f_half == pytest.approx(0.5 * (f_full + f_none))


def test_xray_uses_form_factor_and_anomalous() -> None:
    st = Structure(
        Lattice.cubic(4.0), SpaceGroup.from_symbol("Pm-3m"), [AtomSite("Fe", "Fe", [0, 0, 0])]
    )
    xray = Radiation.characteristic("Cu", kalpha2=False)
    hkl = np.array([[1, 0, 0], [2, 2, 0]])
    f = structure_factors(st, hkl, xray)
    xr = XrayScatteringFactors()
    s = 1.0 / (2.0 * st.lattice.d_spacing(hkl))
    expected = xr.f0("Fe", s) + xr.anomalous("Fe", xray.wavelength)
    assert np.allclose(f, expected)
    # 異常分散の上書き
    f2 = structure_factors(st, hkl, xray, anomalous={"Fe": 0j})
    assert np.allclose(f2, xr.f0("Fe", s))
    # f0 の上書き
    f3 = structure_factors(st, hkl, xray, anomalous={"Fe": 0j}, f0_override={"Fe": [10.0, 5.0]})
    assert np.allclose(f3, [10.0, 5.0])
