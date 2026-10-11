from pathlib import Path

import numpy as np
import pytest

from kaisetsu.crystal import AtomSite, Lattice, SpaceGroup, Structure
from kaisetsu.crystal.atoms import normalize_species, split_species
from kaisetsu.crystal.cif import read_structure, write_structure


def test_species_parsing() -> None:
    assert split_species("Pb2+") == ("Pb", 2)
    assert split_species("Pb+2") == ("Pb", 2)
    assert split_species("O2-") == ("O", -2)
    assert split_species("O-2") == ("O", -2)
    assert split_species("H1-") == ("H", -1)
    assert split_species("Fe") == ("Fe", 0)
    assert split_species("Cl-") == ("Cl", -1)
    assert normalize_species("Pb+2") == "Pb2+"
    assert normalize_species("Fe") == "Fe"
    with pytest.raises(ValueError):
        split_species("Xx12")


def test_atom_site_basics() -> None:
    a = AtomSite("Pb1", "Pb+2", [0.1, 0.2, 0.3], 0.5, 0.01)
    assert a.species == "Pb2+"
    assert a.element == "Pb"
    assert a.charge == 2
    assert not a.is_anisotropic
    assert a.b_iso == pytest.approx(8 * np.pi**2 * 0.01)
    b = AtomSite("O1", "O", [0, 0, 0], u_aniso=[0.01, 0.02, 0.03, 0.001, 0.002, 0.003])
    assert b.is_anisotropic
    assert np.allclose(
        b.u_matrix(), [[0.01, 0.001, 0.002], [0.001, 0.02, 0.003], [0.002, 0.003, 0.03]]
    )
    with pytest.raises(ValueError):
        AtomSite("X", "O", [0, 0, 0], occupancy=1.5)


def test_read_pbso4_cif(pbso4_cif: Path) -> None:
    st = read_structure(pbso4_cif)
    assert st.name == "PbSO4"
    assert st.spacegroup.number == 62
    assert st.lattice.a == pytest.approx(8.48)
    assert [s.label for s in st.sites] == ["Pb", "S", "O1", "O2", "O3"]
    assert list(st.multiplicities()) == [4, 4, 4, 4, 8]
    assert st.composition() == {"Pb": 4.0, "S": 4.0, "O": 16.0}
    assert st.formula_units() == pytest.approx(4.0)
    ex = st.expand()
    assert len(ex) == 24
    assert ex.species == ("Pb", "S", "O")
    assert not ex.is_anisotropic.any()
    assert np.allclose(ex.u_iso, 0.01)


def test_read_diamond_cif_uses_cif_symops(diamond_cif: Path) -> None:
    st = read_structure(diamond_cif)
    assert st.spacegroup.number == 227
    assert st.spacegroup.n_operations == 192
    assert len(st.expand()) == 8  # 8a: 原点選択 1


def test_cif_roundtrip(tmp_path: Path, pbso4_cif: Path) -> None:
    st = read_structure(pbso4_cif)
    st.sites[0].u_aniso = np.array([0.01, 0.02, 0.03, 0.0, 0.001, 0.0])
    out = tmp_path / "out.cif"
    write_structure(st, out)
    back = read_structure(out)
    assert back.spacegroup == st.spacegroup
    assert np.allclose(back.lattice.parameters, st.lattice.parameters)
    assert len(back.sites) == len(st.sites)
    assert back.sites[0].is_anisotropic
    assert np.allclose(back.sites[0].u_aniso, st.sites[0].u_aniso)
    for a, b in zip(back.sites, st.sites, strict=True):
        assert np.allclose(a.position, b.position)
        assert a.species == b.species


def test_expand_anisotropic_transforms_beta() -> None:
    # 正方晶で 4 回軸により U11 と U22 が入れ替わることを確認
    lat = Lattice.tetragonal(4.0, 6.0)
    sg = SpaceGroup.from_symbol("P4/mmm")
    site = AtomSite("A", "Fe", [0.3, 0.0, 0.0], u_aniso=[0.01, 0.02, 0.03, 0, 0, 0])
    st = Structure(lat, sg, [site])
    ex = st.expand()
    assert len(ex) == 4  # 4j? (x,0,0) は 4 重
    rec = lat.reciprocal
    scale = 2 * np.pi**2
    betas = {tuple(np.round(p, 6)): b for p, b in zip(ex.positions, ex.beta, strict=True)}
    b_x = betas[(0.3, 0.0, 0.0)]
    b_y = betas[(0.0, 0.3, 0.0)]
    assert b_x[0, 0] == pytest.approx(scale * rec.a**2 * 0.01)
    assert b_x[1, 1] == pytest.approx(scale * rec.b**2 * 0.02)
    assert b_y[0, 0] == pytest.approx(scale * rec.a**2 * 0.02)
    assert b_y[1, 1] == pytest.approx(scale * rec.b**2 * 0.01)
    assert b_y[2, 2] == pytest.approx(scale * rec.c**2 * 0.03)


def test_duplicate_labels_rejected() -> None:
    lat = Lattice.cubic(4.0)
    sg = SpaceGroup.from_symbol("Pm-3m")
    with pytest.raises(ValueError):
        Structure(lat, sg, [AtomSite("A", "Na", [0, 0, 0]), AtomSite("A", "Cl", [0.5, 0.5, 0.5])])
