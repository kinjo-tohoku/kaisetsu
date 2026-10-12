"""対称性検出(spglib)と opXRD リーダー。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kaisetsu.crystal.atoms import AtomSite
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.crystal.structure import Structure
from kaisetsu.crystal.symmetry import find_symmetry
from kaisetsu.io import read_powder_data
from kaisetsu.io.readers.opxrd import read_opxrd


def _nacl_p1() -> Structure:
    """NaCl(Fm-3m、a = 5.64)を P1 の全原子 8 個で書く。"""
    na = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    cl = [(0.5, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0)]
    sites = [AtomSite(f"Na{i + 1}", "Na1+", np.array(p), u_iso=0.02) for i, p in enumerate(na)]
    sites += [AtomSite(f"Cl{i + 1}", "Cl1-", np.array(p), u_iso=0.015) for i, p in enumerate(cl)]
    return Structure(Lattice(5.64, 5.64, 5.64), SpaceGroup.from_number(1), sites, name="NaCl")


def test_find_symmetry_recovers_fm3m_from_p1() -> None:
    res = find_symmetry(_nacl_p1())
    st = res.structure
    assert res.number == 225 and st.spacegroup.number == 225
    assert res.n_atoms_in_cell == 8 and res.n_sites == 2
    assert st.lattice.a == pytest.approx(5.64) and st.lattice.alpha == pytest.approx(90.0)
    species = sorted(s.species for s in st.sites)
    assert species == ["Cl1-", "Na1+"]
    # 占有率・U_iso は引き継ぐ。展開すると元の 8 原子に戻る
    assert {s.u_iso for s in st.sites} == {0.02, 0.015}
    assert len(st.expand()) == 8
    assert "225" in res.summary()


def test_find_symmetry_monoclinic_standard_setting() -> None:
    """P2_1/c の構造(全原子 8 個)→ 標準設定(b 軸ユニーク)の非対称単位 2 個。"""
    sg = SpaceGroup.from_symbol("P 1 21/c 1")
    base = Structure(
        Lattice(5.0, 7.0, 9.0, 90.0, 105.0, 90.0),
        sg,
        [
            AtomSite("C1", "C", np.array([0.12, 0.31, 0.27])),
            AtomSite("O1", "O", np.array([0.42, 0.11, 0.65])),
        ],
    )
    exp = base.expand()
    p1 = Structure(
        base.lattice,
        SpaceGroup.from_number(1),
        [
            AtomSite(f"X{k}", base.sites[int(exp.site_index[k])].species, exp.positions[k])
            for k in range(len(exp))
        ],
    )
    assert len(exp) == 8
    res = find_symmetry(p1)
    assert res.number == 14 and res.n_sites == 2
    assert res.structure.lattice.beta == pytest.approx(
        105.0, abs=1e-3
    ) or res.structure.lattice.beta == pytest.approx(75.0, abs=1e-3)
    assert len(res.structure.expand()) == 8


def _write_opxrd(path: Path) -> Path:
    sites = [
        json.dumps(
            {"x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1, "symbol": "Na0+", "wyckoff_letter": None}
        ),
        json.dumps(
            {"x": 0.5, "y": 0.5, "z": 0.5, "occupancy": 1, "symbol": "Cl0+", "wyckoff_letter": None}
        ),
    ]
    phase = json.dumps(
        {
            "lattice": "(5.64, 5.64, 5.64, 90.0, 90.0, 90.0)",
            "basis": json.dumps(sites),
            "spacegroup": None,
            "chemical_composition": None,
            "wyckoff_symbols": None,
            "phase_fraction": None,
        }
    )
    label = json.dumps(
        {
            "phases": [phase],
            "xray_info": json.dumps({"primary_wavelength": "1.5406", "secondary_wavelength": None}),
            "is_simulated": False,
            "crystallite_size_nm": None,
            "temp_K": None,
        }
    )
    meta = json.dumps(
        {
            "filename": None,
            "institution": "Test Lab",
            "contributor_name": "X",
            "original_file_format": "pdCIF (.cif)",
        }
    )
    x = np.round(np.arange(10.0, 80.0, 0.02), 3)
    y = 100.0 + 50.0 * np.exp(-0.5 * ((x - 31.7) / 0.1) ** 2)
    path.write_text(
        json.dumps(
            {
                "two_theta_values": x.tolist(),
                "intensities": y.tolist(),
                "label": label,
                "metadata": meta,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_read_opxrd(tmp_path: Path) -> None:
    p = _write_opxrd(tmp_path / "pattern_0.json")
    pat = read_opxrd(p)
    assert pat.wavelength == pytest.approx(1.5406) and pat.wavelength2 is None
    assert pat.institution == "Test Lab" and not pat.is_simulated
    assert len(pat.data.x) == 3500 and pat.data.metadata["wavelength"] == pytest.approx(1.5406)
    assert len(pat.structures) == 1
    st = pat.structures[0]
    assert st.spacegroup.number == 1 and st.n_sites == 2
    assert [s.species for s in st.sites] == ["Na", "Cl"]  # 'Na0+' は電荷なし
    # 共通の読み込み口でもデータだけ読める
    d = read_powder_data(p)
    assert d.metadata.get("format") == "opXRD" and len(d.x) == 3500
