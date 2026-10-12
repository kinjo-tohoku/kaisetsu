"""Phase 1 の GSAS-II 突き合わせ: 反射リストと |F_calc|^2。

参照値は validation/gsas2/run_gsas2.py で生成した JSON。無ければスキップする。
"""

from __future__ import annotations

import math
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

from kaisetsu.crystal.atoms import normalize_species
from kaisetsu.crystal.cif import read_structure
from kaisetsu.reflections import canonical_hkl, generate_reflections, structure_factors
from kaisetsu.scattering import Radiation
from tests.conftest import DATA_DIR, load_reference

CASES = ["pbso4_xray", "pbso4_neutron", "diamond_xray_sim"]

pytestmark = pytest.mark.validation


def _cif_for(ref: dict) -> Path:
    # 参照値 JSON には生成した Windows の絶対パスが入っているので、どの OS でもファイル名だけ取る
    name = PureWindowsPath(ref["case"]["cif"]).name
    hits = list(DATA_DIR.rglob(name))
    assert hits, f"{name} が data/public にありません"
    return hits[0]


def _radiation(ref: dict) -> Radiation:
    inst = ref["instrument"]
    if inst["Type"].startswith("PX"):
        return Radiation.xray(
            float(inst["Lam1"]),
            float(inst.get("Lam2") or 0) or None,
            float(inst.get("I(L2)/I(L1)", 0.5)),
        )
    return Radiation.neutron(float(inst["Lam"]))


def _structure_with_gsas_adps(ref: dict):
    """CIF から構造を読み、ADP と占有率だけ GSAS-II が実際に使った値に揃える。

    (CIF に ADP が無いとき GSAS-II は Uiso = 0.01 を既定にする。座標は原点選択が
    変わることがあるので CIF のままにする。|F| は原点に依らない。)
    """
    st = read_structure(_cif_for(ref))
    by_label = {a["label"]: a for a in ref["phase"]["atoms"]}
    for site in st.sites:
        a = by_label.get(site.label)
        if a is None:
            continue
        site.occupancy = float(a["frac"])
        if a["iso_or_aniso"] == "I":
            site.u_iso = float(a["Uiso"])
            site.u_aniso = None
        else:
            site.u_aniso = np.asarray(a["Uij"], dtype=float)
    return st


def _anomalous_from_ref(ref: dict, el: str) -> complex:
    entries = ref["form_factor_coeff"].get(el)
    coeff = entries[0] if isinstance(entries, list) and entries else (entries or {})
    if isinstance(coeff, dict) and ":0:FP" in coeff:
        return complex(coeff[":0:FP"], coeff[":0:FPP"])
    an = ref["anomalous"][el]
    return complex(an["fp"], an["fpp"])


def _gsas_reflections(ref: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(ref["reflections"]["data"], dtype=float)
    cols = ref["reflections"]["columns"]
    hkl = data[:, :3].astype(np.int64)
    mult = data[:, cols.index("mult")].astype(np.int64)
    d = data[:, cols.index("d")]
    f2 = data[:, cols.index("Fcalc2")]
    return hkl, mult, d, f2


@pytest.mark.parametrize("case", CASES)
def test_reflection_list_matches_gsas2(case: str) -> None:
    ref = load_reference(case)
    st = read_structure(_cif_for(ref))
    g_hkl, g_mult, g_d, _ = _gsas_reflections(ref)
    rad = _radiation(ref)
    tt_min = float(ref["limits"][1][0])
    d_max = rad.wavelength / (2.0 * math.sin(math.radians(tt_min / 2.0)))
    ours = generate_reflections(
        st.lattice, st.spacegroup, float(g_d.min()) * (1 - 1e-9), d_max=d_max
    )

    canon = canonical_hkl(st.spacegroup, g_hkl)
    g_set = {tuple(h) for h in canon}
    o_set = {tuple(h) for h in ours.hkl}
    missing = g_set - o_set
    extra = o_set - g_set
    assert not missing, f"GSAS-II にあって kaisetsu に無い反射: {sorted(missing)[:10]}"
    assert not extra, f"kaisetsu にあって GSAS-II に無い反射: {sorted(extra)[:10]}"
    assert len(g_set) == len(g_hkl), "GSAS-II の反射リストに等価な重複がある"

    index = {tuple(h): i for i, h in enumerate(ours.hkl)}
    for h, m, d in zip(canon, g_mult, g_d, strict=True):
        i = index[tuple(h)]
        assert ours.multiplicity[i] == m, (
            f"多重度不一致 {h}: kaisetsu {ours.multiplicity[i]} vs GSAS-II {m}"
        )
        assert ours.d[i] == pytest.approx(d, rel=1e-6), f"d 不一致 {h}"


@pytest.mark.parametrize("case", CASES)
def test_structure_factors_match_gsas2_with_same_scattering_factors(case: str) -> None:
    """GSAS-II と同じ散乱因子(f0 の値、f'、f''、b)を注入し、構造因子の計算そのものを比べる。"""
    ref = load_reference(case)
    st = _structure_with_gsas_adps(ref)
    g_hkl, _, _, g_f2 = _gsas_reflections(ref)
    rad = _radiation(ref)

    override: dict[str, np.ndarray] = {}
    if rad.is_xray:
        ff = ref["reflections"]["FF"]
        ff_vals = np.asarray(ff["FF"], dtype=float)
        for j, el in enumerate(ff["El"]):
            override[normalize_species(el)] = ff_vals[:, j] + _anomalous_from_ref(ref, el)
        scale = 1.0
    else:
        for el, bl in ref["neutron_blens"].items():
            re_, im_ = bl["SL"]
            override[normalize_species(el)] = np.full(len(g_hkl), complex(re_ * 10.0, im_ * 10.0))
        scale = 1.0 / 100.0  # GSAS-II は 10^-12 cm 単位、kaisetsu は fm 単位

    f = structure_factors(st, g_hkl, rad, override=override)
    f2 = np.abs(f) ** 2 * scale
    strong = g_f2 > 1e-6 * g_f2.max()
    rel = np.abs(f2[strong] - g_f2[strong]) / g_f2[strong]
    worst = int(np.argmax(rel))
    assert rel.max() < 1e-4, (
        f"|F|^2 の最大相対差 {rel.max():.2e} at hkl={g_hkl[strong][worst]} "
        f"(kaisetsu {f2[strong][worst]:.6g}, GSAS-II {g_f2[strong][worst]:.6g})"
    )
    assert np.all(np.abs(f2[~strong] - g_f2[~strong]) < 1e-6 * g_f2.max())


@pytest.mark.parametrize("case", [c for c in CASES if "xray" in c])
def test_own_f0_table_matches_gsas2(case: str) -> None:
    """自前の IT1992 テーブルで f0 を計算し、f', f'' だけ GSAS-II の値を使う。

    テーブルが GSAS-II (atmdata.XrayFF) と同一であることの確認。
    """
    ref = load_reference(case)
    st = _structure_with_gsas_adps(ref)
    g_hkl, _, _, g_f2 = _gsas_reflections(ref)
    rad = _radiation(ref)
    anomalous = {
        normalize_species(el): _anomalous_from_ref(ref, el) for el in ref["reflections"]["FF"]["El"]
    }
    f2 = np.abs(structure_factors(st, g_hkl, rad, anomalous=anomalous)) ** 2
    strong = g_f2 > 1e-6 * g_f2.max()
    rel = np.abs(f2[strong] - g_f2[strong]) / g_f2[strong]
    assert rel.max() < 1e-4


@pytest.mark.parametrize("case", CASES)
def test_structure_factors_match_gsas2_with_own_tables(case: str) -> None:
    """kaisetsu 自身のテーブル(IT1992 + Cromer-Liberman(gemmi) / Sears 1992)で比べる。

    残る差は f' の実装差(GSAS-II と gemmi で Pb の f' が 0.13 e 違う)と
    散乱長表の版の差(Pb 9.4024 vs 9.405 fm 等)。許容差はそれを見込んで緩い。
    内訳は docs/validation/phase1_structure_factors.md に記録。
    """
    ref = load_reference(case)
    st = _structure_with_gsas_adps(ref)
    g_hkl, _, _, g_f2 = _gsas_reflections(ref)
    rad = _radiation(ref)
    f = structure_factors(st, g_hkl, rad)
    f2 = np.abs(f) ** 2 * (1.0 if rad.is_xray else 0.01)
    strong = g_f2 > 1e-2 * g_f2.max()
    rel = np.abs(f2[strong] - g_f2[strong]) / g_f2[strong]
    scaled = np.abs(f2 - g_f2) / g_f2.max()
    tol = 1.0e-2 if rad.is_xray else 2e-3
    print(
        f"[{case}] 自前テーブルとの差: 中央値 {np.median(rel):.2e}, "
        f"強い反射の最大 {rel.max():.2e}, 最大|Δ|/max {scaled.max():.2e}"
    )
    assert np.median(rel) < tol
    assert scaled.max() < tol, (
        f"|F|^2 の最大差/max {scaled.max():.2e} at hkl={g_hkl[int(np.argmax(scaled))]}"
    )
