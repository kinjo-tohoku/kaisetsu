"""構造因子 F(hkl) の計算。

    F(h) = Σ_j occ_j f_j(s, λ) T_j(h) exp(2πi h·x_j)

和は単位胞内の全原子(対称展開済み)にわたる。
    - f_j: X 線では f0(s) + f' + i f''、中性子では散乱長 b [fm]
    - T_j: 温度因子。等方性 exp(-8π^2 U s^2) = exp(-2π^2 U / d^2)、
      異方性 exp(-h^T β h), β_ij = 2π^2 a*_i a*_j U^{ij}
    - s = sinθ/λ = 1/(2d)

出典: Giacovazzo Ch. 3.2; Pecharsky & Zavalij 式 2.90-2.100。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.structure import ExpandedAtoms, Structure
from kaisetsu.scattering.neutron import NeutronScatteringLengths
from kaisetsu.scattering.radiation import Radiation
from kaisetsu.scattering.xray import TableName, XrayScatteringFactors

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

TWO_PI2 = 2.0 * math.pi**2


def species_scattering_factors(
    species: tuple[str, ...],
    s: FloatArray,
    radiation: Radiation,
    *,
    xray_table: TableName = "it1992",
    anomalous: Mapping[str, complex] | None = None,
    f0_override: Mapping[str, npt.ArrayLike] | None = None,
    override: Mapping[str, npt.ArrayLike] | None = None,
) -> ComplexArray:
    """散乱種ごとの散乱因子 (N_refl, N_species)。

    Args:
        species: 散乱種のラベル
        s: sinθ/λ [Å^-1] (N_refl,)
        radiation: 放射線
        xray_table: f0 のテーブル
        anomalous: 散乱種 → f' + i f'' の上書き(検証用)。None なら Cromer-Liberman で計算。
        f0_override: 散乱種 → f0 配列 (N_refl,) の上書き(検証用)。
        override: 散乱種 → 散乱因子そのもの(スカラーまたは (N_refl,) 配列、複素数可)の上書き。
            X 線・中性子どちらでも使える(他の上書きより優先)。
    """
    n = len(s)
    out = np.zeros((n, len(species)), dtype=np.complex128)
    if radiation.is_neutron:
        table = NeutronScatteringLengths()
        for j, sp in enumerate(species):
            if override is not None and sp in override:
                out[:, j] = np.asarray(override[sp], dtype=np.complex128)
            else:
                out[:, j] = table.scattering_length(sp)
        return out
    xr = XrayScatteringFactors(xray_table)
    for j, sp in enumerate(species):
        if override is not None and sp in override:
            out[:, j] = np.asarray(override[sp], dtype=np.complex128)
            continue
        if f0_override is not None and sp in f0_override:
            f0 = np.asarray(f0_override[sp], dtype=np.float64)
        else:
            f0 = xr.f0(sp, s)
        if anomalous is not None and sp in anomalous:
            fa = complex(anomalous[sp])
        else:
            fa = xr.anomalous(sp, radiation.wavelength)
        out[:, j] = f0 + fa
    return out


def structure_factor_parts(
    structure: Structure,
    hkl: npt.ArrayLike,
    radiation: Radiation,
    *,
    expanded: ExpandedAtoms | None = None,
    xray_table: TableName = "it1992",
    anomalous: Mapping[str, complex] | None = None,
    f0_override: Mapping[str, npt.ArrayLike] | None = None,
    override: Mapping[str, npt.ArrayLike] | None = None,
    chunk: int = 2048,
) -> tuple[ComplexArray, ComplexArray]:
    """構造因子を散乱因子の実部と虚部の寄与 (A, B) に分けて返す。F = A + iB。

        A(h) = Σ_j occ_j Re(f_j) T_j exp(2πi h·x_j)
        B(h) = Σ_j occ_j Im(f_j) T_j exp(2πi h·x_j)

    X 線では Re f = f0 + f'、Im f = f''(中性子では散乱長の実部・虚部)。occ、T は実数なので
    F(-h) = conj(A) + i conj(B) となり、Friedel 対の |F|² の平均は |A|² + |B|² になる
    (`powder_structure_factor_squared`)。
    """
    h = np.atleast_2d(np.asarray(hkl, dtype=np.int64))
    ex = expanded if expanded is not None else structure.expand()
    inv_d2 = structure.lattice.inverse_d_squared(h)
    s = 0.5 * np.sqrt(inv_d2)
    f_species = species_scattering_factors(
        ex.species,
        s,
        radiation,
        xray_table=xray_table,
        anomalous=anomalous,
        f0_override=f0_override,
        override=override,
    )

    n = len(h)
    a = np.empty(n, dtype=np.complex128)
    b = np.empty(n, dtype=np.complex128)
    aniso = ex.is_anisotropic
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        hh = h[sl].astype(np.float64)
        f_atom = f_species[sl][:, ex.species_index]  # (n, Na)
        t_iso = np.exp(-TWO_PI2 * np.outer(inv_d2[sl], ex.u_iso))  # (n, Na)
        if aniso.any():
            t_aniso = np.exp(-np.einsum("ni,aij,nj->na", hh, ex.beta, hh))
            t = np.where(aniso[None, :], t_aniso, t_iso)
        else:
            t = t_iso
        phase = np.exp(2j * math.pi * (hh @ ex.positions.T))  # (n, Na)
        w = ex.occupancy[None, :] * t * phase
        a[sl] = np.sum(f_atom.real * w, axis=1)
        b[sl] = np.sum(f_atom.imag * w, axis=1)
    return a, b


def structure_factors(
    structure: Structure,
    hkl: npt.ArrayLike,
    radiation: Radiation,
    *,
    expanded: ExpandedAtoms | None = None,
    xray_table: TableName = "it1992",
    anomalous: Mapping[str, complex] | None = None,
    f0_override: Mapping[str, npt.ArrayLike] | None = None,
    override: Mapping[str, npt.ArrayLike] | None = None,
    chunk: int = 2048,
) -> ComplexArray:
    """反射 hkl (N, 3) の構造因子 F (N,) を返す。

    X 線では電子単位、中性子では fm 単位。
    """
    a, b = structure_factor_parts(
        structure,
        hkl,
        radiation,
        expanded=expanded,
        xray_table=xray_table,
        anomalous=anomalous,
        f0_override=f0_override,
        override=override,
        chunk=chunk,
    )
    return np.asarray(a + 1j * b, dtype=np.complex128)


def powder_structure_factor_squared(
    structure: Structure,
    hkl: npt.ArrayLike,
    radiation: Radiation,
    *,
    expanded: ExpandedAtoms | None = None,
    xray_table: TableName = "it1992",
    anomalous: Mapping[str, complex] | None = None,
    f0_override: Mapping[str, npt.ArrayLike] | None = None,
    override: Mapping[str, npt.ArrayLike] | None = None,
    chunk: int = 2048,
) -> FloatArray:
    """粉末の反射強度に使う |F|²: Friedel 対 (hkl) と (-h-k-l) の |F|² の平均。

    粉末では hkl と -h-k-l が同じ 2θ に重なり、ラウエ群の多重度は両方を数える。
    非中心対称構造では異常分散(f'')のため |F(h)|² ≠ |F(-h)|²(Friedel 則の破れ)なので、
    代表 1 本の |F|² に多重度を掛けると代表の選び方で強度が変わる。平均は

        (|F(h)|² + |F(-h)|²) / 2 = |A|² + |B|²

    (A、B は `structure_factor_parts`)。中心対称構造では A、B の位相が揃うので |F|² と同じ。
    出典: Giacovazzo『Fundamentals of Crystallography』3.5(異常分散と Friedel 則)、
    IUCr Online Dictionary "Friedel's law"。GSAS-II 5.7.4 の粉末の分岐は代表 1 本の |F|² を
    使う(GSASIIstrMath.py の StructureFactor2。平均形はコメントアウトされている)ので、
    非中心対称の X 線では GSAS-II と 1% 程度ずれることがある(レビュー 2026-10-17 #2)。
    """
    a, b = structure_factor_parts(
        structure,
        hkl,
        radiation,
        expanded=expanded,
        xray_table=xray_table,
        anomalous=anomalous,
        f0_override=f0_override,
        override=override,
        chunk=chunk,
    )
    return np.asarray(np.abs(a) ** 2 + np.abs(b) ** 2, dtype=np.float64)
