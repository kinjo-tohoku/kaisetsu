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
    out = np.empty(n, dtype=np.complex128)
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
        out[sl] = np.sum(ex.occupancy[None, :] * f_atom * t * phase, axis=1)
    return out
