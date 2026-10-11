"""Stephens の異方的微小歪み広がり。

反射 hkl の M = 1/d² の分散を 4 次形式で表す:

    σ²(M_hkl) = Σ_{H+K+L=4} S_HKL h^H k^K l^L          (15 項)

ピーク幅への寄与(2θ、deg): Δ(2θ) = (180/π) tanθ × sqrt(σ²(M)) / M × 10⁻⁶
(等方的な歪み ε(×10⁻⁶)では σ²(M) = ε² M² となり、既存の strain_e6 と同じ Γ になる)。

ラウエ群で許される項は、4 次形式をラウエ群の回転で平均した射影の像として求める
(`constraints.constraint_from_projector` と同じ仕組み)。独立な係数だけが精密化パラメータになり、
従属係数は線形拘束で結ばれる。例: 立方晶では S_400 (= S_040 = S_004) と S_220 (= S_202 = S_022)。

出典: P. W. Stephens, J. Appl. Cryst. 32, 281-289 (1999)。
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.refine.constraints import SymmetryConstraint, constraint_from_projector

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: 4 次単項式 (H, K, L) の順序(H + K + L = 4)
MONOMIALS: tuple[tuple[int, int, int], ...] = tuple(
    (h, k, 4 - h - k) for h in range(4, -1, -1) for k in range(4 - h, -1, -1)
)


def monomial_name(hkl: tuple[int, int, int]) -> str:
    return f"S_{hkl[0]}{hkl[1]}{hkl[2]}"


def monomial_values(hkl: npt.ArrayLike) -> FloatArray:
    """各反射の単項式の値 (N, 15)。"""
    h = np.atleast_2d(np.asarray(hkl, dtype=np.float64))
    cols = [h[:, 0] ** a * h[:, 1] ** b * h[:, 2] ** c for a, b, c in MONOMIALS]
    return np.stack(cols, axis=1)


def _generic_points() -> FloatArray:
    pts = [np.array(p, dtype=np.float64) for p in product((1, 2, 3, -1, -2), repeat=3)]
    return np.array(pts[:40])


def symmetrization_projector(spacegroup: SpaceGroup) -> FloatArray:
    """係数ベクトル S を、ラウエ群で不変な 4 次形式の係数に写す射影 (15×15)。"""
    pts = _generic_points()
    basis_at_pts = monomial_values(pts)  # (P, 15)
    pinv = np.linalg.pinv(basis_at_pts)  # 係数 → 値 の逆
    n = len(MONOMIALS)
    proj = np.zeros((n, n))
    rots = spacegroup.laue_rotations
    for rot in rots:
        # f_R(h) = f(h R) の単項式展開: 列 j は m_j(h R) の係数
        vals = monomial_values(pts @ rot)  # (P, 15): m_j(hR) の値
        proj += pinv @ vals
    proj /= len(rots)
    return proj


def stephens_constraint(
    spacegroup: SpaceGroup, values: npt.ArrayLike | None = None
) -> SymmetryConstraint:
    """ラウエ群による S_HKL の拘束(自由成分、従属成分の線形関係、固定成分)。"""
    vals = np.zeros(len(MONOMIALS)) if values is None else np.asarray(values, dtype=np.float64)
    return constraint_from_projector(symmetrization_projector(spacegroup), vals, 1e-8)


def stephens_fwhm(
    hkl: npt.ArrayLike,
    inv_d2: npt.ArrayLike,
    two_theta: npt.ArrayLike,
    coefficients: npt.ArrayLike,
) -> FloatArray:
    """反射ごとの歪み広がり Δ(2θ) [deg]。coefficients は S_HKL(15 個、×10⁻¹² Å⁻⁴ 単位)。"""
    c = np.asarray(coefficients, dtype=np.float64).reshape(len(MONOMIALS))
    var = monomial_values(hkl) @ c
    m = np.asarray(inv_d2, dtype=np.float64)
    th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(m > 0, np.sqrt(np.maximum(var, 0.0)) / m, 0.0)
    return np.asarray((180.0 / math.pi) * 1e-6 * rel * np.tan(th))


def isotropic_coefficients(strain_e6: float, lattice_metric_inv: FloatArray) -> FloatArray:
    """等方的な歪み ε(×10⁻⁶)に相当する S_HKL(σ²(M) = ε² M² を単項式に展開)。

    M = h^T G* h(G*: 逆格子の計量テンソル)。M² の展開係数に ε² を掛ける。
    """
    g = np.asarray(lattice_metric_inv, dtype=np.float64)
    pts = _generic_points()
    m = np.einsum("pi,ij,pj->p", pts, g, pts)
    coef, *_ = np.linalg.lstsq(monomial_values(pts), m * m, rcond=None)
    return np.asarray(strain_e6 * strain_e6 * coef)


__all__ = [
    "MONOMIALS",
    "isotropic_coefficients",
    "monomial_name",
    "monomial_values",
    "stephens_constraint",
    "stephens_fwhm",
    "symmetrization_projector",
]
