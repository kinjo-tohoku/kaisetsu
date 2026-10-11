"""球面調和関数による集合組織(選択配向)の補正。

試料が回折ベクトルのまわりに円筒対称(Bragg-Brentano の平板、Debye-Scherrer のキャピラリー)なら、
反射 h の強度補正は結晶座標系の対称化球面調和関数だけで書ける:

    O(h) = 1 + Σ_{l=2,4,…}^{L} Σ_j C_{l,j} K_{l,j}(ĥ)

ĥ は逆格子ベクトルの単位ベクトル(直交座標)、K_{l,j} はラウエ群で不変な l 次の実球面調和関数
(球面上の二乗平均が 1 になるよう規格化)。試料方位の因子 P_l(cos ψ) は l ごとの定数なので
係数 C に吸収させる(平板と毛細管では C の符号・大きさの解釈が変わるだけ)。
集合組織指数 J = (1/4π)∫O² dΩ = 1 + Σ C² が 1 に近いほど無配向に近い。

ラウエ群で不変な調和関数の基底は、各次数の実球面調和関数をラウエ群の回転で平均した射影の像として
数値的に求める(`profile.stephens` と同じ仕組み)。次元はよく知られた値に一致する:
m-3m で l = 2, 4, 6 に 0, 1, 1 個、6/mmm で 1, 1, 2 個、4/mmm で 1, 2, 2 個、mmm で 2, 3, 4 個。

出典: R. B. Von Dreele, J. Appl. Cryst. 30, 517-525 (1997);
      M. Järvinen, J. Appl. Cryst. 26, 525-531 (1993);
      H.-J. Bunge, *Texture Analysis in Materials Science* (1982)。
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.special import lpmv

from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def real_spherical_harmonics(order: int, unit_vectors: npt.ArrayLike) -> FloatArray:
    """l = order の実球面調和関数 (N, 2l+1)。球面上の二乗平均が 1(√(4π) Y_lm)。

    列の順は m = 0, 1c, 1s, 2c, 2s, …(c: cos mφ、s: sin mφ)。
    """
    u = np.atleast_2d(np.asarray(unit_vectors, dtype=np.float64))
    z = np.clip(u[:, 2], -1.0, 1.0)
    phi = np.arctan2(u[:, 1], u[:, 0])
    cols = [math.sqrt(2 * order + 1) * lpmv(0, order, z)]
    for m in range(1, order + 1):
        norm = math.sqrt(
            2.0 * (2 * order + 1) * math.factorial(order - m) / math.factorial(order + m)
        )
        p = norm * lpmv(m, order, z)
        cols.append(p * np.cos(m * phi))
        cols.append(p * np.sin(m * phi))
    return np.stack(cols, axis=1)


def reciprocal_orthogonalization(lattice: Lattice) -> FloatArray:
    """逆格子の指数 (h, k, l) → 逆格子ベクトルの直交座標 [Å⁻¹] の行列 B(c = B h)。"""
    return np.asarray(np.linalg.inv(lattice.orthogonalization_matrix).T)


def unit_reciprocal_vectors(hkl: npt.ArrayLike, lattice: Lattice) -> FloatArray:
    """反射の逆格子ベクトルの単位ベクトル (N, 3)。"""
    h = np.atleast_2d(np.asarray(hkl, dtype=np.float64))
    c = h @ reciprocal_orthogonalization(lattice).T
    n = np.linalg.norm(c, axis=1)
    n = np.where(n > 0, n, 1.0)
    return np.asarray(c / n[:, None])


def cartesian_laue_rotations(spacegroup: SpaceGroup, lattice: Lattice) -> FloatArray:
    """ラウエ群の回転を逆格子の直交座標での直交行列 Q (M, 3, 3) にする(c' = Q c)。

    hkl への作用 h' = h R(行ベクトル)に対し Q = B Rᵀ B⁻¹。格子が対称性と整合していれば Q は
    直交行列になる。わずかな不整合は最も近い直交行列(極分解)に丸める。
    """
    b = reciprocal_orthogonalization(lattice)
    binv = np.linalg.inv(b)
    out = []
    for rot in spacegroup.laue_rotations:
        q = b @ rot.T.astype(np.float64) @ binv
        u, _, vt = np.linalg.svd(q)
        out.append(u @ vt)
    return np.asarray(np.stack(out))


def _fibonacci_sphere(n: int) -> FloatArray:
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def symmetrized_basis(order: int, rotations: FloatArray, tol: float = 1e-6) -> FloatArray:
    """l = order の実球面調和関数のうち、回転群 rotations で不変な部分空間の正規直交基底。

    Returns:
        U (2l+1, n_l): 不変な調和関数 K_j(ĥ) = Σ_i U_ij Y_i(ĥ) の係数。n_l = 0 のこともある。
    """
    pts = _fibonacci_sphere(max(8 * (2 * order + 1), 200))
    a = real_spherical_harmonics(order, pts)  # (P, 2l+1)
    pinv = np.linalg.pinv(a)
    n = 2 * order + 1
    proj = np.zeros((n, n))
    for q in rotations:
        # f(Q ĥ) の展開: 列 j は Y_j(Q ĥ) の係数ベクトル
        proj += pinv @ real_spherical_harmonics(order, pts @ q.T)
    proj /= len(rotations)
    # 直交表現の平均は直交射影 → 固有値 1 の固有ベクトルが不変基底
    proj = 0.5 * (proj + proj.T)
    w, v = np.linalg.eigh(proj)
    keep = w > 1.0 - 1e-3
    u = v[:, keep]
    # 符号を決めて再現性を保つ(最大成分を正に)
    for j in range(u.shape[1]):
        k = int(np.argmax(np.abs(u[:, j])))
        if u[k, j] < 0:
            u[:, j] *= -1.0
    u[np.abs(u) < tol] = 0.0
    return np.asarray(u)


@lru_cache(maxsize=64)
def _cached_basis(
    triplets: tuple[str, ...], order: int, lat_key: tuple[float, ...]
) -> tuple[tuple[int, FloatArray], ...]:
    lattice = Lattice.from_parameters(np.array(lat_key))
    sg = SpaceGroup.from_symops(triplets)
    rots = cartesian_laue_rotations(sg, lattice)
    out: list[tuple[int, FloatArray]] = []
    for ell in range(2, order + 1, 2):
        u = symmetrized_basis(ell, rots)
        if u.shape[1] > 0:
            out.append((ell, u))
    return tuple(out)


def symmetrized_harmonics(
    spacegroup: SpaceGroup, lattice: Lattice, order: int
) -> tuple[tuple[int, FloatArray], ...]:
    """l = 2, 4, …, order のラウエ群不変な調和関数の基底 [(l, U), …](空の次数は省く)。

    基底は格子定数の値によらない(対称性に整合した格子なら回転 Q が同じ)ので、
    空間群の操作と格子の形(軸比と角度を丸めたもの)でキャッシュする。
    """
    triplets = tuple(op.triplet for op in spacegroup.operations)
    p = lattice.parameters
    lat_key = (
        1.0,
        round(float(p[1] / p[0]), 3),
        round(float(p[2] / p[0]), 3),
        *(round(float(v), 1) for v in p[3:]),
    )
    return _cached_basis(triplets, order, lat_key)


def basis_size(basis: tuple[tuple[int, FloatArray], ...]) -> int:
    return int(sum(u.shape[1] for _, u in basis))


def coefficient_names(basis: tuple[tuple[int, FloatArray], ...]) -> list[str]:
    """係数の名前 sh{l}_{j}(j は 1 から)。"""
    return [f"sh{ell}_{j + 1}" for ell, u in basis for j in range(u.shape[1])]


def harmonics_matrix(
    hkl: npt.ArrayLike, lattice: Lattice, basis: tuple[tuple[int, FloatArray], ...]
) -> FloatArray:
    """各反射での対称化調和関数の値 (N, n)。列の順は coefficient_names と同じ。"""
    uv = unit_reciprocal_vectors(hkl, lattice)
    cols = [real_spherical_harmonics(ell, uv) @ u for ell, u in basis]
    if not cols:
        return np.zeros((len(uv), 0))
    return np.asarray(np.concatenate(cols, axis=1))


def texture_factor(
    hkl: npt.ArrayLike,
    lattice: Lattice,
    basis: tuple[tuple[int, FloatArray], ...],
    coefficients: npt.ArrayLike,
) -> FloatArray:
    """反射ごとの集合組織補正 O(h) = 1 + Σ C K(ĥ)(負にはしない)。"""
    k = harmonics_matrix(hkl, lattice, basis)
    c = np.asarray(coefficients, dtype=np.float64).reshape(k.shape[1])
    return np.asarray(np.maximum(1.0 + k @ c, 0.0))


def texture_index(coefficients: npt.ArrayLike) -> float:
    """集合組織指数 J = 1 + Σ C²(無配向で 1)。"""
    c = np.asarray(coefficients, dtype=np.float64)
    return float(1.0 + np.sum(c * c))


__all__ = [
    "basis_size",
    "cartesian_laue_rotations",
    "coefficient_names",
    "harmonics_matrix",
    "real_spherical_harmonics",
    "reciprocal_orthogonalization",
    "symmetrized_basis",
    "symmetrized_harmonics",
    "texture_factor",
    "texture_index",
    "unit_reciprocal_vectors",
]
