"""密度格子の書き出し(VESTA で読める形式)と断面の取り出し。

- CCP4/MRC (.ccp4): gemmi で書く。VESTA、Coot、ChimeraX が読む
- XPLOR/CNS (.xplor): テキスト。VESTA が読む
- VESTA .grd: テキスト(タイトル、格子定数、格子点数、値)。最も単純
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup

FloatArray = npt.NDArray[np.float64]


def write_ccp4(
    density: FloatArray, lattice: Lattice, spacegroup: SpaceGroup | None, path: str | Path
) -> Path:
    """CCP4 形式で書く(gemmi)。密度は (n1, n2, n3)、添字は a, b, c 軸に沿う分率座標。"""
    n1, n2, n3 = density.shape
    grid = gemmi.FloatGrid(n1, n2, n3)
    grid.set_unit_cell(
        gemmi.UnitCell(lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)
    )
    if spacegroup is not None and spacegroup.gemmi_spacegroup is not None:
        grid.spacegroup = spacegroup.gemmi_spacegroup
    arr = np.array(grid, copy=False)
    arr[...] = np.asarray(density, dtype=np.float32)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    p = Path(path)
    ccp4.write_ccp4_map(str(p))
    return p


def write_xplor(
    density: FloatArray, lattice: Lattice, path: str | Path, title: str = "kaisetsu MEM"
) -> Path:
    """XPLOR/CNS 形式(ASCII)で書く。z 断面ごとに x が最も速く変わる順。"""
    n1, n2, n3 = density.shape
    p = Path(path)
    lines = ["", "       1 !NTITLE", f" REMARKS {title[:70]}"]
    lines.append(f"{n1:8d}{0:8d}{n1 - 1:8d}{n2:8d}{0:8d}{n2 - 1:8d}{n3:8d}{0:8d}{n3 - 1:8d}")
    lines.append(
        "".join(
            f"{v:12.5E}"
            for v in (lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)
        )
    )
    lines.append("ZYX")
    d = np.asarray(density, dtype=np.float64)
    for k in range(n3):
        lines.append(f"{k:8d}")
        section = d[:, :, k].T.ravel()  # y ごとに x が速く変わる
        for i in range(0, len(section), 6):
            lines.append("".join(f"{v:12.5E}" for v in section[i : i + 6]))
    lines.append(f"{-9999:8d}")
    lines.append(f"{float(d.mean()):12.4E}{float(d.std()):12.4E}")
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    return p


def write_grd(
    density: FloatArray, lattice: Lattice, path: str | Path, title: str = "kaisetsu MEM"
) -> Path:
    """VESTA の .grd 形式。1 行目タイトル、2 行目格子定数、3 行目格子点数、以降は値。

    値は x が最も速く変わる順。
    """
    n1, n2, n3 = density.shape
    p = Path(path)
    d = np.asarray(density, dtype=np.float64)
    vals = d.transpose(2, 1, 0).ravel()  # z, y, x の順(x が最も速い)
    with p.open("w", encoding="ascii") as f:
        f.write(title[:70] + "\n")
        f.write(
            f"{lattice.a:.6f} {lattice.b:.6f} {lattice.c:.6f} "
            f"{lattice.alpha:.4f} {lattice.beta:.4f} {lattice.gamma:.4f}\n"
        )
        f.write(f"{n1} {n2} {n3}\n")
        for i in range(0, len(vals), 6):
            f.write(" ".join(f"{v:.6E}" for v in vals[i : i + 6]) + "\n")
    return p


def density_slice(
    density: FloatArray, axis: int, fraction: float
) -> tuple[FloatArray, tuple[int, int]]:
    """軸 axis(0: a, 1: b, 2: c)に垂直な、分率 fraction の位置の 2D 断面(最近接の格子面)。

    Returns:
        (断面 2D 配列, 残り 2 軸の添字)
    """
    n = density.shape[axis]
    k = round((fraction % 1.0) * n) % n
    sl = np.take(density, k, axis=axis)
    others = tuple(i for i in range(3) if i != axis)
    return np.asarray(sl), (others[0], others[1])


def line_profile(
    density: FloatArray,
    lattice: Lattice,
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    n_points: int = 200,
) -> tuple[FloatArray, FloatArray]:
    """2 点(分率座標)を結ぶ線分上の密度(三重線形補間)。距離 [Å] と値を返す。"""
    s = np.asarray(start, dtype=np.float64).reshape(3)
    e = np.asarray(end, dtype=np.float64).reshape(3)
    t = np.linspace(0.0, 1.0, n_points)
    pts = s[None, :] + t[:, None] * (e - s)[None, :]
    vals = trilinear(density, pts)
    dist = t * float(lattice.distance(s, e))
    return dist, vals


def trilinear(density: FloatArray, frac: npt.ArrayLike) -> FloatArray:
    """周期境界つき三重線形補間。frac は (..., 3) の分率座標。"""
    f = np.asarray(frac, dtype=np.float64)
    shape = f.shape[:-1]
    f = f.reshape(-1, 3)
    n = np.array(density.shape)
    g = (f % 1.0) * n
    i0 = np.floor(g).astype(int)
    w = g - i0
    out = np.zeros(len(f))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ii = (i0[:, 0] + dx) % n[0]
                jj = (i0[:, 1] + dy) % n[1]
                kk = (i0[:, 2] + dz) % n[2]
                wt = (
                    (w[:, 0] if dx else 1 - w[:, 0])
                    * (w[:, 1] if dy else 1 - w[:, 1])
                    * (w[:, 2] if dz else 1 - w[:, 2])
                )
                out += wt * density[ii, jj, kk]
    return out.reshape(shape)
