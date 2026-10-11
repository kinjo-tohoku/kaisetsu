"""PbSO4 の実験室 X 線データで Rietveld → MEM 電子密度 → MPF を行う例。

    uv run python examples/mem_pbso4.py            # Rietveld + MEM
    uv run python examples/mem_pbso4.py --mpf 3    # さらに MPF を 3 サイクル

出力(examples/output/):
    pbso4_mem.ccp4 / .xplor / .grd  VESTA で開ける電子密度(e/Å³)
    pbso4_mem_z0.csv                 z = 1/4 の断面(Pb と S が乗る鏡面)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    mem_from_project,
    read_powder_data,
    run_mpf,
    standard_recipe,
)
from kaisetsu.crystal.cif import read_structure
from kaisetsu.mem import density_slice, write_ccp4, write_grd, write_xplor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "pbso4"
OUT = ROOT / "examples" / "output"


def main(argv: list[str]) -> None:
    n_mpf = int(argv[argv.index("--mpf") + 1]) if "--mpf" in argv else 0
    OUT.mkdir(exist_ok=True)
    proj = Project("PbSO4 MEM")
    proj.add_phase(read_structure(DATA / "PbSO4-Wyckoff.cif"), name="PbSO4")
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), u=0.002, v=-0.002, w=0.004, x=0.02, y=0.02,
        s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip
    proj.add_histogram(read_powder_data(DATA / "PBSO4.XRA"), inst, SampleGeometry(), n_background=8)
    t0 = time.perf_counter()
    standard_recipe("PbSO4", "h0", sample_displacement=True).run(proj)
    rf = proj.rfactors()["total"]
    elapsed = time.perf_counter() - t0
    print(f"Rietveld: Rwp = {100 * rf.rwp:.2f}%  χ² = {rf.chi2:.2f}  ({elapsed:.0f} s)")

    t0 = time.perf_counter()
    grid = (64, 40, 54)  # a = 8.48, b = 5.40, c = 6.96 Å → 約 0.13 Å 刻み
    mem = mem_from_project(
        proj, 0, "PbSO4", grid,
        progress=lambda i, lam, c: print(f"  MEM 外側反復 {i}: λ = {lam:.3g}  C = {c:.3f}"),
    )  # fmt: skip
    print(
        f"MEM: {mem.message}  R_F = {100 * mem.r_factor():.2f}%  ({time.perf_counter() - t0:.0f} s)"
    )
    st = proj.phases["PbSO4"].structure
    for label in ("Pb", "S", "O1", "O2", "O3"):
        try:
            site = st.site(label)
        except KeyError:
            continue
        rho = mem.density[tuple(np.round(site.position * np.array(grid)).astype(int) % grid)]
        n_e = mem.integrate_sphere(site.position, 0.8)
        print(f"  {label:3s} 位置の密度 {rho:8.2f} e/Å³   球(0.8 Å)内 {n_e:6.1f} e")
    print("  密度の極大(上位 5):")
    for x, v in mem.maxima(5, 0.8):
        print(f"    ({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})  {v:.1f} e/Å³")
    write_ccp4(mem.density, st.lattice, st.spacegroup, OUT / "pbso4_mem.ccp4")
    write_xplor(mem.density, st.lattice, OUT / "pbso4_mem.xplor")
    write_grd(mem.density, st.lattice, OUT / "pbso4_mem.grd")
    sl, _ = density_slice(mem.density, 1, 0.25)
    np.savetxt(OUT / "pbso4_mem_y025.csv", sl, delimiter=",")
    print(f"  書き出し: {OUT / 'pbso4_mem.ccp4'} ほか")

    if n_mpf > 0:
        t0 = time.perf_counter()
        res = run_mpf(proj, 0, "PbSO4", grid=grid, n_cycles=n_mpf, progress=print)
        history = [f"{100 * r:.2f}%" for r in res.rwp_history]
        print(f"MPF: Rwp {history}  ({time.perf_counter() - t0:.0f} s)")
        write_ccp4(res.final.density, st.lattice, st.spacegroup, OUT / "pbso4_mpf.ccp4")


if __name__ == "__main__":
    main(sys.argv[1:])
