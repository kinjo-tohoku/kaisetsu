"""TOF 中性子(POWGEN、NAC)の GSAS-II 突き合わせ: 同じ手順で精密化した結果の比較。

GSAS-II 側は validation/gsas2/run_gsas2.py の nac_tof_fit ケース
(背景 3 項 + スケール → 格子 + ゼロ点 → sig-1 → beta-0 → 座標 → U_ij)。alpha は装置校正値のまま
(GSAS-II で動かすと 2e8 に発散した)。試料広がりはガウス(LGmix = 0)にして、両者で同じ関数形
(erfc による閉形式)で比べる。
kaisetsu 側も同じ順序で精密化し、Rwp・格子定数・ゼロ点・幅・スケール・背景・座標・U_ij を比べる。
GSAS-II の TOF プロファイル(pyepsvoigt)は複素指数積分を近似で計算し裾が最大 1 割違うので
(docs/validation/phase7_tof.md)、ローレンツ成分を無くして比べる。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kaisetsu.api import Project, SampleGeometry
from kaisetsu.io import read_powder_data
from kaisetsu.pattern.instrument import GeometryKind
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.profile import ChebyshevBackground
from tests.conftest import DATA_DIR, load_reference
from tests.validation.test_gsas2_phase1 import _structure_with_gsas_adps

pytestmark = [pytest.mark.validation, pytest.mark.slow]

STAGES = [
    ("h0:bkg:*", "h0:NAC:scale"),
    ("NAC:cell:*", "h0:inst:zero"),
    ("h0:inst:sig1",),
    ("h0:inst:beta0",),
    ("NAC:*:x", "NAC:*:y", "NAC:*:z"),
    ("NAC:*:u*",),  # NAC の CIF は異方性なので U11〜U23(GSAS-II も同じ)
]


def _build_project() -> tuple[Project, dict]:
    init = load_reference("nac_tof")
    fit = load_reference("nac_tof_fit")
    proj = Project("nac_tof_fit")
    proj.add_phase(_structure_with_gsas_adps(init), name="NAC")
    data = read_powder_data(DATA_DIR / "nac" / "PG3_22048.gsa")
    lo, hi = (float(v) for v in init["limits"][1])
    data = data.cropped(lo - 1e-6, hi + 1e-6)  # GSAS-II の Limits と同じ点集合
    inst = TOFInstrument.from_gsas2(init["instrument"])
    bkg = ChebyshevBackground(np.zeros(3), float(data.x[0]), float(data.x[-1]))
    bkg.coefficients[0] = 1.0  # GSAS-II の初期値 (1, 0, 0)
    h = proj.add_histogram(
        data, inst, SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER), bkg, name="h0"
    )
    pc = proj.phase_contribution(0, "NAC")
    pc.size_um, pc.strain_e6 = 1.0, 1000.0  # GSAS-II の既定
    # ローレンツ成分を無くし(LGmix = 0、X = Y = Z = 0)、両者で厳密に同じガウス ⊗ 指数関数で比べる
    # (GSAS-II の複素指数積分は近似なので、ローレンツ裾があると幅・β・ゼロ点が同じ極小にならず、
    #  GSAS-II は原子の段階で Levenberg-Marquardt が止まった)
    pc.size_lorentz_fraction = 0.0
    pc.strain_lorentz_fraction = 0.0
    pc.scale = 1.0 / 100.0  # GSAS-II の Scale 1.0 に相当(TOF では F² の単位差は無い)
    h.scattering_override = None  # Sears の表(GSAS-II と同じ出典)
    proj.build_parameters()
    return proj, fit


def _run_stages(proj: Project, stages: list[tuple[str, ...]], last_cycles: int = 40) -> None:
    for k, pats in enumerate(stages):
        proj.vary(*pats)
        if k < len(stages) - 1:
            proj.refine(max_cycles=15)
        else:
            proj.refine(max_cycles=last_cycles, ftol=1e-9, xtol=1e-10)


def test_tof_refinement_matches_gsas2() -> None:
    proj, fit = _build_project()
    _run_stages(proj, STAGES)
    rf = proj.rfactors()["total"]
    rv = fit["refinement"]["Rvals"]
    g_rwp, g_gof = float(rv["Rwp"]), float(rv["GOF"])
    print(
        f"[nac_tof_fit] Rwp: kaisetsu {100 * rf.rwp:.3f}%  GSAS-II {g_rwp:.3f}%;  "
        f"GOF: {rf.gof:.3f} vs {g_gof:.3f}"
    )
    assert abs(100 * rf.rwp - g_rwp) < 0.3
    assert rf.gof < g_gof * 1.01
    p = proj.params
    sig = fit["refinement"]["sig"]

    def esd_of(gsas_name: str, ours: str) -> float:
        return max(float(sig.get(gsas_name, 0.0) or 0.0), p[ours].esd or 0.0)

    # 格子定数(立方晶なので a だけ)
    g_a = float(fit["phase"]["cell"][0])
    ours = p["NAC:cell:a"]
    assert ours.esd is not None
    print(f"  a: {ours.value:.6f} ± {ours.esd:.6f} vs {g_a:.6f}")
    assert abs(ours.value - g_a) < max(3.0 * ours.esd, 2e-4)
    # ゼロ点・スケール・プロファイル
    inst = fit["histograms"][0]["instrument"]
    z = p["h0:inst:zero"]
    print(f"  Zero: {z.value:.3f} ± {z.esd or 0:.3f} vs {float(inst['Zero']):.3f} μs")
    assert abs(z.value - float(inst["Zero"])) < max(3.0 * esd_of(":0:Zero", z.name), 0.5)
    g_scale = float(fit["histograms"][0]["sample"]["Scale"][0]) / 100.0
    s = p["h0:NAC:scale"]
    print(f"  scale: {s.value:.5g} ± {s.esd or 0:.2g} vs {g_scale:.5g}")
    assert abs(s.value - g_scale) < max(3.0 * esd_of(":0:Scale", s.name) / 100.0, 5e-3 * g_scale)
    for g, o in (("sig-1", "sig1"), ("beta-0", "beta0")):
        ours = p[f"h0:inst:{o}"]
        g_val = float(inst[g])
        print(f"  {g}: {ours.value:.5g} ± {ours.esd or 0:.2g} vs {g_val:.5g}")
        assert abs(ours.value - g_val) < max(3.0 * esd_of(f":0:{g}", ours.name), 0.02 * abs(g_val))
    # 背景
    bk = fit["histograms"][0]["background"][0]
    for k in range(int(bk[2])):
        ours = p[f"h0:bkg:{k}"]
        g_esd = float(sig.get(f":0:Back;{k}", 0.0) or 0.0)
        tol = max(4.0 * max(g_esd, ours.esd or 0.0), 2e-2 * abs(float(bk[3])))
        print(f"  bkg {k}: {ours.value:.4f} vs {float(bk[3 + k]):.4f} (tol {tol:.4f})")
        assert abs(ours.value - float(bk[3 + k])) < tol
    # 原子座標と U_ij
    for ia, atom in enumerate(fit["phase"]["atoms"]):
        label = atom["label"]
        for k, nm in enumerate(("x", "y", "z")):
            ours = p[f"NAC:{label}:{nm}"]
            if ours.locked or ours.tie is not None:
                continue
            tol = max(3.0 * esd_of(f"0::dA{nm}:{ia}", ours.name), 2e-4)
            assert abs(ours.value - atom["xyz"][k]) < tol, (
                f"{label} {nm}: {ours.value} vs {atom['xyz'][k]} (tol {tol})"
            )
        parts = []
        for k, nm in enumerate(("u11", "u22", "u33", "u12", "u13", "u23")):
            ours = p[f"NAC:{label}:{nm}"]
            g_val = float(atom["Uij"][k])
            parts.append(f"{nm} {ours.value:.4f}/{g_val:.4f}")
            if ours.locked or ours.tie is not None:
                continue
            tol = max(3.0 * esd_of(f"0::A{nm.upper()}:{ia}", ours.name), 1e-3)
            assert abs(ours.value - g_val) < tol, (
                f"{label} {nm}: {ours.value} vs {g_val} (tol {tol})"
            )
        print(f"  {label}: " + ", ".join(parts))
    assert math.isfinite(rf.chi2)
