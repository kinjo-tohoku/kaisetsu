"""公開サンプルデータ(data/public)を標準レシピで精密化する例。

    uv run python examples/refine_sample_data.py fap     # 蛍石燐灰石、実験室 Cu Kα(GSAS-II LabData)
    uv run python examples/refine_sample_data.py lab6    # LaB6、Bruker .raw、Cu Kα
    uv run python examples/refine_sample_data.py garnet  # Y3(Fe,Al)5O12、ILL D1A 中性子 1.909 Å
    uv run python examples/refine_sample_data.py nac     # Na2Ca3Al2F14 + CaF2、放射光 11-BM
    uv run python examples/refine_sample_data.py ni_pdf  # Ni の PDF(X 線 G(r)、PDFfit2 相当の実空間フィット)
    uv run python examples/refine_sample_data.py ni_pdf_neutron  # 同、中性子
    uv run python examples/refine_sample_data.py all

GUI のウィザードで同じデータを選んだときと同じ初期値(プリセット相当)から始める。
結果は examples/output/ にプロジェクトファイルとして保存する。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    pdf_recipe,
    read_pdf_data,
    read_powder_data,
)
from kaisetsu.io.project import save_project
from kaisetsu.pattern.instrument import GeometryKind
from kaisetsu.profile.corrections import polarization_from_monochromator
from kaisetsu.refine.recipe import Recipe, standard_recipe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public"
OUT = ROOT / "examples" / "output"


def lab_cu_kalpha(polarization: float) -> CWInstrument:
    """GUI のプリセット「実験室 X 線 Cu Kα (Bragg-Brentano)」と同じ初期値。"""
    return CWInstrument(
        Radiation.xray(1.5405929, 1.5444274, 0.5),
        u=0.004, v=-0.003, w=0.004, x=0.02, y=0.02, s_l=0.02, h_l=0.02,
        polarization=polarization,
    )  # fmt: skip


def setup_fap() -> tuple[Project, Recipe]:
    d = DATA / "fap"
    project = Project("Fluorapatite lab X-ray")
    project.add_phase(Structure.from_cif(str(d / "fluorapatite_COD9001232.cif")), name="FAP")
    project.add_histogram(
        read_powder_data(d / "FAP.XRA"),
        lab_cu_kalpha(polarization_from_monochromator(26.6)),
        SampleGeometry(GeometryKind.BRAGG_BRENTANO),
        n_background=10,
    )
    return project, standard_recipe("*", "h0", sample_displacement=True)


def setup_lab6() -> tuple[Project, Recipe]:
    d = DATA / "lab6"
    project = Project("LaB6 Bruker D8")
    project.add_phase(Structure.from_cif(str(d / "LaB6_COD1000055.cif")), name="LaB6")
    data = read_powder_data(d / "LaB6_Jan2018.raw")
    project.add_histogram(
        data,
        lab_cu_kalpha(0.5),  # Ni フィルタ相当。モノクロメータ付きなら 0.78 程度
        SampleGeometry(GeometryKind.BRAGG_BRENTANO, radius_mm=250.0),
        n_background=8,
    )
    return project, standard_recipe("*", "h0", sample_displacement=True)


def setup_garnet() -> tuple[Project, Recipe]:
    d = DATA / "garnet"
    project = Project("YIAG neutron D1A")
    project.add_phase(Structure.from_cif(str(d / "Y3Fe2Al3O12_start.cif")), name="garnet")
    # inst_d1a.prm の U, V, W(GSAS の centideg² 単位)から換算
    inst = CWInstrument.from_gsas2(
        {"Type": "PNC", "Lam": 1.909, "Zero": 0.0, "U": 354.031, "V": -760.404, "W": 651.592,
         "X": 0.0, "Y": 0.0, "Z": 0.0, "SH/L": 0.002, "Polariz.": 0.0}
    )  # fmt: skip
    project.add_histogram(
        read_powder_data(d / "garnet.raw"),
        inst,
        SampleGeometry(GeometryKind.DEBYE_SCHERRER),
        n_background=6,
    )
    # 混合サイト(Fe/Al)の占有率も動かす。Al1, Fe2 は Fe1, Al2 に自動で従属する。
    # 試料変位(キャピラリの x, y ずれ)を入れると Rwp が 6.4% → 4.2% まで下がる
    return project, standard_recipe("*", "h0", occupancy=True, sample_displacement=True)


def setup_nac(x_max: float = 30.0) -> tuple[Project, Recipe]:
    d = DATA / "nac"
    project = Project("NAC + CaF2 11-BM")
    project.add_phase(Structure.from_cif(str(d / "NAC.cif")), name="NAC")
    project.add_phase(Structure.from_cif(str(d / "CaF2.cif")), name="CaF2")
    # 11bm_gsas.prm の値から換算(λ = 0.413909 Å、偏光 0.99)
    inst = CWInstrument.from_gsas2(
        {"Type": "PXC", "Lam": 0.413909, "Zero": 0.0, "U": 1.163, "V": -0.126, "W": 0.063,
         "X": 0.173, "Y": 0.0, "Z": 0.0, "SH/L": 0.0022, "Polariz.": 0.99}
    )  # fmt: skip
    project.add_histogram(
        read_powder_data(d / "11BM_NAC.fxye"),
        inst,
        SampleGeometry(GeometryKind.DEBYE_SCHERRER, radius_mm=1000.0),
        n_background=10,
        x_max=x_max,
    )
    return project, standard_recipe("*", "h0")


def setup_ni_pdf(radiation: str = "xray") -> tuple[Project, Recipe]:
    d = DATA / "pdf"
    project = Project(f"Ni PDF ({radiation})")
    st = Structure.from_cif(str(d / "Ni.cif"))
    st.sites[0].u_iso = 0.005
    project.add_phase(st, name="Ni")
    fname = "Ni-xray.gr" if radiation == "xray" else "Ni-neutron.gr"
    data = read_pdf_data(d / fname)
    # Q_max は .gr のヘッダから(X 線 40、中性子 27 Å⁻¹)。Q_damp の初期値は装置の目安
    project.add_pdf(
        data, radiation, qdamp=0.06 if radiation == "xray" else 0.02, r_min=1.5, r_max=20.0
    )
    project.pdf_phase(0, "Ni").delta2 = 2.0
    project.build_parameters()
    return project, pdf_recipe("Ni", "g0")


SETUPS = {
    "fap": setup_fap,
    "lab6": setup_lab6,
    "garnet": setup_garnet,
    "nac": setup_nac,
    "ni_pdf": setup_ni_pdf,
    "ni_pdf_neutron": lambda: setup_ni_pdf("neutron"),
}


def run(key: str) -> None:
    OUT.mkdir(exist_ok=True)
    project, recipe = SETUPS[key]()
    t0 = time.perf_counter()

    def report(step: object, fit: object) -> None:
        rf = project.rfactors()["total"]
        name = getattr(step, "name", "")
        n = getattr(fit, "n_params", 0)
        label = "Rw" if project.pdf_datasets and not project.histograms else "Rwp"
        print(f"  [{name}] 変数 {n:2d}  {label} = {100 * rf.rwp:6.3f}%  χ²_red = {rf.chi2:7.3f}")

    print(f"=== {key}: {project.name} ===")
    recipe.run(project, progress=report)
    print(f"  所要時間 {time.perf_counter() - t0:.1f} s")
    print(project.summary())
    for name in project.phases if project.histograms else []:
        rep = project.reflection_report(0, name)
        print(
            f"  {name}: R_Bragg = {100 * rep.r_bragg:.2f}%  R_F = {100 * rep.r_f:.2f}%"
            f"  反射数 {len(rep.hkl)}"
        )
    for note in project.diagnostics():
        print(f"  注意: {note}")
    path = save_project(project, OUT / f"{key}.kaisetsu.json")
    print(f"  保存: {path}\n")


def main(argv: list[str]) -> None:
    keys = argv[1:] or ["fap"]
    if keys == ["all"]:
        keys = list(SETUPS)
    for k in keys:
        if k not in SETUPS:
            sys.exit(f"不明なデータ名 {k!r}。選べるのは {', '.join(SETUPS)} か all")
        run(k)


if __name__ == "__main__":
    main(sys.argv)
