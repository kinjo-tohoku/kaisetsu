"""PbSO4(実験室 X 線、IUCr ラウンドロビンデータ)の Rietveld 精密化の例。

    uv run python examples/refine_pbso4.py

標準レシピで段階的に精密化し、結果を表示して、プロジェクトファイルと図を保存する。
"""

from __future__ import annotations

from pathlib import Path

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.io.project import save_project
from kaisetsu.refine.recipe import standard_recipe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "pbso4"
OUT = ROOT / "examples" / "output"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    structure = Structure.from_cif(str(DATA / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(DATA / "PBSO4.XRA")

    # 実験室 X 線 (Cu Kα1/Kα2)、Bragg-Brentano、グラファイトモノクロメータ相当の偏光
    instrument = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5),
        u=0.001, v=-0.001, w=0.003, s_l=0.01, h_l=0.01, polarization=0.7,
    )  # fmt: skip
    project = Project("PbSO4 X-ray")
    project.add_phase(structure, name="PbSO4")
    project.add_histogram(data, instrument, SampleGeometry(), n_background=6, x_max=159.975)
    pc = project.phase_contribution(0, "PbSO4")
    pc.scale = 0.02  # 粗い初期値。最初の段階で決まる
    project.build_parameters()

    def report(step, fit):
        rf = project.rfactors()["total"]
        print(
            f"[{step.name}] 変数 {fit.n_params:2d}  Rwp = {100 * rf.rwp:6.3f}%  χ² = {rf.chi2:6.3f}"
        )

    recipe = standard_recipe("PbSO4", "h0")
    recipe.run(project, progress=report)

    print()
    print(project.summary())
    rep = project.reflection_report(0, "PbSO4")
    print(
        f"\nR_Bragg = {100 * rep.r_bragg:.2f}%  R_F = {100 * rep.r_f:.2f}%  (反射数 {len(rep.hkl)})"
    )

    path = save_project(project, OUT / "pbso4_xray.kaisetsu.json")
    print(f"\nプロジェクトを保存しました: {path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib が無いので図は省略します(uv sync --extra gui で入ります)")
        return
    h = project.histograms[0]
    res = project.calculate(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(h.data.x, h.data.y, "k+", ms=3, label="観測")
    ax.plot(res.x, res.y_calc, "r-", lw=0.8, label="計算")
    offset = -0.15 * h.data.y.max()
    ax.plot(res.x, h.data.y - res.y_calc + offset, "b-", lw=0.6, label="差")
    ax.vlines(rep.two_theta, offset * 0.4, offset * 0.6, color="g", lw=0.6, label="Bragg 位置")
    ax.set_xlabel("2θ (deg)")
    ax.set_ylabel("強度 (counts)")
    ax.set_title(f"PbSO4 (Cu Kα)  Rwp = {100 * project.rfactors()['total'].rwp:.2f}%")
    ax.legend()
    fig.tight_layout()
    png = OUT / "pbso4_xray.png"
    fig.savefig(png, dpi=130)
    print(f"図を保存しました: {png}")


if __name__ == "__main__":
    main()
