"""逐次精密化: 同じモデルで多数のデータ(温度・時間系列)を順に精密化し、結果を表にする。

前のデータの精密化結果を次の初期値にする(温度変化で格子定数が連続的に動く場合に収束が速い)。
失敗したデータは記録して次に進む。

使い方:
    proj = ...(1 つ目のデータで精密化済みのプロジェクト)
    table = refine_sequence(proj, ["T100.xy", "T200.xy", ...], recipe=None,
                            params=["Si:cell:a", "h0:inst:zero"])
    table.to_csv("sequence.csv")
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kaisetsu.io.readers import read_powder_data
from kaisetsu.refine.project import Project
from kaisetsu.refine.recipe import Recipe


@dataclass
class SequenceRow:
    name: str
    path: str
    ok: bool
    rwp: float
    chi2: float
    values: dict[str, float]
    esds: dict[str, float | None]
    elapsed: float
    message: str = ""


@dataclass
class SequenceResult:
    rows: list[SequenceRow] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)

    def column(self, name: str) -> np.ndarray:
        return np.array([r.values.get(name, np.nan) for r in self.rows])

    def to_csv(self, path: str | Path) -> Path:
        p = Path(path)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["name", "ok", "Rwp", "chi2"]
            for n in self.param_names:
                header += [n, f"esd({n})"]
            w.writerow(header)
            for r in self.rows:
                row: list[object] = [r.name, int(r.ok), f"{r.rwp:.5f}", f"{r.chi2:.4f}"]
                for n in self.param_names:
                    v = r.values.get(n)
                    e = r.esds.get(n)
                    row += ["" if v is None else f"{v:.8g}", "" if e is None else f"{e:.3g}"]
                w.writerow(row)
        return p

    def summary(self) -> str:
        lines = [
            f"{'name':24s} {'Rwp%':>7s} {'χ²':>7s}  "
            + "  ".join(f"{n:>16s}" for n in self.param_names)
        ]
        for r in self.rows:
            vals = "  ".join(f"{r.values.get(n, float('nan')):16.6g}" for n in self.param_names)
            flag = "" if r.ok else "  (失敗)"
            lines.append(f"{r.name[:24]:24s} {100 * r.rwp:7.3f} {r.chi2:7.3f}  {vals}{flag}")
        return "\n".join(lines)


def refine_sequence(
    project: Project,
    paths: Iterable[str | Path],
    *,
    hist: int = 0,
    recipe: Recipe | None = None,
    params: Sequence[str] | None = None,
    max_cycles: int = 20,
    reset_background: bool = True,
    progress: Callable[[str], None] | None = None,
) -> SequenceResult:
    """ヒストグラム hist のデータを順に差し替えて精密化する。

    Args:
        project: モデルと装置が入ったプロジェクト。hist のデータを順に差し替える
        paths: データファイル
        recipe: 各データで走らせる手順(None なら現在 vary の立っているパラメータで refine)
        params: 表に出すパラメータ名(None なら精密化したもの全部)
        reset_background: 真ならデータごとに背景を観測から推定し直す
    """
    result = SequenceResult()
    h = project.histograms[hist]
    vary_names = project.params.free_names()
    for path in paths:
        p = Path(path)
        t0 = time.perf_counter()
        try:
            data = read_powder_data(p)
            h.data = data
            h.x_min = max(h.x_min, float(data.x[0]))
            h.x_max = min(h.x_max, float(data.x[-1]))
            h.result = None
            if reset_background:
                project.reset_background(hist, None, True)
            project.build_parameters()
            if recipe is not None:
                recipe.run(project)
            else:
                if vary_names:
                    project.fix_all()
                    project.vary(*[n for n in vary_names if n in project.params])
                project.refine(max_cycles=max_cycles)
            rf = project.rfactors()[h.name]
            names = list(params) if params is not None else project.params.free_names()
            if not result.param_names:
                result.param_names = names
            row = SequenceRow(
                name=p.stem,
                path=str(p),
                ok=True,
                rwp=rf.rwp,
                chi2=rf.chi2,
                values={n: project.params[n].value for n in names if n in project.params},
                esds={n: project.params[n].esd for n in names if n in project.params},
                elapsed=time.perf_counter() - t0,
            )
        except Exception as exc:
            row = SequenceRow(p.stem, str(p), False, float("nan"), float("nan"), {}, {},
                              time.perf_counter() - t0, str(exc))  # fmt: skip
        result.rows.append(row)
        if progress is not None:
            progress(
                f"{row.name}: "
                + (
                    f"Rwp = {100 * row.rwp:.3f}%  χ² = {row.chi2:.3f}"
                    if row.ok
                    else f"失敗: {row.message}"
                )
            )
    return result
