"""逐次精密化: 同じモデルで多数のデータ(温度・時間系列)を順に精密化し、結果を表にする。

前のデータの精密化結果を次の初期値にする(温度変化で格子定数が連続的に動く場合に収束が速い)。
失敗したデータは記録して次に進む。

精密化が失敗したか Rwp が前のデータより大きく悪化したときは、格子定数の初期値を ±1% 振って
谷を見つけ直してからもう一度精密化する(`kaisetsu.refine.cell_scan`)。相転移や大きな熱膨張で
前の値から外れたときに、以後のデータが全部失敗するのを防ぐ(相見, 電気化学 94, 264 (2026) の
CaMnTi2O6 の例: 300 K の格子定数を全温度に使うと 800 K 以上で解析不能)。
振り直しで例外が出ても 1 回目の結果が残っていればそれを採り(備考に記録)、2 回とも失敗したか
Rwp が悪いままなら、発散した値ではなく前のデータの値を次の初期値にする。

使い方:
    proj = ...(1 つ目のデータで精密化済みのプロジェクト)
    table = refine_sequence(proj, ["T100.xy", "T200.xy", ...], recipe=None,
                            params=["Si:cell:a", "h0:inst:zero"])
    table.to_csv("sequence.csv")
"""

from __future__ import annotations

import contextlib
import csv
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kaisetsu.io.readers import read_powder_data
from kaisetsu.refine.cell_scan import CellScanResult, scan_all_cells
from kaisetsu.refine.lsq import FitResult, convergence_note
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
        """表を CSV に書く。最後の列 note は備考(未収束、振り直し、失敗の理由)。"""
        p = Path(path)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["name", "ok", "Rwp", "chi2"]
            for n in self.param_names:
                header += [n, f"esd({n})"]
            header.append("note")
            w.writerow(header)
            for r in self.rows:
                row: list[object] = [r.name, int(r.ok), f"{r.rwp:.5f}", f"{r.chi2:.4f}"]
                for n in self.param_names:
                    v = r.values.get(n)
                    e = r.esds.get(n)
                    row += ["" if v is None else f"{v:.8g}", "" if e is None else f"{e:.3g}"]
                row.append(r.message)
                w.writerow(row)
        return p

    def summary(self) -> str:
        lines = [
            f"{'name':24s} {'Rwp%':>7s} {'χ²':>7s}  "
            + "  ".join(f"{n:>16s}" for n in self.param_names)
            + "  備考"
        ]
        for r in self.rows:
            vals = "  ".join(f"{r.values.get(n, float('nan')):16.6g}" for n in self.param_names)
            flag = "" if r.ok else "  (失敗)"
            note = f"  {r.message}" if r.message else ""
            lines.append(f"{r.name[:24]:24s} {100 * r.rwp:7.3f} {r.chi2:7.3f}  {vals}{flag}{note}")
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
    cell_scan: bool = True,
    progress: Callable[[str], None] | None = None,
) -> SequenceResult:
    """ヒストグラム hist のデータを順に差し替えて精密化する。

    Args:
        project: モデルと装置が入ったプロジェクト。hist のデータを順に差し替える
        paths: データファイル
        recipe: 各データで走らせる手順(None なら現在 vary の立っているパラメータで refine)
        params: 表に出すパラメータ名(None なら精密化したもの全部)
        reset_background: 真ならデータごとに背景を観測から推定し直す
        cell_scan: 真なら、失敗したか Rwp が前のデータの 1.5 倍 + 2 ポイントより悪いときに
            格子定数の初期値を ±1% 振って谷を見つけ直し、もう一度精密化する
    """
    result = SequenceResult()
    h = project.histograms[hist]
    vary_names = project.params.free_names()
    prev_rwp: float | None = None

    real_fit: FitResult | None = None  # 本番の精密化の結果(esd 付け直しの 1 サイクルは除く)

    def _refine_once() -> float:
        nonlocal real_fit
        if recipe is not None:
            recipe.run(project)
        else:
            if vary_names:
                project.fix_all()
                project.vary(*[n for n in vary_names if n in project.params])
            project.refine(max_cycles=max_cycles)
        real_fit = project.last_fit
        return project.rfactors()[h.name].rwp

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
            start = project.snapshot()  # 前のデータの結果(失敗したらここに戻す)
            notes: list[str] = []
            try:
                rwp = _refine_once()
                failure: Exception | None = None
            except Exception as exc:
                rwp, failure = float("inf"), exc

            def is_bad(value: float, ref: float | None = prev_rwp) -> bool:
                """前のデータの Rwp(ref)より 1.5 倍 + 2 ポイント悪ければ「悪化」。"""
                return ref is not None and value > 1.5 * ref + 0.02

            bad = failure is not None or is_bad(rwp)
            if cell_scan and bad:
                # 前の値から外れた(相転移・大きな熱膨張)。谷を見つけ直してやり直す。
                # 振り直しで例外が出ても、1 回目が成功していればその結果を採る
                end = project.snapshot()
                fit_first = real_fit
                rwp2 = float("inf")
                retry_error: Exception | None = None
                scans: list[CellScanResult] = []
                try:
                    project.restore(start)
                    scans = list(scan_all_cells(project))
                    rwp2 = _refine_once()
                except Exception as exc:
                    retry_error = exc
                if rwp2 < rwp:
                    rwp, failure = rwp2, None
                    notes.append(
                        "格子定数の初期値を振り直して再精密化: "
                        + "; ".join(f"{r.phase}: {r.message}" for r in scans if r.names)
                    )
                elif failure is None:
                    project.restore(end)
                    real_fit = fit_first
                    with contextlib.suppress(Exception):
                        project.refine(max_cycles=1)  # esd を付け直す(収束判定には使わない)
                    if retry_error is not None:
                        notes.append(f"振り直しの精密化でエラー({retry_error})。1 回目の結果を採用")
                    else:
                        notes.append("振り直しても改善せず。1 回目の結果を採用")
                elif retry_error is not None and retry_error is not failure:
                    notes.append(f"振り直しの精密化でもエラー({retry_error})")
            if failure is not None:
                # 2 回とも失敗。発散した値を次のデータの初期値にせず、前のデータの値に戻す
                project.restore(start)
                raise failure
            unconverged = convergence_note(real_fit, short=True)
            if unconverged:
                # 評価回数の上限で止まった行は備考で分かるようにする(レビュー 2026-10-17 #5)
                notes.append(unconverged)
            rf = project.rfactors()[h.name]
            names = list(params) if params is not None else project.params.free_names()
            if not result.param_names:
                result.param_names = names
            still_bad = is_bad(rf.rwp)
            if still_bad:
                notes.append(
                    f"Rwp が前のデータ({100 * (prev_rwp or 0.0):.2f}%)より大きく悪化。"
                    "次のデータは前のデータの値から始めます"
                )
            row = SequenceRow(
                name=p.stem,
                path=str(p),
                ok=True,
                rwp=rf.rwp,
                chi2=rf.chi2,
                values={n: project.params[n].value for n in names if n in project.params},
                esds={n: project.params[n].esd for n in names if n in project.params},
                elapsed=time.perf_counter() - t0,
                message="; ".join(notes),
            )
            if still_bad:
                # 悪い値(相転移で合わないモデルなど)を引きずらないよう、前のデータの値に戻す
                project.restore(start)
            else:
                prev_rwp = rf.rwp
        except Exception as exc:
            row = SequenceRow(p.stem, str(p), False, float("nan"), float("nan"), {}, {},
                              time.perf_counter() - t0, str(exc))  # fmt: skip
        result.rows.append(row)
        if progress is not None:
            progress(
                f"{row.name}: "
                + (
                    f"Rwp = {100 * row.rwp:.3f}%  χ² = {row.chi2:.3f}"
                    + (f"  ({row.message})" if row.message else "")
                    if row.ok
                    else f"失敗: {row.message}"
                )
            )
    return result
