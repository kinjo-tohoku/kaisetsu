"""精密化をバックグラウンドスレッドで実行するワーカー。"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from kaisetsu.refine.lsq import convergence_note
from kaisetsu.refine.project import Project
from kaisetsu.refine.recipe import Recipe


class RefinementCancelled(Exception):
    pass


class RefineWorker(QObject):
    """Project.refine / Recipe.run を別スレッドで走らせる。

    Signals:
        progress(str): ログに出す 1 行
        iteration(int, float, object): 反復番号、χ²、曲線
            {"hist": [(x, y_calc, y_bkg), ...], "pdf": [(r, g_calc), ...]}
        step_started(str): レシピの段階が始まった(χ² 推移の区切り用)
        finished(bool, str): 成否とメッセージ
    """

    progress = Signal(str)
    iteration = Signal(int, float, object)
    step_started = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        project: Project,
        recipe: Recipe | None = None,
        max_cycles: int = 20,
        *,
        companion: Callable[[Callable[[str], None]], str] | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.recipe = recipe
        self.max_cycles = max_cycles
        #: レシピの後に走らせる付随処理(ウィザードの Le Bail 並走)。ログに出す文を返す
        self.companion = companion
        self._cancel = False
        self._step_name = ""

    def cancel(self) -> None:
        self._cancel = True

    def _curves(self) -> dict[str, list[tuple[np.ndarray, ...]]]:
        hist: list[tuple[np.ndarray, ...]] = []
        for h in self.project.histograms:
            r = h.result
            if r is None:
                continue
            hist.append((r.x, r.y_calc.copy(), r.y_background.copy()))
        pdf: list[tuple[np.ndarray, ...]] = []
        for g in self.project.pdf_datasets:
            pr = g.result
            if pr is None:
                continue
            pdf.append((pr.r, pr.g_calc.copy()))
        return {"hist": hist, "pdf": pdf}

    def _on_iteration(self, iteration: int, chi2: float) -> None:
        prefix = f"[{self._step_name}] " if self._step_name else ""
        self.progress.emit(f"  {prefix}反復 {iteration}: χ² = {chi2:.4f}")
        self.iteration.emit(iteration, chi2, self._curves())
        if self._cancel:
            raise RefinementCancelled()

    def _on_step_done(self, step: Any, fit: Any) -> None:
        rf = self.project.rfactors()["total"]
        label = "Rw" if self.project.pdf_datasets and not self.project.histograms else "Rwp"
        self.progress.emit(
            f"[{step.name}] 完了: 変数 {fit.n_params}  {label} = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}"
        )
        if self._cancel:
            raise RefinementCancelled()

    def run(self) -> None:
        try:
            if self.recipe is not None:
                # 段階ごとに走らせ、段階名を途中経過に出す
                self.project.fix_all()
                for step in self.recipe.steps:
                    self._step_name = step.name
                    self.progress.emit(f"[{step.name}] 開始")
                    self.step_started.emit(step.name)
                    Recipe(self.recipe.name, [step]).run(
                        self.project,
                        reset=False,
                        progress=self._on_step_done,
                        iteration_progress=self._on_iteration,
                    )
                    if self._cancel:
                        raise RefinementCancelled()
                self._step_name = ""
            else:
                self.project.refine(max_cycles=self.max_cycles, progress=self._on_iteration)
            rf = self.project.rfactors()["total"]
            label = "Rw" if self.project.pdf_datasets and not self.project.histograms else "Rwp"
            message = f"完了: {label} = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}"
            note = convergence_note(self.project.last_fit)
            if note:
                # 最後の段階が評価回数の上限で止まったときは「完了」に添えて知らせる
                message += "\n" + note
            if self.companion is not None:
                message += "\n" + self._run_companion()
            self.finished.emit(True, message)
        except RefinementCancelled:
            self.finished.emit(False, "中止しました(途中までの値が入っています)")
        except Exception as exc:
            self.finished.emit(False, f"エラー: {exc}\n{traceback.format_exc(limit=3)}")

    def _run_companion(self) -> str:
        """付随処理(Le Bail 並走など)。参考情報なので失敗しても本体の結果は残す。"""
        assert self.companion is not None
        self._step_name = "Le Bail 並走"
        self.progress.emit(
            "[Le Bail 並走] 開始(構造を使わずプロファイルと背景だけで合わせ、Rwp の下限を出します)"
        )
        self.step_started.emit(self._step_name)

        def _prog(msg: str) -> None:
            if self._cancel:
                raise RefinementCancelled()
            self.progress.emit(f"  [Le Bail 並走] {msg}")

        try:
            text = self.companion(_prog)
        except RefinementCancelled:
            raise
        except Exception as exc:
            text = f"Le Bail 並走でエラー: {exc}"
        self._step_name = ""
        self.progress.emit(text)
        return text


class RefineThread(QThread):
    """ワーカーを載せるスレッド。"""

    def __init__(self, worker: RefineWorker, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker = worker
        worker.moveToThread(self)
        self.started.connect(worker.run)
        worker.finished.connect(self.quit)
