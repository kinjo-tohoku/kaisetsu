"""精密化をバックグラウンドスレッドで実行するワーカー。"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from kaisetsu.refine.project import Project
from kaisetsu.refine.recipe import Recipe


class RefinementCancelled(Exception):
    pass


class RefineWorker(QObject):
    """Project.refine / Recipe.run を別スレッドで走らせる。

    Signals:
        progress(str): ログに出す 1 行
        iteration(int, float, object): 反復番号、χ²、ヒストグラムごとの (x, y_calc, y_bkg) のリスト
        finished(bool, str): 成否とメッセージ
    """

    progress = Signal(str)
    iteration = Signal(int, float, object)
    finished = Signal(bool, str)

    def __init__(
        self, project: Project, recipe: Recipe | None = None, max_cycles: int = 20
    ) -> None:
        super().__init__()
        self.project = project
        self.recipe = recipe
        self.max_cycles = max_cycles
        self._cancel = False
        self._step_name = ""

    def cancel(self) -> None:
        self._cancel = True

    def _curves(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out = []
        for h in self.project.histograms:
            r = h.result
            if r is None:
                continue
            out.append((r.x, r.y_calc.copy(), r.y_background.copy()))
        return out

    def _on_iteration(self, iteration: int, chi2: float) -> None:
        prefix = f"[{self._step_name}] " if self._step_name else ""
        self.progress.emit(f"  {prefix}反復 {iteration}: χ² = {chi2:.4f}")
        self.iteration.emit(iteration, chi2, self._curves())
        if self._cancel:
            raise RefinementCancelled()

    def _on_step_done(self, step: Any, fit: Any) -> None:
        rf = self.project.rfactors()["total"]
        self.progress.emit(
            f"[{step.name}] 完了: 変数 {fit.n_params}  Rwp = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}"
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
            self.finished.emit(True, f"完了: Rwp = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}")
        except RefinementCancelled:
            self.finished.emit(False, "中止しました(途中までの値が入っています)")
        except Exception as exc:
            self.finished.emit(False, f"エラー: {exc}\n{traceback.format_exc(limit=3)}")


class RefineThread(QThread):
    """ワーカーを載せるスレッド。"""

    def __init__(self, worker: RefineWorker, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker = worker
        worker.moveToThread(self)
        self.started.connect(worker.run)
        worker.finished.connect(self.quit)
