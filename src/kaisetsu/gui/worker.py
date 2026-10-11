"""精密化をバックグラウンドスレッドで実行するワーカー。"""

from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from kaisetsu.refine.project import Project
from kaisetsu.refine.recipe import Recipe


class RefinementCancelled(Exception):
    pass


class RefineWorker(QObject):
    """Project.refine / Recipe.run を別スレッドで走らせる。"""

    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self, project: Project, recipe: Recipe | None = None, max_cycles: int = 20
    ) -> None:
        super().__init__()
        self.project = project
        self.recipe = recipe
        self.max_cycles = max_cycles
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _check(self, iteration: int, chi2: float) -> None:
        self.progress.emit(f"  反復 {iteration}: χ² = {chi2:.4f}")
        if self._cancel:
            raise RefinementCancelled()

    def run(self) -> None:
        try:
            if self.recipe is not None:

                def on_step(step: Any, fit: Any) -> None:
                    rf = self.project.rfactors()["total"]
                    self.progress.emit(
                        f"[{step.name}] 変数 {fit.n_params}  Rwp = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}"
                    )
                    if self._cancel:
                        raise RefinementCancelled()

                self.recipe.run(self.project, progress=on_step)
            else:
                self.project.refine(max_cycles=self.max_cycles, progress=self._check)
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
