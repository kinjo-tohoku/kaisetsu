"""MEM(最大エントロピー法)のダイアログ: 条件 → 実行(別スレッド)→ 断面表示 → 書き出し。"""

from __future__ import annotations

import math
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.mem.core import MEMInput, MEMResult, run_mem
from kaisetsu.mem.io import density_slice, write_ccp4, write_grd, write_xplor
from kaisetsu.mem.mpf import observed_structure_factors, run_mpf
from kaisetsu.refine.project import Project


class _MEMWorker(QObject):
    progress = Signal(str)
    finished = Signal(bool, str, object)

    def __init__(
        self, project: Project, hist: int, phase: str, grid: tuple[int, int, int],
        source: str, mpf_cycles: int, two_channel: bool | None, group_overlaps: bool = False,
    ) -> None:  # fmt: skip
        super().__init__()
        self.project = project
        self.hist = hist
        self.phase = phase
        self.grid = grid
        self.source = source
        self.mpf_cycles = mpf_cycles
        self.two_channel = two_channel
        self.group_overlaps = group_overlaps

    def run(self) -> None:
        try:
            proj = self.project
            if self.mpf_cycles > 0:
                res = run_mpf(
                    proj, self.hist, self.phase, grid=self.grid, n_cycles=self.mpf_cycles,
                    progress=self.progress.emit, group_overlaps=self.group_overlaps,
                )  # fmt: skip
                mem = res.final
                self.finished.emit(True, f"MPF 完了: Rwp {100 * res.rwp_history[-1]:.3f}%", mem)
                return
            if self.source == "pawley":
                hkl, f2, sig = proj.pawley_structure_factors(self.hist, self.phase)
                obs = observed_structure_factors(proj, self.hist, self.phase)
                f_abs = np.sqrt(np.maximum(f2, 0.0))
                unit = np.where(
                    np.abs(obs.f_obs) > 0, obs.f_obs / np.maximum(np.abs(obs.f_obs), 1e-300), 1.0
                )
                sigma = np.where(
                    np.isfinite(sig) & (f_abs > 0), sig / (2.0 * np.maximum(f_abs, 1e-300)), np.inf
                )
                sigma = np.where(
                    np.isfinite(sigma),
                    np.maximum(sigma, 0.01 * np.maximum(f_abs, 1e-3)),
                    1e3 * max(float(f_abs.max()), 1.0),
                )
                ph = proj.phases[self.phase]
                inp = MEMInput(
                    ph.structure.lattice, ph.structure.spacegroup, hkl, f_abs * unit, sigma,
                    obs.f000, self.grid, two_channel=bool(self.two_channel),
                    f000_negative=obs.f000_negative if self.two_channel else 0.0,
                )  # fmt: skip
                mem = run_mem(
                    inp,
                    progress=lambda i, lam, c: self.progress.emit(
                        f"  外側反復 {i}: λ = {lam:.3g}  C = {c:.3f}"
                    ),
                )
            else:
                from kaisetsu.mem.mpf import mem_from_project

                mem = mem_from_project(
                    proj, self.hist, self.phase, self.grid, two_channel=self.two_channel,
                    group_overlaps=self.group_overlaps,
                    progress=lambda i, lam, c: self.progress.emit(f"  外側反復 {i}: λ = {lam:.3g}  C = {c:.3f}"),
                )  # fmt: skip
            self.finished.emit(True, mem.message, mem)
        except Exception as exc:
            self.finished.emit(False, f"エラー: {exc}\n{traceback.format_exc(limit=3)}", None)


class DensityView(QWidget):
    """密度の 2D 断面(軸と位置を選ぶ)。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.axis = QComboBox()
        for i, lab in enumerate(("a 軸に垂直 (bc 面)", "b 軸に垂直 (ac 面)", "c 軸に垂直 (ab 面)")):
            self.axis.addItem(lab, i)
        self.axis.setCurrentIndex(2)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.pos_label = QLabel("z = 0.00")
        row.addWidget(QLabel("断面:"))
        row.addWidget(self.axis)
        row.addWidget(self.slider, 1)
        row.addWidget(self.pos_label)
        layout.addLayout(row)
        self.image = pg.ImageView()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        layout.addWidget(self.image, 1)
        self._density: np.ndarray | None = None
        self.axis.currentIndexChanged.connect(self.refresh)
        self.slider.valueChanged.connect(self.refresh)

    def set_density(self, density: np.ndarray | None) -> None:
        self._density = density
        self.refresh()

    def refresh(self) -> None:
        if self._density is None:
            self.image.clear()
            return
        ax = int(self.axis.currentData())
        frac = self.slider.value() / 100.0
        sl, _ = density_slice(self._density, ax, frac)
        self.pos_label.setText(f"{'abc'[ax]} = {frac:.2f}")
        self.image.setImage(
            sl, autoLevels=False, levels=(0.0, float(np.percentile(self._density, 99.5)))
        )


class MEMDialog(QDialog):
    """MEM の実行と結果の確認。"""

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.result: MEMResult | None = None
        self._thread: QThread | None = None
        self._worker: _MEMWorker | None = None
        self.setWindowTitle("MEM 電子密度 / 核密度")
        self.resize(1000, 700)
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 0)

        cond = QGroupBox("条件")
        cf = QFormLayout(cond)
        self.hist = QComboBox()
        for i, h in enumerate(project.histograms):
            self.hist.addItem(f"h{i}: {h.name}", i)
        cf.addRow("ヒストグラム", self.hist)
        self.phase = QComboBox()
        for n in project.phases:
            self.phase.addItem(n, n)
        self.phase.currentIndexChanged.connect(self._suggest_grid)
        cf.addRow("相", self.phase)
        self.source = QComboBox()
        self.source.addItem("Rietveld 分配(位相は構造モデル)", "rietveld")
        self.source.addItem("Pawley 法の強度(σ と相関込み。相を Pawley にしておく)", "pawley")
        cf.addRow("観測 |F| の作り方", self.source)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.02, 1.0)
        self.spacing.setDecimals(3)
        self.spacing.setSingleStep(0.01)
        self.spacing.setValue(0.1)
        self.spacing.valueChanged.connect(self._suggest_grid)
        cf.addRow("格子点の間隔 [Å]", self.spacing)
        self.grid = [QSpinBox() for _ in range(3)]
        row = QHBoxLayout()
        for sb in self.grid:
            sb.setRange(8, 512)
            row.addWidget(sb)
        cf.addRow("格子点数 (a, b, c)", row)
        self.two_channel = QCheckBox("2 チャンネル(負の散乱長がある中性子データ)")
        cf.addRow(self.two_channel)
        self.group_overlaps = QCheckBox("重なった反射は強度の和だけを拘束する(G 拘束)")
        self.group_overlaps.setToolTip(
            "位置の差が FWHM の半分以内で連なる反射を群にし、群の強度和だけを拘束します。"
            "分配の仮定(モデルの強度比)を MEM に持ち込まないための選択肢"
        )
        cf.addRow(self.group_overlaps)
        self.mpf = QSpinBox()
        self.mpf.setRange(0, 20)
        self.mpf.setValue(0)
        self.mpf.setToolTip("0 なら MEM のみ。1 以上で MEM ↔ パターンフィッティングを往復(MPF)")
        cf.addRow("MPF のサイクル数", self.mpf)
        left.addWidget(cond)
        self.run_btn = QPushButton("MEM を実行")
        self.run_btn.clicked.connect(self.run)
        left.addWidget(self.run_btn)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        left.addWidget(self.log, 1)
        exp = QGroupBox("書き出し(VESTA で開けます)")
        ef = QVBoxLayout(exp)
        self.btn_ccp4 = QPushButton("CCP4 (.ccp4)…")
        self.btn_xplor = QPushButton("XPLOR (.xplor)…")
        self.btn_grd = QPushButton("VESTA grd (.grd)…")
        for b, kind in ((self.btn_ccp4, "ccp4"), (self.btn_xplor, "xplor"), (self.btn_grd, "grd")):
            b.clicked.connect(lambda _=False, k=kind: self.export(k))
            b.setEnabled(False)
            ef.addWidget(b)
        left.addWidget(exp)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        left.addWidget(buttons)

        right = QVBoxLayout()
        root.addLayout(right, 1)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        right.addWidget(self.summary)
        self.view = DensityView()
        right.addWidget(self.view, 1)
        self._suggest_grid()
        if not project.histograms or not project.phases:
            self.run_btn.setEnabled(False)
            self.log.appendPlainText(
                "先に粉末データと相を読み込み、Rietveld 精密化を行ってください"
            )

    def _suggest_grid(self) -> None:
        name = self.phase.currentData()
        if not name:
            return
        lat = self.project.phases[str(name)].structure.lattice
        d = float(self.spacing.value())
        for sb, length in zip(self.grid, (lat.a, lat.b, lat.c), strict=True):
            n = math.ceil(length / d)
            n += n % 2  # 偶数に
            sb.setValue(max(8, min(n, 512)))

    def run(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        grid = tuple(int(sb.value()) for sb in self.grid)
        self._worker = _MEMWorker(
            self.project, int(self.hist.currentData()), str(self.phase.currentData()),
            grid, str(self.source.currentData()), int(self.mpf.value()),
            True if self.two_channel.isChecked() else None,
            self.group_overlaps.isChecked(),
        )  # fmt: skip
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.finished.connect(self._thread.quit)
        self.run_btn.setEnabled(False)
        self.log.appendPlainText(f"MEM 開始: 格子 {grid}")
        self._thread.start()

    def _done(self, ok: bool, message: str, result: object) -> None:
        self.run_btn.setEnabled(True)
        self.log.appendPlainText(message)
        if not ok or not isinstance(result, MEMResult):
            return
        self.result = result
        mem = result
        lines = [
            f"C = {mem.chi2:.3f}, R_F = {100 * mem.r_factor():.2f}%, 反射 {len(mem.hkl)} 本, "
            f"格子 {mem.grid}, {mem.elapsed:.0f} 秒",
            "密度の極大(分率座標、値):",
        ]
        for x, v in mem.maxima(8, 0.7):
            lines.append(f"  ({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})  {v:.1f}")
        self.summary.setText("\n".join(lines))
        self.view.set_density(mem.density)
        for b in (self.btn_ccp4, self.btn_xplor, self.btn_grd):
            b.setEnabled(True)

    def export(self, kind: str) -> None:
        if self.result is None:
            return
        filt = {"ccp4": "CCP4 (*.ccp4)", "xplor": "XPLOR (*.xplor)", "grd": "VESTA grd (*.grd)"}[
            kind
        ]
        path, _ = QFileDialog.getSaveFileName(self, "密度を書き出す", last_dir(), filt)
        if not path:
            return
        if not path.lower().endswith("." + kind):
            path += "." + kind
        mem = self.result
        if kind == "ccp4":
            write_ccp4(mem.density, mem.lattice, mem.spacegroup, path)
        elif kind == "xplor":
            write_xplor(mem.density, mem.lattice, path)
        else:
            write_grd(mem.density, mem.lattice, path)
        remember_dir(path)
        self.log.appendPlainText(f"書き出しました: {Path(path).name}")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(10000)
        event.accept()


__all__ = ["DensityView", "MEMDialog"]
