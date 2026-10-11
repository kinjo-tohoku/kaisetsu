"""メインウィンドウ。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu import __version__
from kaisetsu.crystal.cif import read_structure, write_structure
from kaisetsu.gui.dialogs import InstrumentDialog, PhaseNameDialog, RecipeDialog
from kaisetsu.gui.param_model import COL_DESC, COL_NAME, ParameterFilter, ParameterModel
from kaisetsu.gui.plot_widget import PatternPlot
from kaisetsu.gui.worker import RefineThread, RefineWorker
from kaisetsu.io.project import load_project, save_project
from kaisetsu.io.readers import read_powder_data
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.refine.project import HistogramEntry, Project
from kaisetsu.refine.recipe import Recipe

log = logging.getLogger(__name__)

DATA_FILTER = (
    "回折データ (*.xy *.xye *.dat *.txt *.gsa *.gsas *.fxye *.raw *.xra *.cwn *.csv);;すべて (*)"
)
CIF_FILTER = "CIF (*.cif);;すべて (*)"
PROJECT_FILTER = "kaisetsu プロジェクト (*.kaisetsu.json *.json);;すべて (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"kaisetsu {__version__} — 粉末回折 Rietveld 解析")
        self.resize(1400, 850)
        self.project = Project("新規プロジェクト")
        self.project_path: Path | None = None
        self.current_hist = 0
        self._thread: RefineThread | None = None
        self._worker: RefineWorker | None = None
        self.settings = QSettings("kaisetsu", "kaisetsu")

        self._build_ui()
        self._build_menu()
        self._refresh_all()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["プロジェクト"])
        self.tree.itemClicked.connect(self._on_tree_clicked)
        self.tree.setMinimumWidth(220)

        self.plot = PatternPlot()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        center = QSplitter(Qt.Orientation.Vertical)
        center.addWidget(self.plot)
        center.addWidget(self.log_view)
        center.setStretchFactor(0, 4)
        center.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # パラメータ表(ドック)
        dock = QDockWidget("パラメータ", self)
        dock.setObjectName("paramDock")
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        filt = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("絞り込み(例: *cell*, h0:inst:*, Pb)")
        self.only_free = QCheckBox("精密化中のみ")
        filt.addWidget(self.filter_edit, 1)
        filt.addWidget(self.only_free)
        v.addLayout(filt)
        quick = QHBoxLayout()
        for label, pat in (
            ("全て", "*"),
            ("格子", "*:cell:*"),
            ("原子", "*:*:[xyz]"),
            ("U", "*:*:u*"),
            ("装置", "h*:inst:*"),
            ("背景", "h*:bkg:*"),
            ("スケール", "h*:*:scale"),
        ):
            b = QPushButton(label)
            b.setFlat(True)
            b.clicked.connect(lambda _=False, p=pat: self.filter_edit.setText(p))
            quick.addWidget(b)
        quick.addStretch(1)
        v.addLayout(quick)
        self.param_model = ParameterModel(self.project.params)
        self.param_filter = ParameterFilter()
        self.param_filter.setSourceModel(self.param_model)
        self.table = QTableView()
        self.table.setModel(self.param_filter)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            COL_NAME, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(COL_DESC, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.btn_vary_shown = QPushButton("表示中を精密化する")
        self.btn_fix_shown = QPushButton("表示中を固定する")
        self.btn_vary_shown.clicked.connect(lambda: self._set_vary_shown(True))
        self.btn_fix_shown.clicked.connect(lambda: self._set_vary_shown(False))
        btns.addWidget(self.btn_vary_shown)
        btns.addWidget(self.btn_fix_shown)
        v.addLayout(btns)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setMinimumWidth(480)
        self.filter_edit.textChanged.connect(self.param_filter.set_pattern)
        self.only_free.toggled.connect(self.param_filter.set_only_free)

        self.status_label = QLabel("データと相を読み込んでください")
        self.statusBar().addWidget(self.status_label, 1)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("ファイル(&F)")
        self._add_action(
            m_file, "新規プロジェクト(&N)", self.new_project, QKeySequence.StandardKey.New
        )
        self._add_action(
            m_file, "プロジェクトを開く(&O)…", self.open_project, QKeySequence.StandardKey.Open
        )
        self._add_action(
            m_file, "プロジェクトを保存(&S)", self.save_project, QKeySequence.StandardKey.Save
        )
        self._add_action(
            m_file, "名前を付けて保存(&A)…", self.save_project_as, QKeySequence.StandardKey.SaveAs
        )
        m_file.addSeparator()
        self._add_action(m_file, "計算パターンを CSV に書き出す…", self.export_pattern_csv)
        self._add_action(m_file, "反射リストを CSV に書き出す…", self.export_reflections_csv)
        self._add_action(m_file, "相を CIF に書き出す…", self.export_cif)
        m_file.addSeparator()
        self._add_action(m_file, "終了(&X)", self.close, QKeySequence.StandardKey.Quit)

        m_data = mb.addMenu("データ(&D)")
        self._add_action(m_data, "回折データを読み込む(&L)…", self.load_data, "Ctrl+L")
        self._add_action(m_data, "CIF から相を追加(&P)…", self.add_phase_from_cif, "Ctrl+P")
        self._add_action(m_data, "装置と試料の設定(&I)…", self.edit_instrument, "Ctrl+I")
        m_data.addSeparator()
        self._add_action(m_data, "はじめての精密化(ウィザード)(&W)…", self.run_wizard)

        m_ref = mb.addMenu("精密化(&R)")
        self._add_action(m_ref, "パターンを計算(&C)", self.recalculate, "F5")
        self.act_refine = self._add_action(m_ref, "精密化を実行(&R)", self.refine, "F9")
        self.act_recipe = self._add_action(
            m_ref, "標準の手順で精密化(&S)…", self.refine_recipe, "Shift+F9"
        )
        self.act_stop = self._add_action(m_ref, "中止(&T)", self.stop_refine, "Esc")
        self.act_stop.setEnabled(False)
        m_ref.addSeparator()
        self._add_action(m_ref, "すべて固定(&F)", self.fix_all)
        self._add_action(m_ref, "結果の要約を表示(&U)", self.show_summary)

        m_help = mb.addMenu("ヘルプ(&H)")
        self._add_action(m_help, "パラメータ名の規則", self.show_naming_help)
        self._add_action(m_help, "バージョン情報(&A)", self.show_about)

    def _add_action(
        self,
        menu: QMenu,
        text: str,
        slot: Callable[..., Any],
        shortcut: QKeySequence | QKeySequence.StandardKey | str | None = None,
    ) -> QAction:
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ------------------------------------------------------------------
    # 表示更新
    # ------------------------------------------------------------------
    def log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _refresh_tree(self) -> None:
        self.tree.clear()
        root = QTreeWidgetItem([self.project.name or "プロジェクト"])
        root.setData(0, Qt.ItemDataRole.UserRole, ("project", None))
        self.tree.addTopLevelItem(root)
        phases = QTreeWidgetItem(["相"])
        root.addChild(phases)
        for ph in self.project.phases.values():
            st = ph.structure
            item = QTreeWidgetItem([f"{ph.name}  ({st.spacegroup.hm}, {st.n_sites} サイト)"])
            item.setData(0, Qt.ItemDataRole.UserRole, ("phase", ph.name))
            item.setToolTip(0, repr(st.lattice))
            phases.addChild(item)
        hists = QTreeWidgetItem(["ヒストグラム"])
        root.addChild(hists)
        for i, h in enumerate(self.project.histograms):
            rad = h.instrument.radiation
            kind = "中性子" if rad.is_neutron else "X 線"
            item = QTreeWidgetItem(
                [f"h{i}: {h.name}  ({kind} λ={rad.wavelength:.4f} Å, {len(h.data)} 点)"]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, ("hist", i))
            hists.addChild(item)
        self.tree.expandAll()

    def _refresh_params(self) -> None:
        self.param_model.set_params(self.project.params)

    def _refresh_plot(self) -> None:
        if not self.project.histograms:
            self.plot.set_data(None)
            return
        i = min(self.current_hist, len(self.project.histograms) - 1)
        h = self.project.histograms[i]
        self.plot.set_data(h.data, h.instrument.radiation.wavelength)
        if self.project.phases:
            try:
                res = self.project.calculate(i)
            except Exception as exc:
                self.log(f"計算に失敗しました: {exc}")
                return
            ticks = {name: res.peaks[k].two_theta for k, name in enumerate(h.contributions)}
            self.plot.set_result(res, ticks, h.mask())
            rf = self.project.rfactors()
            self.status_label.setText(f"h{i} {h.name}: {rf[h.name]}")

    def _refresh_all(self) -> None:
        self._refresh_tree()
        self._refresh_params()
        self._refresh_plot()
        has_model = bool(self.project.phases and self.project.histograms)
        self.act_refine.setEnabled(has_model)
        self.act_recipe.setEnabled(has_model)

    def _on_tree_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, ref = data
        if kind == "hist":
            self.current_hist = int(ref)
            self._refresh_plot()
            self.filter_edit.setText(f"h{ref}:*")
        elif kind == "phase":
            self.filter_edit.setText(f"{ref}:*")

    def _set_vary_shown(self, flag: bool) -> None:
        for r in range(self.param_filter.rowCount()):
            src = self.param_filter.mapToSource(self.param_filter.index(r, 0))
            p = self.param_model.parameter(src.row())
            if p.tie is None and not p.locked:
                p.vary = flag
        self.param_model.refresh()

    # ------------------------------------------------------------------
    # ファイル
    # ------------------------------------------------------------------
    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.project = Project("新規プロジェクト")
        self.project_path = None
        self.current_hist = 0
        self.log_view.clear()
        self._refresh_all()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "プロジェクトを開く", self._last_dir(), PROJECT_FILTER
        )
        if path:
            self.open_project_file(path)

    def open_project_file(self, path: str) -> None:
        try:
            self.project = load_project(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "読み込みエラー", f"プロジェクトを読み込めませんでした:\n{exc}"
            )
            return
        self.project_path = Path(path)
        self._remember_dir(path)
        self.current_hist = 0
        self.log(f"プロジェクトを開きました: {path}")
        for line in self.project.log[-20:]:
            self.log("  " + line)
        self._refresh_all()

    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        save_project(self.project, self.project_path)
        self.log(f"保存しました: {self.project_path}")

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "名前を付けて保存", self._last_dir(), PROJECT_FILTER
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".kaisetsu.json"
        self.project_path = Path(path)
        self._remember_dir(path)
        self.save_project()

    def export_pattern_csv(self) -> None:
        if not self.project.histograms:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "計算パターンを書き出す", self._last_dir(), "CSV (*.csv)"
        )
        if not path:
            return
        i = self.current_hist
        h = self.project.histograms[i]
        res = self.project.calculate(i)
        cols = [res.x, h.data.y, res.y_calc, res.y_background, h.data.y - res.y_calc]
        header = "two_theta,y_obs,y_calc,y_bkg,diff"
        for name, yp in zip(h.contributions, res.y_phases, strict=True):
            cols.append(yp)
            header += f",{name}"
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="")
        self.log(f"計算パターンを書き出しました: {path}")

    def export_reflections_csv(self) -> None:
        if not (self.project.histograms and self.project.phases):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "反射リストを書き出す", self._last_dir(), "CSV (*.csv)"
        )
        if not path:
            return
        i = self.current_hist
        with open(path, "w", encoding="utf-8") as f:
            f.write("phase,h,k,l,mult,d,two_theta,F2_calc,F2_obs,I_calc,I_obs,fwhm\n")
            for name in self.project.histograms[i].contributions:
                rep = self.project.reflection_report(i, name)
                for k in range(len(rep.hkl)):
                    hh, kk, ll = rep.hkl[k]
                    f.write(
                        f"{name},{hh},{kk},{ll},{rep.multiplicity[k]},{rep.d[k]:.5f},{rep.two_theta[k]:.4f},"
                        f"{rep.f2_calc[k]:.4f},{rep.f2_obs[k]:.4f},{rep.i_calc[k]:.5g},{rep.i_obs[k]:.5g},{rep.fwhm[k]:.4f}\n"
                    )
        self.log(f"反射リストを書き出しました: {path}")

    def export_cif(self) -> None:
        if not self.project.phases:
            return
        name = next(iter(self.project.phases))
        path, _ = QFileDialog.getSaveFileName(self, "CIF に書き出す", self._last_dir(), CIF_FILTER)
        if not path:
            return
        ph = self.project.phases[name]
        ph.sync()
        write_structure(ph.structure, path)
        self.log(f"CIF を書き出しました: {path}")

    # ------------------------------------------------------------------
    # データ・相・装置
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "回折データを読み込む", self._last_dir(), DATA_FILTER
        )
        if not path:
            return
        self.load_data_file(path)

    def load_data_file(
        self,
        path: str,
        instrument: CWInstrument | None = None,
        sample: SampleGeometry | None = None,
    ) -> None:
        try:
            data = read_powder_data(path)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みエラー", f"データを読み込めませんでした:\n{exc}")
            return
        self._remember_dir(path)
        if instrument is None:
            dlg = InstrumentDialog(None, None, self)
            dlg.setWindowTitle("このデータの装置と試料の設定")
            if dlg.exec() != InstrumentDialog.DialogCode.Accepted:
                return
            instrument, sample = dlg.result_instrument()
        h = self.project.add_histogram(data, instrument, sample, n_background=6)
        self.current_hist = len(self.project.histograms) - 1
        for pc in h.contributions.values():
            pc.scale = self._initial_scale(h)
        self.project.build_parameters()
        self.log(
            f"データを読み込みました: {path} ({len(data)} 点, {data.x[0]:.3f}–{data.x[-1]:.3f}°)"
        )
        self._refresh_all()

    def add_phase_from_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "CIF から相を追加", self._last_dir(), CIF_FILTER
        )
        if not path:
            return
        self.add_phase_file(path)

    def add_phase_file(self, path: str, name: str | None = None) -> None:
        try:
            st = read_structure(path)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みエラー", f"CIF を読み込めませんでした:\n{exc}")
            return
        self._remember_dir(path)
        if name is None:
            default = (st.name or Path(path).stem).replace(":", "_")
            dlg = PhaseNameDialog(default, self)
            if dlg.exec() != PhaseNameDialog.DialogCode.Accepted:
                return
            name = dlg.value() or default
        if name in self.project.phases:
            QMessageBox.warning(self, "重複", f"相 {name!r} はすでにあります")
            return
        self.project.add_phase(st, name=name)
        for h in self.project.histograms:
            h.contributions[name].scale = self._initial_scale(h)
        self.project.build_parameters()
        self.log(f"相を追加しました: {name} ({st.spacegroup.hm}, {st.lattice})")
        self._refresh_all()

    def _initial_scale(self, h: HistogramEntry) -> float:
        """観測強度の大きさからスケール因子の粗い初期値を作る。"""
        try:
            pc_list = list(h.contributions.values())
            if not pc_list:
                return 1.0
            save = [(pc.scale) for pc in pc_list]
            for pc in pc_list:
                pc.scale = 1.0
            res = self.project.calculate(self.project.histograms.index(h))
            for pc, s in zip(pc_list, save, strict=True):
                pc.scale = s
            peak = np.nanmax(res.y_bragg)
            obs = np.nanmax(h.data.y - np.median(h.data.y))
            return float(obs / peak) if peak > 0 else 1.0
        except Exception:
            return 1.0

    def edit_instrument(self) -> None:
        if not self.project.histograms:
            QMessageBox.information(self, "装置", "先に回折データを読み込んでください")
            return
        h = self.project.histograms[self.current_hist]
        dlg = InstrumentDialog(h.instrument, h.sample, self)
        if dlg.exec() != InstrumentDialog.DialogCode.Accepted:
            return
        inst, smp = dlg.result_instrument()
        h.instrument = inst
        h.sample = smp
        self.project.build_parameters()
        self.log("装置パラメータを更新しました")
        self._refresh_all()

    # ------------------------------------------------------------------
    # 精密化
    # ------------------------------------------------------------------
    def recalculate(self) -> None:
        self._refresh_plot()
        if self.project.histograms and self.project.phases:
            rf = self.project.rfactors()["total"]
            self.log(f"計算: {rf}")

    def fix_all(self) -> None:
        self.project.fix_all()
        self.param_model.refresh()

    def refine(self) -> None:
        if not self.project.params.free():
            QMessageBox.information(
                self,
                "精密化",
                "精密化するパラメータが選ばれていません。\n表の「精密化」列にチェックを入れるか、標準の手順を使ってください。",
            )
            return
        self._start_worker(RefineWorker(self.project, None, max_cycles=20))

    def refine_recipe(self) -> None:
        dlg = RecipeDialog(list(self.project.phases), self)
        if dlg.exec() != RecipeDialog.DialogCode.Accepted:
            return
        self._start_worker(RefineWorker(self.project, dlg.recipe()))

    def run_recipe(self, recipe: Recipe) -> None:
        self._start_worker(RefineWorker(self.project, recipe))

    def _start_worker(self, worker: RefineWorker) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "精密化", "すでに精密化が実行中です")
            return
        self._worker = worker
        self._thread = RefineThread(worker, self)
        worker.progress.connect(self.log)
        worker.finished.connect(self._on_refine_finished)
        self.act_refine.setEnabled(False)
        self.act_recipe.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.status_label.setText("精密化中…")
        self.log(
            "精密化を開始: "
            + ", ".join(self.project.params.free_names()[:12])
            + (" …" if len(self.project.params.free()) > 12 else "")
        )
        self._thread.start()

    def stop_refine(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log("中止を要求しました(現在の反復が終わると止まります)")

    def _on_refine_finished(self, ok: bool, message: str) -> None:
        self.log(message)
        self.act_refine.setEnabled(True)
        self.act_recipe.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.param_model.refresh()
        self._refresh_plot()
        if ok and self.project.last_fit is not None:
            hc = self.project.last_fit.high_correlations(0.95)
            if hc:
                self.log(
                    "注意: 強く相関するパラメータがあります(片方を固定することを検討): "
                    + "; ".join(f"{a}–{b} ({c:+.2f})" for a, b, c in hc[:5])
                )

    def show_summary(self) -> None:
        if not (self.project.histograms and self.project.phases):
            return
        text = self.project.summary()
        self.log(text)
        box = QMessageBox(self)
        box.setWindowTitle("結果の要約")
        box.setText(text[:4000])
        box.exec()

    # ------------------------------------------------------------------
    # ウィザード
    # ------------------------------------------------------------------
    def run_wizard(self) -> None:
        from kaisetsu.gui.wizard import StartWizard

        wiz = StartWizard(self)
        if wiz.exec() == StartWizard.DialogCode.Accepted:
            self._refresh_all()

    # ------------------------------------------------------------------
    # ヘルプ
    # ------------------------------------------------------------------
    def show_naming_help(self) -> None:
        QMessageBox.information(
            self,
            "パラメータ名の規則",
            "相:cell:a|b|c|alpha|beta|gamma … 格子定数\n"
            "相:サイト:x|y|z … 分率座標(特殊位置の拘束は自動)\n"
            "相:サイト:occ … 占有率、相:サイト:uiso … 等方性変位パラメータ\n"
            "h0:inst:zero|u|v|w|p|x|y|z|s_l|h_l|pol … 装置(h0 は 1 番目のデータ)\n"
            "h0:sample:displacement|mu_eff|disp_x|disp_y|mu_r … 試料\n"
            "h0:bkg:0,1,2,… … 背景(Chebyshev 係数)\n"
            "h0:相:scale|md_r|size|strain … スケール、選択配向、サイズ、歪み\n\n"
            "絞り込み欄では * と ? が使えます(例: *:cell:*, h0:inst:[uvw])。",
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "kaisetsu について",
            f"kaisetsu {__version__}\n初心者向け・日本語対応の粉末回折 Rietveld 解析プログラム。\n\n"
            "計算の検証は GSAS-II 5.7.4 と突き合わせています(docs/validation/)。",
        )

    # ------------------------------------------------------------------
    def _confirm_discard(self) -> bool:
        if not self.project.histograms and not self.project.phases:
            return True
        r = QMessageBox.question(
            self, "確認", "現在のプロジェクトを閉じますか?(保存していない変更は失われます)"
        )
        return r == QMessageBox.StandardButton.Yes

    def _last_dir(self) -> str:
        return str(self.settings.value("last_dir", str(Path.home())))

    def _remember_dir(self, path: str) -> None:
        self.settings.setValue("last_dir", str(Path(path).parent))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            self._thread.wait(5000)
        event.accept()


__all__ = ["MainWindow"]
