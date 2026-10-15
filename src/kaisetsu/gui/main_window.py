"""メインウィンドウ。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
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
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableView,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu import __version__
from kaisetsu.crystal.cif import read_structure, write_structure
from kaisetsu.gui.background_dialog import BackgroundDialog
from kaisetsu.gui.dialogs import InstrumentDialog, PhaseNameDialog, RecipeDialog
from kaisetsu.gui.help_viewer import HelpWindow
from kaisetsu.gui.i18n import tr
from kaisetsu.gui.param_model import COL_DESC, COL_NAME, ParameterFilter, ParameterModel
from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.gui.plot_widget import HistoryPlot, PatternPlot, PDFPlot
from kaisetsu.gui.structure_view import StructureView
from kaisetsu.gui.update_dialog import SETTING_CHECK_ON_START, UpdateChecker, check_on_start
from kaisetsu.gui.worker import RefineThread, RefineWorker
from kaisetsu.io.project import load_project, save_project
from kaisetsu.io.readers import read_powder_data
from kaisetsu.pattern.instrument import CWInstrument, SampleGeometry
from kaisetsu.pattern.tof import TOFInstrument
from kaisetsu.pdf.data import PDFData, read_pdf_data
from kaisetsu.refine.project import HistogramEntry, Project
from kaisetsu.refine.recipe import Recipe
from kaisetsu.userdocs import GUIDE_URL, user_guide_dir

log = logging.getLogger(__name__)

DATA_FILTER = (
    "回折データ (*.xy *.xye *.chi *.dat *.txt *.csv *.asc *.ras *.rasx *.xrdml "
    "*.gsa *.gsas *.fxye *.raw *.xra *.cwn *.cif);;すべて (*)"
)
PDF_FILTER = "PDF G(r) (*.gr *.dat *.txt *.csv);;すべて (*)"
CIF_FILTER = "CIF (*.cif);;すべて (*)"
PROJECT_FILTER = "kaisetsu プロジェクト (*.kaisetsu.json *.json);;すべて (*)"
IMAGE_FILTER = "PNG 画像 (*.png);;SVG 画像 (*.svg)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"kaisetsu {__version__} — 粉末回折 Rietveld / PDF / MEM 解析")
        # ノート PC(1366×768、拡大率 150% なら論理 911×512)でも画面からはみ出さない大きさにする
        avail = self.screen().availableGeometry()
        self.resize(min(1400, avail.width() - 40), min(850, avail.height() - 60))
        self.project = Project("新規プロジェクト")
        self.project_path: Path | None = None
        self.current_hist = 0
        self.current_pdf = 0
        self.current_view = "hist"  # "hist" か "pdf"
        self.current_phase: str | None = None
        self._thread: RefineThread | None = None
        self._worker: RefineWorker | None = None
        # 直前の精密化(または初期値に戻す操作)の前の値。Ctrl+Z で戻す
        self._undo_values: dict[str, float] | None = None
        self.settings = QSettings("kaisetsu", "kaisetsu")
        self._help_window: HelpWindow | None = None

        self._build_ui()
        self._build_menu()
        self._refresh_all()

        # 新しい版の確認(ヘルプ → 更新を確認、起動時は 1 日 1 回)
        self.updater = UpdateChecker(
            self, self.settings, log=self.log, confirm_quit=self._confirm_discard
        )
        if self.updater.should_auto_check():
            QTimer.singleShot(1500, lambda: self.updater.check(manual=False))

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["プロジェクト"])
        self.tree.itemClicked.connect(self._on_tree_clicked)
        self.tree.setMinimumWidth(220)

        self.plot = PatternPlot()
        self.plot.excludeRequested.connect(self._on_exclude_requested)
        self.plot.excludedChanged.connect(self._on_excluded_changed)
        self.pdf_plot = PDFPlot()
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.plot)
        self.plot_stack.addWidget(self.pdf_plot)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        center = QSplitter(Qt.Orientation.Vertical)
        center.addWidget(self.plot_stack)
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
        dock = QDockWidget(tr("パラメータ"), self)
        dock.setObjectName("paramDock")
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        filt = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("絞り込み(例: *cell*, h0:inst:*, Pb)")
        self.only_free = QCheckBox(tr("精密化中のみ"))
        self.show_advanced = QCheckBox(tr("詳細も表示"))
        self.show_advanced.setToolTip(
            "普段は使わない量(装置の P と Z、表面粗さ b など)も表に出します。"
            "動かしている量は常に表示されます"
        )
        filt.addWidget(self.filter_edit, 1)
        filt.addWidget(self.only_free)
        filt.addWidget(self.show_advanced)
        v.addLayout(filt)
        quick = QHBoxLayout()
        # (表示名, glob, 種類)。種類で相の量とヒストグラムの量を区別する
        # (例: 「U」で装置の U まで出ないように)
        for label, pat, scope in (
            ("全て", "*", "all"),
            ("格子", "*:cell:*", "phase"),
            ("原子", "*:*:[xyz]", "phase"),
            ("U", "*:*:u*", "phase"),
            ("装置", "h*:inst:*", "hist"),
            ("試料", "h*:sample:*", "hist"),
            ("背景", "h*:bkg:*", "hist"),
            ("スケール", "*:*:scale", "all"),
            ("PDF", "g*:*", "hist"),
        ):
            b = QPushButton(tr(label))
            b.setFlat(True)
            b.clicked.connect(lambda _=False, p=pat, s=scope: self._quick_filter(p, s))
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
        hint = QLabel(
            "「精密化」列にチェックを付けた量が、精密化(F9)で動きます。"
            "赤字 = 動かす、灰色 = 対称性で固定または他の量に拘束。"
            "値はダブルクリックで書き換えられます(F5 で再計算)。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444;")
        v.addWidget(hint)
        v.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.btn_vary_shown = QPushButton(tr("表示中の項目に全部チェック"))
        self.btn_vary_shown.setToolTip(
            "いま表に表示されている(絞り込み後の)パラメータの「精密化」をまとめてオンにします"
        )
        self.btn_fix_shown = QPushButton(tr("表示中のチェックを全部外す"))
        self.btn_fix_shown.setToolTip(
            "いま表に表示されているパラメータの「精密化」をまとめてオフにします"
        )
        self.btn_reset_shown = QPushButton(tr("表示中の値を初期値に戻す"))
        self.btn_reset_shown.setToolTip(
            "いま表に表示されているパラメータの値を、読み込んだとき(または設定し直したとき)の"
            "値に戻します。全部戻すには「精密化」→「すべて初期値に戻す」。Ctrl+Z で戻す前に戻せます"
        )
        self.btn_vary_shown.clicked.connect(lambda: self._set_vary_shown(True))
        self.btn_fix_shown.clicked.connect(lambda: self._set_vary_shown(False))
        self.btn_reset_shown.clicked.connect(self._reset_shown)
        btns.addWidget(self.btn_vary_shown)
        btns.addWidget(self.btn_fix_shown)
        btns.addWidget(self.btn_reset_shown)
        v.addLayout(btns)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setMinimumWidth(400)
        self.param_dock = dock
        self.filter_edit.textChanged.connect(self._on_filter_text)
        self.only_free.toggled.connect(self.param_filter.set_only_free)
        self.show_advanced.toggled.connect(self.param_filter.set_show_advanced)

        # 構造の 3D 表示(ドック)
        self.structure_dock = QDockWidget(tr("構造"), self)
        self.structure_dock.setObjectName("structureDock")
        self.structure_view = StructureView()
        self.structure_dock.setWidget(self.structure_view)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.structure_dock)
        # χ² の推移(ドック)
        self.history_dock = QDockWidget(tr("χ² の推移"), self)
        self.history_dock.setObjectName("historyDock")
        self.history_plot = HistoryPlot()
        self.history_dock.setWidget(self.history_plot)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.history_dock)
        self.history_dock.hide()
        # 幅が狭い画面では構造の 3D 表示を隠して図の幅を確保する(「表示」メニューから出せる)
        if self.screen().availableGeometry().width() < 1500:
            self.structure_dock.hide()

        self.status_label = QLabel(tr("データと相を読み込んでください"))
        self.statusBar().addWidget(self.status_label, 1)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #b02020; font-weight: bold;")
        self.statusBar().addPermanentWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 動いていることを示すビジー表示
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu(tr("ファイル(&F)"))
        self._add_action(
            m_file, tr("新規プロジェクト(&N)"), self.new_project, QKeySequence.StandardKey.New
        )
        self.act_open = self._add_action(
            m_file, tr("プロジェクトを開く(&O)…"), self.open_project, QKeySequence.StandardKey.Open
        )
        self.act_save = self._add_action(
            m_file, tr("プロジェクトを保存(&S)"), self.save_project, QKeySequence.StandardKey.Save
        )
        self._add_action(
            m_file,
            tr("名前を付けて保存(&A)…"),
            self.save_project_as,
            QKeySequence.StandardKey.SaveAs,
        )
        m_file.addSeparator()
        self._add_action(m_file, tr("計算パターンを CSV に書き出す…"), self.export_pattern_csv)
        self._add_action(m_file, tr("反射リストを CSV に書き出す…"), self.export_reflections_csv)
        self._add_action(m_file, tr("相を CIF に書き出す(VESTA で開けます)…"), self.export_cif)
        self._add_action(m_file, tr("図を画像で保存 (PNG/SVG)…"), self.export_plot_image)
        m_file.addSeparator()
        self._add_action(m_file, tr("終了(&X)"), self.close, QKeySequence.StandardKey.Quit)

        m_data = mb.addMenu(tr("データ(&D)"))
        self.act_load = self._add_action(
            m_data, tr("回折データを読み込む(&L)…"), self.load_data, "Ctrl+L"
        )
        self.act_add_phase = self._add_action(
            m_data, tr("CIF から相を追加(&P)…"), self.add_phase_from_cif, "Ctrl+P"
        )
        self.act_instrument = self._add_action(
            m_data, tr("装置と試料の設定(&I)…"), self.edit_instrument, "Ctrl+I"
        )
        self.act_background = self._add_action(
            m_data, tr("背景の設定(&B)…"), self.edit_background, "Ctrl+B"
        )
        self.act_range = self._add_action(
            m_data, tr("データ範囲と除外領域(&E)…"), self.edit_range, "Ctrl+E"
        )
        self.act_phase = self._add_action(
            m_data, tr("相の設定(強度の扱い・選択配向軸・異方性)…"), self.edit_phase
        )
        self.act_lattice = self._add_action(
            m_data, tr("Le Bail で格子定数だけ(放射光マイクロ XRD 向け)…"), self.open_lattice_dialog
        )
        self.act_lattice.setToolTip(
            "データと CIF を入れるだけで、強度を使わない Le Bail 法で格子定数と esd を出します。"
            "複数ファイルの一括処理と CSV 保存もできます"
        )
        m_data.addSeparator()
        self.act_load_pdf = self._add_action(
            m_data, tr("PDF データ G(r) を読み込む…"), self.load_pdf_data
        )
        self.act_transform = self._add_action(
            m_data, tr("I(Q) から G(r) を作る…"), self.transform_iq
        )
        m_data.addSeparator()
        self.act_wizard = self._add_action(
            m_data, tr("はじめての精密化(ウィザード)(&W)…"), self.run_wizard
        )
        self.act_sequence = self._add_action(
            m_data, tr("連続データを順に精密化…"), self.run_sequence
        )

        m_ref = mb.addMenu(tr("精密化(&R)"))
        self.act_calc = self._add_action(m_ref, tr("パターンを計算(&C)"), self.recalculate, "F5")
        self.act_refine = self._add_action(m_ref, tr("精密化を実行(&R)"), self.refine, "F9")
        self.act_recipe = self._add_action(
            m_ref, tr("標準の手順で精密化(&S)…"), self.refine_recipe, "Shift+F9"
        )
        self.act_pdf_recipe = self._add_action(
            m_ref, tr("PDF の標準手順で精密化…"), self.refine_pdf_recipe
        )
        self.act_stop = self._add_action(m_ref, tr("中止(&T)"), self.stop_refine, "Esc")
        self.act_stop.setEnabled(False)
        m_ref.addSeparator()
        self._add_action(m_ref, tr("すべて固定(&F)"), self.fix_all)
        self.act_undo = self._add_action(
            m_ref, tr("直前の精密化の前に戻す(&Z)"), self.undo_refine, "Ctrl+Z"
        )
        self.act_undo.setEnabled(False)
        self.act_undo.setToolTip(
            "直前の精密化(または「初期値に戻す」)の前の値に全パラメータを戻します。"
            "もう一度選ぶと戻す前の値に戻ります"
        )
        self.act_reset = self._add_action(
            m_ref, tr("すべて初期値に戻す(&I)"), self.reset_to_initial
        )
        self.act_reset.setToolTip(
            "データ・CIF を読み込んだとき(装置や背景を設定し直したときはその値)に全パラメータを"
            "戻します。精密化の結果は失われますが、Ctrl+Z で戻す前の値に戻せます"
        )
        self._add_action(m_ref, tr("結果の要約を表示(&U)"), self.show_summary)
        self.act_residual = self._add_action(
            m_ref, tr("残差の点検(未同定ピーク・こぶ)…"), self.open_residual_dialog
        )
        self.act_residual.setToolTip(
            "差のプロットに残った、相の反射で説明できないピークと広いこぶの候補を示します。"
            "選んだものを除外領域や背景のハンプにできます(自動では除外しません)"
        )
        self.act_width_penalty = self._add_action(
            m_ref, tr("ピーク幅が負にならないよう拘束する(罰則)"), self._toggle_width_penalty
        )
        self.act_width_penalty.setCheckable(True)
        self.act_width_penalty.setToolTip(
            "U, V, W(X, Y, Z、TOF の σ²)の組み合わせがデータ範囲で負にならないように精密化します。\n"
            "既定では GSAS-II と同じく負の範囲はその成分を 0 として計算し、結果の要約で知らせます。\n"
            "拘束すると Rwp が悪くなることがあります(装置の特性として意味のある U, V, W が要るとき用)"
        )
        m_ref.addSeparator()
        self.act_mem = self._add_action(m_ref, tr("MEM 電子密度 / 核密度(&M)…"), self.run_mem)

        m_view = mb.addMenu(tr("表示(&V)"))
        m_view.addAction(self.param_dock.toggleViewAction())
        m_view.addAction(self.structure_dock.toggleViewAction())
        m_view.addAction(self.history_dock.toggleViewAction())

        m_help = mb.addMenu(tr("ヘルプ(&H)"))
        self.act_guide = self._add_action(
            m_help, tr("利用者ガイド(&G)"), lambda: self.show_user_guide(), "F1"
        )
        self._add_action(
            m_help, tr("インストールと更新の説明"), lambda: self.show_user_guide("install.md")
        )
        self._add_action(m_help, tr("オンラインの説明書(GitHub)…"), self.open_online_guide)
        m_help.addSeparator()
        self._add_action(m_help, tr("パラメータ名の規則"), self.show_naming_help)
        m_help.addSeparator()
        self._add_action(m_help, tr("更新を確認(&U)…"), self.check_for_updates)
        self.act_auto_update = QAction(tr("起動時に更新を確認する"), self)
        self.act_auto_update.setCheckable(True)
        self.act_auto_update.setChecked(check_on_start(self.settings))
        self.act_auto_update.toggled.connect(
            lambda on: self.settings.setValue(SETTING_CHECK_ON_START, bool(on))
        )
        m_help.addAction(self.act_auto_update)
        m_help.addSeparator()
        self._add_action(m_help, tr("バージョン情報(&A)"), self.show_about)

        self._build_toolbar()

    def _build_toolbar(self) -> None:
        tb = QToolBar("メイン")
        tb.setObjectName("mainToolbar")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(tb)
        style = self.style()
        sp = QStyle.StandardPixmap

        def add(act: QAction, icon: QStyle.StandardPixmap, text: str) -> None:
            act.setIcon(style.standardIcon(icon))
            act.setIconText(text)
            tb.addAction(act)

        add(self.act_wizard, sp.SP_DialogHelpButton, tr("ウィザード"))
        add(self.act_load, sp.SP_DialogOpenButton, tr("データ読込"))
        add(self.act_add_phase, sp.SP_FileIcon, tr("CIF 追加"))
        add(self.act_instrument, sp.SP_FileDialogDetailedView, tr("装置設定"))
        add(self.act_background, sp.SP_FileDialogListView, tr("背景設定"))
        add(self.act_phase, sp.SP_FileDialogInfoView, tr("相の設定"))
        tb.addSeparator()
        add(self.act_calc, sp.SP_BrowserReload, tr("計算 (F5)"))
        add(self.act_refine, sp.SP_MediaPlay, tr("精密化 (F9)"))
        add(self.act_recipe, sp.SP_MediaSeekForward, tr("標準の手順で精密化"))
        add(self.act_stop, sp.SP_MediaStop, tr("中止"))
        tb.addSeparator()
        add(self.act_load_pdf, sp.SP_DirOpenIcon, tr("PDF 読込"))
        add(self.act_mem, sp.SP_ComputerIcon, tr("MEM"))
        tb.addSeparator()
        add(self.act_save, sp.SP_DialogSaveButton, tr("保存"))

    def _add_action(
        self,
        menu: QMenu,
        text: str,
        slot: Callable[[], Any],
        shortcut: str | QKeySequence.StandardKey | None = None,
    ) -> QAction:
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
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
        root = QTreeWidgetItem([self.project.name or tr("プロジェクト")])
        root.setData(0, Qt.ItemDataRole.UserRole, ("project", None))
        self.tree.addTopLevelItem(root)
        phases = QTreeWidgetItem([tr("相")])
        root.addChild(phases)
        for ph in self.project.phases.values():
            st = ph.structure
            mode = "、Pawley" if ph.pawley else ("、Le Bail" if ph.le_bail else "")
            item = QTreeWidgetItem([f"{ph.name}  ({st.spacegroup.hm}, {st.n_sites} サイト{mode})"])
            item.setData(0, Qt.ItemDataRole.UserRole, ("phase", ph.name))
            item.setToolTip(0, repr(st.lattice))
            phases.addChild(item)
        hists = QTreeWidgetItem([tr("ヒストグラム")])
        root.addChild(hists)
        for i, h in enumerate(self.project.histograms):
            if isinstance(h.instrument, TOFInstrument):
                label = (
                    f"h{i}: {h.name}  (TOF 中性子 DIFC={h.instrument.difc:.0f}, {len(h.data)} 点)"
                )
            else:
                rad = h.instrument.radiation
                kind = "中性子" if rad.is_neutron else "X 線"
                label = f"h{i}: {h.name}  ({kind} λ={rad.wavelength:.4f} Å, {len(h.data)} 点)"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, ("hist", i))
            hists.addChild(item)
        if self.project.pdf_datasets:
            pdfs = QTreeWidgetItem([tr("PDF データ")])
            root.addChild(pdfs)
            for i, g in enumerate(self.project.pdf_datasets):
                kind = "中性子" if g.radiation == "neutron" else "X 線"
                item = QTreeWidgetItem(
                    [
                        f"g{i}: {g.name}  ({kind}, Q_max={g.qmax:g}, r={g.r_min:.1f}〜{g.r_max:.1f} Å)"
                    ]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, ("pdf", i))
                pdfs.addChild(item)
        self.tree.expandAll()

    def _refresh_params(self) -> None:
        self.param_filter.set_phase_names(self.project.phases.keys())
        self.param_model.set_params(self.project.params)

    def _refresh_plot(self) -> None:
        if self.current_view == "pdf" and self.project.pdf_datasets:
            self.plot_stack.setCurrentWidget(self.pdf_plot)
            i = min(self.current_pdf, len(self.project.pdf_datasets) - 1)
            g = self.project.pdf_datasets[i]
            self.pdf_plot.set_data(g.data, (g.r_min, g.r_max))
            if self.project.phases:
                try:
                    res = self.project.calculate_pdf(i)
                except Exception as exc:
                    self.log(f"PDF の計算に失敗しました: {exc}")
                    return
                self.pdf_plot.set_result(res, g.mask())
                rf = self.project.rfactors()
                r = rf[f"g{i}"]
                self.status_label.setText(
                    f"g{i} {g.name}: Rw = {100 * r.rwp:.3f}%  χ² = {r.chi2:.3f}"
                )
            return
        self.plot_stack.setCurrentWidget(self.plot)
        if not self.project.histograms:
            self.plot.set_data(None)
            return
        i = min(self.current_hist, len(self.project.histograms) - 1)
        h = self.project.histograms[i]
        if isinstance(h.instrument, TOFInstrument):
            self.plot.set_tof(h.instrument.difc)
            self.plot.set_data(h.data, None)
        else:
            self.plot.set_tof(None)
            self.plot.set_data(h.data, h.instrument.radiation.wavelength)
        self.plot.set_excluded(h.excluded)
        self.plot.set_fit_range(h.x_min, h.x_max)
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

    def _refresh_structure(self) -> None:
        name = self.current_phase
        if name not in self.project.phases:
            name = next(iter(self.project.phases), None)
        self.current_phase = name
        if name is None:
            self.structure_view.set_structure(None)
            return
        ph = self.project.phases[name]
        ph.sync()
        self.structure_view.set_structure(ph.structure)
        self.structure_dock.setWindowTitle(f"構造: {name}")

    def _refresh_all(self) -> None:
        self.act_width_penalty.setChecked(bool(self.project.penalize_negative_widths))
        self._refresh_tree()
        self._refresh_params()
        self._refresh_plot()
        self._refresh_structure()
        has_model = bool(
            self.project.phases and (self.project.histograms or self.project.pdf_datasets)
        )
        self.act_refine.setEnabled(has_model)
        self.act_recipe.setEnabled(bool(self.project.phases and self.project.histograms))
        self.act_pdf_recipe.setEnabled(bool(self.project.phases and self.project.pdf_datasets))
        self.act_mem.setEnabled(bool(self.project.phases and self.project.histograms))

    def _on_tree_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, ref = data
        if kind == "hist":
            self.current_hist = int(ref)
            self.current_view = "hist"
            self._refresh_plot()
            self.filter_edit.setText(f"h{ref}:*")
        elif kind == "pdf":
            self.current_pdf = int(ref)
            self.current_view = "pdf"
            self._refresh_plot()
            self.filter_edit.setText(f"g{ref}:*")
        elif kind == "phase":
            self.current_phase = str(ref)
            self._refresh_structure()
            self.filter_edit.setText(f"{ref}:*")

    def _on_filter_text(self, text: str) -> None:
        """絞り込み欄を手で書き換えたら種類フィルタは外す(クイックボタンは後で付け直す)。"""
        self.param_filter.set_scope("all")
        self.param_filter.set_pattern(text)

    def _quick_filter(self, pattern: str, scope: str) -> None:
        self.filter_edit.setText(pattern)
        self.param_filter.set_scope(scope)

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
        self.current_pdf = 0
        self.current_view = "hist"
        self.current_phase = None
        self.log_view.clear()
        self.history_plot.clear()
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
        self.current_pdf = 0
        self.current_view = "hist" if self.project.histograms else "pdf"
        self.current_phase = None
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
        if self.current_view == "pdf" and self.project.pdf_datasets:
            self._export_pdf_csv()
            return
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

    def _export_pdf_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "計算 G(r) を書き出す", self._last_dir(), "CSV (*.csv)"
        )
        if not path:
            return
        i = self.current_pdf
        g = self.project.pdf_datasets[i]
        res = self.project.calculate_pdf(i)
        cols = [res.r, g.data.g, res.g_calc, g.data.g - res.g_calc]
        header = "r,G_obs,G_calc,diff"
        for name, gp in zip(g.phases, res.g_phases, strict=True):
            cols.append(gp)
            header += f",{name}"
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="")
        self.log(f"計算 G(r) を書き出しました: {path}")

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
        name = self.current_phase or next(iter(self.project.phases))
        path, _ = QFileDialog.getSaveFileName(self, "CIF に書き出す", self._last_dir(), CIF_FILTER)
        if not path:
            return
        ph = self.project.phases[name]
        ph.sync()
        write_structure(ph.structure, path)
        self.log(f"CIF を書き出しました: {path}")

    def export_plot_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "図を画像で保存", self._last_dir(), IMAGE_FILTER
        )
        if not path:
            return
        if not path.lower().endswith((".png", ".svg")):
            path += ".png"
        widget = self.pdf_plot if self.plot_stack.currentWidget() is self.pdf_plot else self.plot
        try:
            widget.export_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "画像の保存", f"保存できませんでした:\n{exc}")
            return
        self._remember_dir(path)
        self.log(f"図を保存しました: {path}")

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
        instrument: CWInstrument | TOFInstrument | None = None,
        sample: SampleGeometry | None = None,
        n_background: int = 8,
        bank: int | None = None,
        all_banks: bool = False,
    ) -> None:
        """回折データを読み込んでヒストグラムを追加する。

        複数 BANK の GSAS ファイルは、bank で番号を指定するか、all_banks=True でバンクごとに
        ヒストグラムを追加する(装置・試料は複製する。TOF ならバンクごとに装置ダイアログを出す)。
        どちらも無く instrument も無いときは選択ダイアログを出す。
        """
        if all_banks or (bank is None and instrument is None):
            from kaisetsu.io.readers import bank_count

            n_banks = bank_count(path)
            if n_banks > 1:
                if not all_banks:
                    from PySide6.QtWidgets import QInputDialog

                    items = [f"バンク {k}" for k in range(1, n_banks + 1)] + [
                        "すべてのバンク(バンクごとにヒストグラムを追加)"
                    ]
                    item, ok = QInputDialog.getItem(
                        self,
                        "バンクの選択",
                        f"このファイルには {n_banks} 個の BANK があります:",
                        items,
                        0,
                        False,
                    )
                    if not ok:
                        return
                    idx = items.index(item)
                    if idx < n_banks:
                        bank = idx + 1
                    else:
                        all_banks = True
                if all_banks:
                    from dataclasses import replace

                    for k in range(1, n_banks + 1):
                        self.load_data_file(
                            path,
                            replace(instrument) if instrument is not None else None,
                            replace(sample) if sample is not None else None,
                            n_background,
                            bank=k,
                        )
                    return
        kwargs: dict[str, object] = {}
        if bank is not None:
            kwargs["bank"] = bank
        try:
            data = read_powder_data(path, **kwargs)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みエラー", f"データを読み込めませんでした:\n{exc}")
            return
        self._remember_dir(path)
        if instrument is None and data.metadata.get("xunit") == "tof_us":
            from kaisetsu.gui.tof_dialog import TOFInstrumentDialog

            guess = _guess_tof_instrument(data.metadata)
            tdlg = TOFInstrumentDialog(guess, self, sample)
            if bank is not None:
                tdlg.setWindowTitle(f"TOF 装置パラメータ(バンク {bank})")
            if tdlg.exec() != TOFInstrumentDialog.DialogCode.Accepted:
                return
            instrument = tdlg.result_instrument()
            sample = tdlg.result_sample()
        if instrument is None:
            dlg = InstrumentDialog(None, None, self)
            dlg.setWindowTitle("このデータの装置と試料の設定")
            dlg.apply_hints(data.metadata)
            if data.metadata.get("target"):
                self.log(
                    f"ファイルから読み取り: ターゲット {data.metadata.get('target')}, "
                    f"λ1 = {data.metadata.get('wavelength')} Å"
                )
            if dlg.exec() != InstrumentDialog.DialogCode.Accepted:
                return
            instrument, sample = dlg.result_instrument()
        h = self.project.add_histogram(data, instrument, sample, n_background=n_background)
        self.current_hist = len(self.project.histograms) - 1
        self.current_view = "hist"
        for pc in h.contributions.values():
            pc.scale = self._initial_scale(h)
        self.project.build_parameters()
        unit = " μs" if data.metadata.get("xunit") == "tof_us" else "°"
        where = f" バンク {bank}" if bank is not None else ""
        self.log(
            f"データを読み込みました: {path}{where} "
            f"({len(data)} 点, {data.x[0]:.3f}–{data.x[-1]:.3f}{unit})"
        )
        self._refresh_all()

    def load_pdf_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "PDF データ G(r) を読み込む", self._last_dir(), PDF_FILTER
        )
        if not path:
            return
        self.load_pdf_file(path)

    def load_pdf_file(self, path: str, options: dict[str, object] | None = None) -> None:
        try:
            data = read_pdf_data(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "読み込みエラー", f"PDF データを読み込めませんでした:\n{exc}"
            )
            return
        self._remember_dir(path)
        self.add_pdf_dataset(data, options)
        self.log(
            f"PDF データを読み込みました: {path} ({len(data)} 点, r = {data.r[0]:.2f}〜{data.r[-1]:.2f} Å)"
        )

    def add_pdf_dataset(self, data: PDFData, options: dict[str, object] | None = None) -> None:
        if options is None:
            from kaisetsu.gui.pdf_dialogs import PDFDataDialog

            dlg = PDFDataDialog(data, self)
            if dlg.exec() != PDFDataDialog.DialogCode.Accepted:
                return
            options = dlg.values()
        self.project.add_pdf(
            data,
            str(options.get("radiation", "xray")),
            qmax=float(options.get("qmax", 0.0)),
            qdamp=float(options.get("qdamp", 0.0)),
            qbroad=float(options.get("qbroad", 0.0)),
            r_min=options.get("r_min"),
            r_max=options.get("r_max"),
        )
        self.current_pdf = len(self.project.pdf_datasets) - 1
        self.current_view = "pdf"
        self._refresh_all()

    def transform_iq(self) -> None:
        from kaisetsu.gui.pdf_dialogs import TransformDialog

        comp = ""
        if self.project.phases:
            st = next(iter(self.project.phases.values())).structure
            comp = "".join(f"{el}{n:g}" if n != 1 else el for el, n in st.composition().items())
        dlg = TransformDialog(self, comp)
        if dlg.exec() != TransformDialog.DialogCode.Accepted:
            return
        data = dlg.pdf_data()
        if data is None:
            return
        for line in dlg.result.log if dlg.result else []:
            self.log("  " + line)
        settings = dlg.settings()
        self.add_pdf_dataset(
            data,
            {
                "radiation": settings.radiation,
                "qmax": settings.qmax,
                "qdamp": 0.04,
                "qbroad": 0.0,
                "r_min": 1.0,
                "r_max": float(data.r[-1]),
            },
        )
        self.log(f"I(Q) から G(r) を作ってプロジェクトに追加しました: {data.name}")

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
        self.current_phase = name
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

    def edit_background(self) -> None:
        if not self.project.histograms:
            QMessageBox.information(self, "背景", "先に回折データを読み込んでください")
            return
        h = self.project.histograms[self.current_hist]
        dlg = BackgroundDialog(
            h.background.order,
            getattr(h.background, "kind", "chebyshev"),
            len(h.humps),
            (float(h.data.x[0]), float(h.data.x[-1])),
            self,
        )
        if dlg.exec() != BackgroundDialog.DialogCode.Accepted:
            return
        st = dlg.settings()
        kind = str(st["kind"])
        n_terms = int(st["n_terms"])
        estimate = bool(st["estimate"])
        self.project.reset_background(self.current_hist, n_terms, estimate, kind=kind)
        n_humps = int(st["n_humps"])
        if n_humps != len(h.humps):
            self.project.clear_humps(self.current_hist)
            center = float(st["hump_center"])
            fwhm = float(st["hump_fwhm"])
            span = float(h.data.x[-1] - h.data.x[0])
            for k in range(n_humps):
                c = center + (k - 0.5 * (n_humps - 1)) * min(fwhm, span / max(n_humps, 1))
                i = int(np.clip(np.searchsorted(h.data.x, c), 0, len(h.data.x) - 1))
                height = max(float(h.data.y[i] - h.background(h.data.x[i])), 0.0) * 0.5
                self.project.add_hump(self.current_hist, c, fwhm, height)
        label = "固定点補間" if kind == "points" else "Chebyshev"
        self.log(
            f"背景を {label} {n_terms} 項にしました"
            + ("(観測から推定し直し)" if estimate else "(係数は引き継ぎ)")
            + (f"、ハンプ {n_humps} 個" if n_humps else "")
        )
        self._refresh_all()

    def edit_instrument(self) -> None:
        if not self.project.histograms:
            QMessageBox.information(self, "装置", "先に回折データを読み込んでください")
            return
        h = self.project.histograms[self.current_hist]
        if isinstance(h.instrument, TOFInstrument):
            from kaisetsu.gui.tof_dialog import TOFInstrumentDialog

            tdlg = TOFInstrumentDialog(h.instrument, self, h.sample)
            if tdlg.exec() != TOFInstrumentDialog.DialogCode.Accepted:
                return
            h.instrument = tdlg.result_instrument()
            h.sample = tdlg.result_sample()
            self.project.build_parameters()
            self.log("TOF 装置パラメータを更新しました")
            self._refresh_all()
            return
        dlg = InstrumentDialog(h.instrument, h.sample, self)
        if dlg.exec() != InstrumentDialog.DialogCode.Accepted:
            return
        inst, smp = dlg.result_instrument()
        h.instrument = inst
        h.sample = smp
        self.project.build_parameters()
        self.log("装置パラメータを更新しました")
        self._refresh_all()

    def edit_range(self) -> None:
        """データ範囲と除外領域を数値で編集する。"""
        if not self.project.histograms:
            QMessageBox.information(self, "データ範囲", "先に回折データを読み込んでください")
            return
        from kaisetsu.gui.range_dialog import RangeDialog

        i = min(self.current_hist, len(self.project.histograms) - 1)
        h = self.project.histograms[i]
        dlg = RangeDialog(h, self)
        if dlg.exec() != RangeDialog.DialogCode.Accepted:
            return
        x_min, x_max, ranges = dlg.values()
        if not self._range_keeps_points(h, x_min, x_max, ranges):
            return
        self.project.set_range(i, x_min, x_max)
        h.set_excluded(ranges)
        self.log(
            f"h{i}: 範囲 {x_min:g}〜{x_max:g}、除外領域 "
            + ("、".join(f"{a:g}〜{b:g}" for a, b in h.excluded) or "なし")
        )
        self._refresh_all()

    def _toggle_width_penalty(self, _checked: bool = False) -> None:
        flag = self.act_width_penalty.isChecked()
        self.project.penalize_negative_widths = flag
        self.log(
            "ピーク幅が負にならない拘束を"
            + ("有効" if flag else "無効")
            + "にしました"
            + ("(次の精密化から)" if flag else "")
        )

    def _range_keeps_points(
        self,
        h: HistogramEntry,
        x_min: float,
        x_max: float,
        ranges: list[tuple[float, float]],
        minimum: int = 20,
    ) -> bool:
        """範囲・除外領域を変えた後もフィットに使える点が残るか。残らなければ知らせて False。"""
        from kaisetsu.refine.project import merge_ranges

        x = h.data.x
        assert h.data.weight is not None
        m = (x >= x_min) & (x <= x_max) & (h.data.weight > 0)
        for lo, hi in merge_ranges(ranges):
            m &= ~((x >= lo) & (x <= hi))
        n = int(np.count_nonzero(m))
        if n >= minimum:
            return True
        QMessageBox.warning(
            self,
            "データ範囲",
            f"この設定ではフィットに使える点が {n} 点しか残りません(最低 {minimum} 点)。"
            "範囲か除外領域を見直してください。",
        )
        return False

    def _current_histogram(self) -> tuple[int, HistogramEntry] | None:
        if not self.project.histograms or self.current_view != "hist":
            return None
        i = min(self.current_hist, len(self.project.histograms) - 1)
        return i, self.project.histograms[i]

    def _on_exclude_requested(self, lo: float, hi: float) -> None:
        """図の上で Shift + ドラッグされた範囲を除外領域に加える。"""
        cur = self._current_histogram()
        if cur is None:
            return
        i, h = cur
        if not self._range_keeps_points(h, h.x_min, h.x_max, [*h.excluded, (lo, hi)]):
            return
        h.set_excluded([*h.excluded, (lo, hi)])
        self.log(f"h{i}: {lo:.3f}〜{hi:.3f} を除外しました(帯の右クリックで解除)")
        self._refresh_plot()

    def _on_excluded_changed(self, ranges: list) -> None:  # type: ignore[type-arg]
        """帯の移動・削除を反映する。"""
        cur = self._current_histogram()
        if cur is None:
            return
        _, h = cur
        h.set_excluded([(float(a), float(b)) for a, b in ranges])
        self._refresh_plot()

    def edit_phase(self) -> None:
        if not self.project.phases:
            QMessageBox.information(self, "相の設定", "先に CIF から相を追加してください")
            return
        from kaisetsu.gui.phase_dialog import PhaseDialog

        name = self.current_phase or next(iter(self.project.phases))
        dlg = PhaseDialog(
            self.project,
            name,
            min(self.current_hist, max(len(self.project.histograms) - 1, 0)),
            self,
        )
        if dlg.exec() != PhaseDialog.DialogCode.Accepted:
            return
        for line in dlg.apply():
            self.log(line)
        self._refresh_all()

    # ------------------------------------------------------------------
    # 精密化
    # ------------------------------------------------------------------
    def recalculate(self) -> None:
        self._refresh_plot()
        if self.project.phases and (self.project.histograms or self.project.pdf_datasets):
            rf = self.project.rfactors()["total"]
            self.log(f"計算: {rf}")

    def fix_all(self) -> None:
        self.project.fix_all()
        self.param_model.refresh()

    def undo_refine(self) -> None:
        """直前の精密化(または初期値に戻す操作)の前の値に戻す。もう一度でやり直し。"""
        if self._undo_values is None:
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "戻す", "精密化の実行中は戻せません(Esc で中止)")
            return
        current = self.project.snapshot()
        names = self.project.restore(self._undo_values)
        self._undo_values = current
        self.log(
            f"直前の精密化の前の値に戻しました({len(names)} 個)。"
            "もう一度「直前の精密化の前に戻す」を選ぶと、戻す前の値に戻ります"
        )
        self._after_restore()

    def reset_to_initial(self, confirm: bool = True) -> None:
        """全パラメータを初期値(読み込んだとき・設定し直したときの値)に戻す。"""
        if not any(True for _ in self.project.params):
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "戻す", "精密化の実行中は戻せません(Esc で中止)")
            return
        if confirm:
            ans = QMessageBox.question(
                self,
                "初期値に戻す",
                "すべてのパラメータを初期値(データ・CIF を読み込んだときの値)に戻します。\n"
                "精密化の結果は失われます(Ctrl+Z で戻す前の値に戻せます)。よろしいですか?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        self._undo_values = self.project.snapshot()
        self.act_undo.setEnabled(True)
        names = self.project.reset_to_initial()
        self.log(
            f"すべてのパラメータを初期値に戻しました({len(names)} 個)。Ctrl+Z で戻す前の値に戻せます"
        )
        self._after_restore()

    def _reset_shown(self) -> None:
        """表に表示中(絞り込み後)のパラメータだけ初期値に戻す。"""
        if self._thread is not None and self._thread.isRunning():
            return
        names = []
        for r in range(self.param_filter.rowCount()):
            src = self.param_filter.mapToSource(self.param_filter.index(r, 0))
            names.append(self.param_model.parameter(src.row()).name)
        if not names:
            return
        self._undo_values = self.project.snapshot()
        self.act_undo.setEnabled(True)
        done = self.project.reset_to_initial(names)
        self.log(
            f"表示中の {len(done)} 個のパラメータを初期値に戻しました(Ctrl+Z で戻す前に戻せます)"
        )
        self._after_restore()

    def _after_restore(self) -> None:
        self.param_model.refresh()
        self._refresh_plot()
        self._refresh_structure()
        if self.project.phases and (self.project.histograms or self.project.pdf_datasets):
            rf = self.project.rfactors()["total"]
            self.log(f"計算: {rf}")

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

    def refine_pdf_recipe(self) -> None:
        from kaisetsu.gui.pdf_dialogs import PDFRecipeDialog

        dlg = PDFRecipeDialog(list(self.project.phases), self)
        if dlg.exec() != PDFRecipeDialog.DialogCode.Accepted:
            return
        self.current_view = "pdf"
        self._refresh_plot()
        self._start_worker(RefineWorker(self.project, dlg.recipe()))

    def run_recipe(
        self,
        recipe: Recipe,
        companion: Callable[[Callable[[str], None]], str] | None = None,
    ) -> None:
        """レシピを別スレッドで走らせる。companion はレシピの後に走らせる付随処理"""
        self._start_worker(RefineWorker(self.project, recipe, companion=companion))

    def _start_worker(self, worker: RefineWorker) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "精密化", "すでに精密化が実行中です")
            return
        self._worker = worker
        self._undo_values = self.project.snapshot()
        self._thread = RefineThread(worker, self)
        worker.progress.connect(self.log)
        worker.iteration.connect(self._on_iteration)
        worker.step_started.connect(self.history_plot.mark_step)
        worker.finished.connect(self._on_refine_finished)
        self.act_refine.setEnabled(False)
        self.act_recipe.setEnabled(False)
        self.act_pdf_recipe.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.progress_label.setText("精密化中…")
        self.progress_bar.show()
        self.history_plot.clear()
        self.history_dock.show()
        n_free = len(self.project.params.free())
        if worker.recipe is None:
            self.log(
                f"精密化を開始({n_free} 変数): "
                + ", ".join(self.project.params.free_names()[:12])
                + (" …" if n_free > 12 else "")
            )
        else:
            self.log(f"手順 '{worker.recipe.name}' で精密化を開始({len(worker.recipe.steps)} 段階)")
        self._thread.start()

    def _on_iteration(self, iteration: int, chi2: float, curves: object) -> None:
        self.progress_label.setText(f"精密化中: 反復 {iteration}  χ² = {chi2:.3f}")
        self.history_plot.append(chi2)
        try:
            if not isinstance(curves, dict):
                return
            if self.current_view == "pdf":
                i = min(self.current_pdf, len(self.project.pdf_datasets) - 1)
                pdfs = curves.get("pdf", [])
                if 0 <= i < len(pdfs):
                    r, gc = pdfs[i]
                    self.pdf_plot.update_curves(r, gc)
            else:
                i = min(self.current_hist, len(self.project.histograms) - 1)
                hists = curves.get("hist", [])
                if 0 <= i < len(hists):
                    x, yc, yb = hists[i]
                    self.plot.update_curves(x, yc, yb)
        except Exception:
            pass

    def stop_refine(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log("中止を要求しました(現在の反復が終わると止まります)")

    def _on_refine_finished(self, ok: bool, message: str) -> None:
        self.log(message)
        self.act_undo.setEnabled(self._undo_values is not None)
        self.act_refine.setEnabled(True)
        self.act_recipe.setEnabled(bool(self.project.histograms))
        self.act_pdf_recipe.setEnabled(bool(self.project.pdf_datasets))
        self.act_stop.setEnabled(False)
        self.progress_bar.hide()
        self.progress_label.setText(message.split("\n")[0])
        self.param_model.refresh()
        self._refresh_plot()
        self._refresh_structure()
        if ok and self.project.last_fit is not None:
            hc = self.project.last_fit.high_correlations(0.95)
            if hc:
                self.log(
                    "注意: 強く相関するパラメータがあります(片方を固定することを検討): "
                    + "; ".join(f"{a}–{b} ({c:+.2f})" for a, b, c in hc[:5])
                )
            for note in self.project.diagnostics():
                self.log("注意: " + note)

    def open_residual_dialog(self) -> None:
        """残差の点検(未同定ピーク・こぶの候補)。"""
        if not (self.project.phases and self.project.histograms):
            QMessageBox.information(
                self, tr("残差の点検"), "先に粉末データと相を読み込み、計算か精密化をしてください"
            )
            return
        from kaisetsu.gui.residual_dialog import ResidualDialog

        dlg = ResidualDialog(self.project, self, main=self)
        dlg.exec()

    def show_summary(self) -> None:
        if not (self.project.phases and (self.project.histograms or self.project.pdf_datasets)):
            return
        text = self.project.summary()
        notes = self.project.diagnostics()
        if notes:
            text += "\n\n注意:\n" + "\n".join(f"- {n}" for n in notes)
        self.log(text)
        box = QMessageBox(self)
        box.setWindowTitle("結果の要約")
        box.setText(text[:4000])
        box.exec()

    def run_mem(self) -> None:
        if not (self.project.phases and self.project.histograms):
            QMessageBox.information(
                self, "MEM", "先に粉末データと相を読み込み、Rietveld 精密化を行ってください"
            )
            return
        from kaisetsu.gui.mem_dialog import MEMDialog

        dlg = MEMDialog(self.project, self)
        dlg.exec()
        self._refresh_all()

    def run_sequence(self) -> None:
        if not (self.project.phases and self.project.histograms):
            QMessageBox.information(
                self, tr("連続データを順に精密化"), "先に 1 つ目のデータで精密化しておいてください"
            )
            return
        from kaisetsu.gui.sequence_dialog import SequenceDialog

        dlg = SequenceDialog(self.project, self)
        dlg.exec()
        self._refresh_all()

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
    def show_user_guide(self, doc: str = "getting-started.md", heading: str | None = None) -> None:
        """ヘルプ → 利用者ガイド(F1)。同梱の説明書を別窓で開く。同梱されていなければ GitHub を開く。"""
        if user_guide_dir() is None:
            self.log(
                tr("説明書がこの配布物に入っていないので、GitHub の説明書をブラウザで開きます")
            )
            self.open_online_guide()
            return
        if self._help_window is None:
            self._help_window = HelpWindow(self)
        self._help_window.show_doc(doc, heading)
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()

    def open_online_guide(self) -> None:
        """ヘルプ → オンラインの説明書。GitHub の説明書をブラウザで開く。"""
        QDesktopServices.openUrl(QUrl(GUIDE_URL))

    def show_naming_help(self) -> None:
        QMessageBox.information(
            self,
            "パラメータ名の規則",
            "相:cell:a|b|c|alpha|beta|gamma … 格子定数\n"
            "相:サイト:x|y|z … 分率座標(特殊位置の拘束は自動)\n"
            "相:サイト:occ … 占有率、相:サイト:uiso … 等方性変位パラメータ(異方性は u11〜u23)\n"
            "h0:inst:zero|u|v|w|x|y|sh_l|pol … 装置(h0 は 1 番目のデータ。詳細: p, z)\n"
            "h0:sample:displacement|transparency|rough_a|disp_x|disp_y|mu_r … 試料(詳細: rough_b)\n"
            "h0:bkg:0,1,2,… … 背景(Chebyshev 係数)\n"
            "h0:相:scale|md_r|size|strain|ext … スケール、選択配向、サイズ、歪み、Sabine 消衰\n"
            "h0:相:size_ax … 一軸異方的サイズの軸方向成分、h0:相:S_HKL … Stephens 異方的歪み\n"
            "h0:相:sh2_1,sh4_1,… … 球面調和関数の集合組織(相の設定で有効にしたとき)\n"
            "g0:pdf:scale|qdamp|qbroad … PDF データセット(g0 は 1 番目の G(r))\n"
            "g0:相:scale|delta1|delta2|sratio|spdiameter … 相の PDF パラメータ\n\n"
            "絞り込み欄では * と ? が使えます(例: *:cell:*, h0:inst:[uvw])。",
        )

    def check_for_updates(self) -> None:
        """ヘルプ → 更新を確認。結果はダイアログかメッセージで出る。"""
        self.updater.check(manual=True)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "kaisetsu について",
            f"kaisetsu {__version__}\n初心者向け・日本語対応の粉末回折解析プログラム(Rietveld / PDF / MEM)。\n\n"
            "計算の検証は GSAS-II 5.7.4(Rietveld)と diffpy.pdffit2 1.6.0(PDF)と突き合わせています"
            "(docs/validation/)。",
        )

    # ------------------------------------------------------------------
    def open_lattice_dialog(self) -> None:
        """Le Bail で格子定数だけを出すダイアログ(放射光マイクロ XRD 向け)。"""
        from kaisetsu.gui.lattice_dialog import LatticeDialog

        dlg = LatticeDialog(self.project, self, main=self)
        dlg.exec()

    def adopt_project(self, project: Project, *, source: str = "") -> None:
        """別のところで作ったプロジェクト(Le Bail の結果など)をメインウィンドウで開く。"""
        if not self._confirm_discard():
            return
        self.project = project
        self.project_path = None
        self.current_hist = 0
        self.current_pdf = 0
        self.current_view = "hist" if project.histograms else "pdf"
        self.current_phase = None
        self._undo_values = None
        self.act_undo.setEnabled(False)
        self.log_view.clear()
        self.history_plot.clear()
        self.log(f"プロジェクトを開きました: {source or project.name}")
        for line in project.log[-20:]:
            self.log("  " + line)
        self._refresh_all()

    def _confirm_discard(self) -> bool:
        if (
            not self.project.histograms
            and not self.project.phases
            and not self.project.pdf_datasets
        ):
            return True
        r = QMessageBox.question(
            self, "確認", "現在のプロジェクトを閉じますか?(保存していない変更は失われます)"
        )
        return r == QMessageBox.StandardButton.Yes

    def _last_dir(self) -> str:
        return last_dir(self.settings)

    def _remember_dir(self, path: str) -> None:
        remember_dir(path, self.settings)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            self._thread.wait(5000)
        self.updater.wait(3000)
        event.accept()


def _guess_tof_instrument(meta: dict[str, object]) -> TOFInstrument:
    """Mantid の出力ヘッダ(DIFC と 2θ)から初期値を作る。"""
    difc = 20000.0
    tth = 90.0
    comment = str(meta.get("comment", "") or "")
    import re

    m = re.search(r"DIFC\s+([\d.]+)", comment)
    if m:
        difc = float(m.group(1))
    m = re.search(r"tth\s+([\d.]+)", comment)
    if m:
        tth = float(m.group(1))
    return TOFInstrument(difc=difc, two_theta=tth)


__all__ = ["MainWindow"]
