"""残差の点検: 相の反射で説明できないピークとこぶの候補を示し、人が選んで除外・ハンプにする。

自動では除外しない(構造モデルの強度が合わない反射まで異物とみなしてしまう)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.i18n import tr
from kaisetsu.refine.project import Project
from kaisetsu.refine.residual import ResidualFeatures, find_residual_features

if TYPE_CHECKING:
    from kaisetsu.gui.main_window import MainWindow

COL_APPLY, COL_KIND, COL_HIST, COL_POS, COL_WIDTH, COL_HEIGHT, COL_NEAR, COL_NOTE = range(8)


class ResidualDialog(QDialog):
    """残差の点検ダイアログ。"""

    def __init__(
        self, project: Project, parent: QWidget | None = None, main: MainWindow | None = None
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.main = main
        self.features: list[ResidualFeatures] = []
        self._rows: list[tuple[str, int, int]] = []  # (種類, ヒストグラム, 要素の添字)
        self.setWindowTitle(tr("残差の点検"))
        self.resize(900, 520)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "差のプロット(観測 − 計算)に残った、相の反射で説明できないピークと広いこぶの候補です。"
            "不純物や試料台のピークはチェックして「除外領域に追加」、非晶質のこぶは"
            "「背景のハンプに追加」で背景に足せます。「強度の不一致」は反射の位置に観測ピークがあるのに"
            "強度が合わないもので、除外ではなく占有率・選択配向・第二相を疑ってください。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["適用", "種類", "データ", "位置", "幅 / 範囲", "高さ(最大比)", "近い反射", "判定"]
        )
        self.table.horizontalHeader().setSectionResizeMode(COL_NOTE, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.status = QLabel("")
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.exclude_btn = QPushButton(tr("選んだピークを除外領域に追加"))
        self.exclude_btn.clicked.connect(self.apply_excludes)
        row.addWidget(self.exclude_btn)
        self.hump_btn = QPushButton(tr("選んだこぶを背景のハンプに追加"))
        self.hump_btn.clicked.connect(self.apply_humps)
        row.addWidget(self.hump_btn)
        self.redo_btn = QPushButton(tr("再検出"))
        self.redo_btn.clicked.connect(self.detect)
        row.addWidget(self.redo_btn)
        row.addStretch(1)
        close = QPushButton(tr("閉じる"))
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)
        self.detect()

    # ------------------------------------------------------------------
    def detect(self) -> None:
        """残差を調べ直して表を埋める。"""
        self.features = []
        for i in range(len(self.project.histograms)):
            try:
                self.features.append(find_residual_features(self.project, i))
            except (ValueError, IndexError) as exc:
                self.status.setText(f"h{i}: 調べられませんでした({exc})")
        self._fill()

    def _fill(self) -> None:
        self.table.setRowCount(0)
        self._rows = []
        n_unidx = n_misfit = n_hump = 0
        for f in self.features:
            u = f.unit
            for j, p in enumerate(f.peaks):
                kind = "未同定ピーク" if p.unindexed else "強度の不一致"
                note = (
                    "相の反射で説明できない(不純物・試料台・第二相)"
                    if p.unindexed
                    else "反射はあるが観測が計算より強い(占有率・配向・第二相の重なり)"
                )
                self._add_row(
                    kind,
                    f.hist,
                    j,
                    f"{p.position:.3f}{u}",
                    f"{p.lo:.2f}〜{p.hi:.2f}{u}",
                    f"{p.height:.0f} ({100 * p.height / f.net_max:.0f}%)",
                    f"{p.nearest_reflection:.3f}{u}" if p.nearest_reflection is not None else "—",
                    note,
                    checkable=p.unindexed,
                )
                if p.unindexed:
                    n_unidx += 1
                else:
                    n_misfit += 1
            for j, hm in enumerate(f.humps):
                self._add_row(
                    "こぶ",
                    f.hist,
                    j,
                    f"{hm.center:.2f}{u}",
                    f"幅 {hm.fwhm:.2f}{u}",
                    f"{hm.height:.0f} ({100 * hm.height / f.net_max:.0f}%)",
                    "—",
                    "広い山(非晶質成分、または背景の項数不足)",
                    checkable=True,
                )
                n_hump += 1
        self.status.setText(
            f"未同定ピーク {n_unidx}、強度の不一致 {n_misfit}、こぶ {n_hump}"
            if self._rows
            else "残差に目立つ特徴はありません"
        )
        self.exclude_btn.setEnabled(n_unidx > 0)
        self.hump_btn.setEnabled(n_hump > 0)

    def _add_row(
        self,
        kind: str,
        hist: int,
        index: int,
        pos: str,
        width: str,
        height: str,
        near: str,
        note: str,
        *,
        checkable: bool,
    ) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        apply = QTableWidgetItem("")
        if checkable:
            apply.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            apply.setCheckState(Qt.CheckState.Unchecked)
        else:
            apply.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(r, COL_APPLY, apply)
        for c, text in (
            (COL_KIND, kind),
            (COL_HIST, f"h{hist}"),
            (COL_POS, pos),
            (COL_WIDTH, width),
            (COL_HEIGHT, height),
            (COL_NEAR, near),
            (COL_NOTE, note),
        ):
            self.table.setItem(r, c, QTableWidgetItem(text))
        self._rows.append((kind, hist, index))

    def checked(self, kind: str) -> list[tuple[int, int]]:
        """チェックされた行の (ヒストグラム, 要素の添字)。"""
        out: list[tuple[int, int]] = []
        for r, (k, hist, idx) in enumerate(self._rows):
            item = self.table.item(r, COL_APPLY)
            if k == kind and item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append((hist, idx))
        return out

    def set_checked(self, kind: str, hist: int, index: int, on: bool = True) -> None:
        for r, (k, h, idx) in enumerate(self._rows):
            if (k, h, idx) == (kind, hist, index):
                item = self.table.item(r, COL_APPLY)
                if item is not None:
                    item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def apply_excludes(self) -> int:
        """チェックした未同定ピークの範囲を除外領域に足す。"""
        chosen = self.checked("未同定ピーク")
        if not chosen:
            QMessageBox.information(
                self, tr("残差の点検"), "除外するピークにチェックを入れてください"
            )
            return 0
        by_hist: dict[int, list[tuple[float, float]]] = {}
        for hist, idx in chosen:
            p = self.features[hist].peaks[idx]
            by_hist.setdefault(hist, []).append((p.lo, p.hi))
        for hist, ranges in by_hist.items():
            h = self.project.histograms[hist]
            h.set_excluded([*h.excluded, *ranges])
            self.project.log.append(
                f"h{hist}: 残差の点検から除外領域を追加: "
                + ", ".join(f"{lo:.2f}〜{hi:.2f}" for lo, hi in ranges)
            )
        self._refresh_main()
        self.detect()
        return len(chosen)

    def apply_humps(self) -> int:
        """チェックしたこぶをガウスのハンプとして背景に足す(パラメータ h{i}:hump{k}:*)。"""
        chosen = self.checked("こぶ")
        if not chosen:
            QMessageBox.information(
                self, tr("残差の点検"), "ハンプにするこぶにチェックを入れてください"
            )
            return 0
        for hist, idx in chosen:
            hm = self.features[hist].humps[idx]
            self.project.add_hump(hist, hm.center, hm.fwhm, hm.height)
            self.project.log.append(
                f"h{hist}: 残差の点検から背景にハンプを追加: 中心 {hm.center:.2f}、幅 {hm.fwhm:.2f}、"
                f"高さ {hm.height:.0f}(パラメータ h{hist}:hump*:* を動かして精密化してください)"
            )
        self._refresh_main()
        self.detect()
        return len(chosen)

    def _refresh_main(self) -> None:
        if self.main is not None:
            self.main.param_model.refresh()
            self.main._refresh_plot()
            for line in self.project.log[-3:]:
                if "残差の点検から" in line:
                    self.main.log(line)


__all__ = ["ResidualDialog"]
