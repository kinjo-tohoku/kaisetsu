"""パラメータ表(QAbstractTableModel)とフィルタ。"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Iterable
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QBrush, QColor

from kaisetsu.refine.parameters import Parameter, ParameterSet

COLUMNS = ("名前", "値", "精密化", "esd", "単位", "説明")
COL_NAME, COL_VALUE, COL_VARY, COL_ESD, COL_UNIT, COL_DESC = range(6)
COLUMN_TIPS = (
    "パラメータの名前(相:サイト:量、h0:inst:量 など)",
    "現在の値。ダブルクリックで書き換えられます(F5 で再計算)",
    "チェックを付けた量を精密化(最小二乗で最適化)します",
    "直近の精密化で求めた標準偏差(不確かさの目安)",
    "単位",
    "説明。拘束や対称性による固定もここに出ます",
)

_GROUP_LABELS = {
    "cell": "格子定数",
    "inst": "装置",
    "sample": "試料",
    "bkg": "背景",
}


def _format_value(p: Parameter) -> str:
    v = p.value
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.5g}"
    return f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:.4f}"


class ParameterModel(QAbstractTableModel):
    """ParameterSet を表として見せる。値と精密化フラグを編集できる。"""

    def __init__(self, params: ParameterSet | None = None, parent: Any = None) -> None:
        super().__init__(parent)
        self._params: ParameterSet = params if params is not None else ParameterSet()
        self._rows: list[Parameter] = list(self._params)
        self._read_only = False

    # ------------------------------------------------------------------
    def set_read_only(self, flag: bool) -> None:
        """精密化の実行中は値も「精密化」のチェックも変えられないようにする(レビュー GUI #3)。"""
        if self._read_only == bool(flag):
            return
        self._read_only = bool(flag)
        self.refresh()

    def is_read_only(self) -> bool:
        return self._read_only

    def set_params(self, params: ParameterSet) -> None:
        self.beginResetModel()
        self._params = params
        self._rows = list(params)
        self.endResetModel()

    def refresh(self) -> None:
        """値が外で変わったとき(精密化後など)に表示を更新する。"""
        self.beginResetModel()
        self._rows = list(self._params)
        self.endResetModel()

    def parameter(self, row: int) -> Parameter:
        return self._rows[row]

    # ------------------------------------------------------------------
    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return len(COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal:
            return COLUMN_TIPS[section]
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        p = self._rows[index.row()]
        editable = p.tie is None and not p.locked and not self._read_only
        if index.column() == COL_VARY and editable:
            f |= Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == COL_VALUE and editable:
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        p = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if col == COL_NAME:
                return p.name
            if col == COL_VALUE:
                return _format_value(p) if role == Qt.ItemDataRole.DisplayRole else p.value
            if col == COL_ESD:
                return "" if p.esd is None else f"{p.esd:.3g}"
            if col == COL_UNIT:
                return p.unit
            if col == COL_DESC:
                if p.tie is not None:
                    rel = " + ".join(
                        f"{c:g}×{m}" for m, c in zip(p.tie.masters, p.tie.coeffs, strict=True)
                    )
                    off = f" + {p.tie.offset:g}" if p.tie.offset else ""
                    return f"拘束: = {rel}{off}"
                if p.locked:
                    return (p.description + " (対称性で固定)").strip()
                return p.description
            return None
        if role == Qt.ItemDataRole.CheckStateRole and col == COL_VARY:
            if p.tie is not None or p.locked:
                return None
            return Qt.CheckState.Checked if p.vary else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.ToolTipRole:
            tip = p.description or p.name
            if p.min != -math.inf or p.max != math.inf:
                tip += f"\n範囲: {p.min:g} 〜 {p.max:g}"
            if p.tie is not None:
                tip += "\n他のパラメータに拘束されています"
            if p.locked:
                tip += "\n対称性により固定されています"
            return tip
        if role == Qt.ItemDataRole.ForegroundRole:
            if p.tie is not None or p.locked:
                return QBrush(QColor(128, 128, 128))
            if p.is_free:
                return QBrush(QColor(180, 40, 40))
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (COL_VALUE, COL_ESD):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def setData(
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if self._read_only:
            return False
        p = self._rows[index.row()]
        col = index.column()
        if col == COL_VARY and role == Qt.ItemDataRole.CheckStateRole:
            if p.tie is not None or p.locked:
                return False
            p.vary = Qt.CheckState(value) == Qt.CheckState.Checked
            self.dataChanged.emit(index, index, [role])
            return True
        if col == COL_VALUE and role == Qt.ItemDataRole.EditRole:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(v):
                return False
            p.value = min(max(v, p.min), p.max)
            p.initial = p.value  # 手で入れた値は「設定した初期値」として扱う
            p.apply()
            self._params.evaluate_ties()
            self._params.apply_all()
            self.dataChanged.emit(
                self.index(index.row(), 0), self.index(index.row(), len(COLUMNS) - 1)
            )
            return True
        return False


class ParameterFilter(QSortFilterProxyModel):
    """名前の glob、種類(相 / ヒストグラム)、「精密化中のみ」、「詳細も表示」でフィルタする。

    glob だけでは `*:*:u*` が原子の U_iso と装置の U の両方に当たるので、
    クイックボタンは種類(scope)も指定する:
        'all'   … すべて
        'phase' … 相のパラメータ(名前の先頭が相の名前)だけ
        'hist'  … ヒストグラム・PDF のパラメータだけ
    詳細パラメータ(Parameter.advanced)は、動かしているか効いている値のときだけ出す。
    """

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._pattern = "*"
        self._only_free = False
        self._scope = "all"
        self._phase_names: frozenset[str] = frozenset()
        self._show_advanced = False

    def set_pattern(self, pattern: str) -> None:
        pattern = pattern.strip() or "*"
        if "*" not in pattern and "?" not in pattern:
            pattern = f"*{pattern}*"
        self._pattern = pattern
        self._refilter()

    def set_only_free(self, flag: bool) -> None:
        self._only_free = flag
        self._refilter()

    def set_scope(self, scope: str) -> None:
        """'all' / 'phase' / 'hist'。"""
        if scope not in ("all", "phase", "hist"):
            raise ValueError(f"未知の scope: {scope}")
        self._scope = scope
        self._refilter()

    def scope(self) -> str:
        return self._scope

    def set_phase_names(self, names: Iterable[str]) -> None:
        """種類フィルタで「相のパラメータ」を見分けるための相の名前。"""
        self._phase_names = frozenset(names)
        self._refilter()

    def set_show_advanced(self, flag: bool) -> None:
        self._show_advanced = flag
        self._refilter()

    def filterAcceptsRow(
        self, source_row: int, source_parent: QModelIndex | QPersistentModelIndex
    ) -> bool:
        model = self.sourceModel()
        if not isinstance(model, ParameterModel):
            return True
        p = model.parameter(source_row)
        if self._only_free and not p.is_free:
            return False
        if p.advanced and not self._show_advanced and not p.vary:
            if p.neutral_value is None or p.value == p.neutral_value:
                return False
        if self._scope != "all":
            is_phase = p.name.split(":", 1)[0] in self._phase_names
            if is_phase != (self._scope == "phase"):
                return False
        return fnmatch.fnmatchcase(p.name, self._pattern)

    def _refilter(self) -> None:
        """Qt 6.10 以降の API があればそれを使う(invalidateFilter は非推奨)。"""
        begin = getattr(self, "beginFilterChange", None)
        end = getattr(self, "endFilterChange", None)
        if callable(begin) and callable(end):
            begin()
            end()
        else:
            self.invalidateRowsFilter()
