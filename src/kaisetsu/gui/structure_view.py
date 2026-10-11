"""結晶構造の簡易 3D 表示(QPainter の正射影。OpenGL 不要)。

単位胞内の原子(境界上の等価原子も含む)を球、原子間距離が共有結合半径の和 × 1.2 以内の対を
結合として描く。マウスのドラッグで回転、ホイールで拡大縮小、ダブルクリックで初期視点。
"""

from __future__ import annotations

import math

import gemmi
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from kaisetsu.crystal.atoms import split_species
from kaisetsu.crystal.structure import Structure

# 元素の色(VESTA/Jmol 風の代表的なもの。無い元素は灰色)
_COLORS: dict[str, str] = {
    "H": "#f0f0f0", "C": "#808080", "N": "#3050f8", "O": "#ff0d0d", "F": "#90e050",
    "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6", "Si": "#f0c8a0", "P": "#ff8000",
    "S": "#ffff30", "Cl": "#1ff01f", "K": "#8f40d4", "Ca": "#3dff00", "Ti": "#bfc2c7",
    "V": "#a6a6ab", "Cr": "#8a99c7", "Mn": "#9c7ac7", "Fe": "#e06633", "Co": "#f090a0",
    "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0", "Ga": "#c28f8f", "Ge": "#668f8f",
    "Sr": "#00ff00", "Y": "#94ffff", "Zr": "#94e0e0", "Nb": "#73c2c9", "Mo": "#54b5b5",
    "Ag": "#c0c0c0", "Sn": "#668080", "Ba": "#00c900", "La": "#70d4ff", "Ce": "#ffffc7",
    "Nd": "#c7ffc7", "Gd": "#45ffc7", "W": "#2194d6", "Pt": "#d0d0e0", "Au": "#ffd123",
    "Pb": "#575961", "Bi": "#9e4fb5", "U": "#008fff",
}  # fmt: skip


def _rotation(theta: float, phi: float) -> np.ndarray:
    """x 軸回り theta、y 軸回り phi の回転行列。"""
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


class StructureView(QWidget):
    """結晶構造の 3D 表示ウィジェット。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self._structure: Structure | None = None
        self._theta = 0.35
        self._phi = -0.6
        self._zoom = 1.0
        self._last: QPointF | None = None
        self._atoms: np.ndarray = np.zeros((0, 3))  # 直交座標 [Å]
        self._labels: list[str] = []
        self._elements: list[str] = []
        self._radii: np.ndarray = np.zeros(0)
        self._bonds: list[tuple[int, int]] = []
        self._cell_edges: list[tuple[np.ndarray, np.ndarray]] = []
        self.show_bonds = True
        self.show_labels = False
        self.setToolTip("ドラッグで回転、ホイールで拡大縮小、ダブルクリックで初期視点")

    # ------------------------------------------------------------------
    def set_structure(self, structure: Structure | None) -> None:
        self._structure = structure
        self._atoms = np.zeros((0, 3))
        self._labels = []
        self._elements = []
        self._radii = np.zeros(0)
        self._bonds = []
        self._cell_edges = []
        if structure is not None and structure.n_sites > 0:
            self._build(structure)
        self.update()

    def _build(self, st: Structure) -> None:
        ex = st.expand()
        m = st.lattice.orthogonalization_matrix
        pos_list: list[np.ndarray] = []
        labels: list[str] = []
        elements: list[str] = []
        eps = 1e-4
        shifts = [np.array(v, dtype=float) for v in np.ndindex(2, 2, 2)]
        for i, p in enumerate(ex.positions):
            site = st.sites[int(ex.site_index[i])]
            base = p % 1.0
            for s in shifts:
                q = base + s
                if np.all(q <= 1.0 + eps):
                    pos_list.append(q)
                    labels.append(site.label)
                    elements.append(site.element)
        frac = np.array(pos_list).reshape(-1, 3)
        # 重複(同じ点)を除く
        if len(frac):
            key = np.round(frac, 4)
            _, idx = np.unique(key, axis=0, return_index=True)
            idx = np.sort(idx)
            frac = frac[idx]
            labels = [labels[k] for k in idx]
            elements = [elements[k] for k in idx]
        self._atoms = frac @ m.T
        self._labels = labels
        self._elements = elements
        self._radii = (
            np.array([max(float(gemmi.Element(e).covalent_r), 0.5) for e in elements], dtype=float)
            if elements
            else np.zeros(0)
        )
        # 結合
        n = len(self._atoms)
        bonds: list[tuple[int, int]] = []
        if n:
            d = np.linalg.norm(self._atoms[:, None, :] - self._atoms[None, :, :], axis=-1)
            limit = 1.2 * (self._radii[:, None] + self._radii[None, :])
            ii, jj = np.nonzero((d < limit) & (d > 0.4))
            bonds = [(int(a), int(b)) for a, b in zip(ii, jj, strict=True) if a < b]
        self._bonds = bonds
        # 単位胞の辺
        corners = {tuple(v): np.array(v, dtype=float) @ m.T for v in np.ndindex(2, 2, 2)}
        edges = []
        for a, pa in corners.items():
            for b, pb in corners.items():
                if sum(abs(x - y) for x, y in zip(a, b, strict=True)) == 1 and a < b:
                    edges.append((pa, pb))
        self._cell_edges = edges

    # ------------------------------------------------------------------
    def reset_view(self) -> None:
        self._theta, self._phi, self._zoom = 0.35, -0.6, 1.0
        self.update()

    def _project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """回転して画面座標 (N, 2) と奥行き (N,) を返す。"""
        if self._structure is None or len(self._cell_edges) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        rot = _rotation(self._theta, self._phi)
        center = np.mean(
            [e[0] for e in self._cell_edges] + [e[1] for e in self._cell_edges], axis=0
        )
        r = (pts - center) @ rot.T
        extent = max(
            float(
                np.max(np.linalg.norm(np.array([e[0] for e in self._cell_edges]) - center, axis=1))
            ),
            1.0,
        )
        scale = 0.42 * min(self.width(), self.height()) / extent * self._zoom
        xy = r[:, :2] * np.array([1.0, -1.0]) * scale + np.array(
            [self.width() / 2, self.height() / 2]
        )
        return xy, r[:, 2] * scale

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))
        if self._structure is None or not len(self._atoms):
            painter.setPen(QColor("#666"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "相を選ぶと構造を表示します"
            )
            painter.end()
            return
        # 単位胞
        pen = QPen(QColor("#555"), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for a, b in self._cell_edges:
            xy, _ = self._project(np.array([a, b]))
            painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))
        xy, depth = self._project(self._atoms)
        scale = self._project(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))[0]
        px_per_a = float(np.linalg.norm(scale[1] - scale[0]))
        # 結合(奥から)
        if self.show_bonds:
            painter.setPen(QPen(QColor("#777"), max(1.0, 0.12 * px_per_a)))
            order_b = sorted(self._bonds, key=lambda t: (depth[t[0]] + depth[t[1]]) / 2)
            for i, j in order_b:
                painter.drawLine(QPointF(*xy[i]), QPointF(*xy[j]))
        # 原子(奥から描く)
        order = np.argsort(depth)
        dmin, dmax = float(depth.min()), float(depth.max())
        for k in order:
            el = self._elements[k]
            color = QColor(_COLORS.get(el, "#b0b0b0"))
            shade = 0.65 + 0.35 * ((depth[k] - dmin) / (dmax - dmin) if dmax > dmin else 1.0)
            c = QColor(
                int(color.red() * shade), int(color.green() * shade), int(color.blue() * shade)
            )
            r = 0.45 * self._radii[k] * px_per_a
            painter.setBrush(QBrush(c))
            painter.setPen(QPen(QColor("#222"), 1))
            painter.drawEllipse(QPointF(*xy[k]), r, r)
            if self.show_labels:
                painter.setPen(QColor("#000"))
                painter.drawText(QPointF(xy[k][0] + r, xy[k][1] - r), self._labels[k])
        painter.setPen(QColor("#333"))
        lat = self._structure.lattice
        painter.drawText(
            8, self.height() - 8,
            f"{self._structure.spacegroup.hm}  a={lat.a:.4f} b={lat.b:.4f} c={lat.c:.4f} Å  "
            f"{len(self._atoms)} 原子",
        )  # fmt: skip
        painter.end()

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last is None:
            return
        d = event.position() - self._last
        self._last = event.position()
        self._phi += d.x() * 0.01
        self._theta += d.y() * 0.01
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.reset_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._zoom *= 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self._zoom = min(max(self._zoom, 0.2), 8.0)
        self.update()

    def element_of(self, species: str) -> str:
        return split_species(species)[0]


__all__ = ["StructureView"]
