"""アプリのアイコン(packaging/kaisetsu.ico と kaisetsu_256.png)を描いて書き出す。

    uv run python packaging/make_icon.py

Qt の QPainter で粉末回折パターン風の図を描き、16〜256 px を PNG 形式で詰めた ICO にする
(Windows Vista 以降は ICO 内の PNG を読める)。
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

HERE = Path(__file__).resolve().parent
SIZES = (16, 24, 32, 48, 64, 128, 256)
PEAKS = (
    (0.22, 0.95, 0.018),
    (0.36, 0.55, 0.016),
    (0.50, 0.72, 0.015),
    (0.63, 0.35, 0.015),
    (0.74, 0.48, 0.014),
    (0.86, 0.28, 0.014),
)


def _pattern(x: float) -> float:
    import math

    y = 0.04
    for c, h, w in PEAKS:
        y += h * math.exp(-0.5 * ((x - c) / w) ** 2)
    return y


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    # 背景: 角丸の濃紺
    grad = QLinearGradient(0, 0, 0, s)
    grad.setColorAt(0.0, QColor("#2b4c7e"))
    grad.setColorAt(1.0, QColor("#162a4a"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    r = s * 0.18
    p.drawRoundedRect(QRectF(0, 0, s, s), r, r)
    # 回折パターン(折れ線)
    margin = s * 0.12
    base = s * 0.80
    height = s * 0.58
    path = QPainterPath()
    n = max(32, size * 2)
    for i in range(n + 1):
        t = i / n
        x = margin + t * (s - 2 * margin)
        y = base - _pattern(t) * height
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    pen = QPen(QColor("#ffd166"))
    pen.setWidthF(max(1.0, s * 0.045))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    # 基線
    pen2 = QPen(QColor(255, 255, 255, 110))
    pen2.setWidthF(max(1.0, s * 0.02))
    p.setPen(pen2)
    p.drawLine(QPointF(margin, base + s * 0.02), QPointF(s - margin, base + s * 0.02))
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(path: Path, images: list[tuple[int, bytes]]) -> None:
    header = struct.pack("<HHH", 0, 1, len(images))
    entries = b""
    offset = 6 + 16 * len(images)
    blobs = b""
    for size, data in images:
        w = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        blobs += data
    path.write_bytes(header + entries + blobs)


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    images = [(sz, png_bytes(render(sz))) for sz in SIZES]
    write_ico(HERE / "kaisetsu.ico", images)
    (HERE / "kaisetsu_256.png").write_bytes(images[-1][1])
    print(f"書き出し: {HERE / 'kaisetsu.ico'} ({len(images)} サイズ)")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
