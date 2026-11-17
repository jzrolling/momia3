"""Small monochrome line icons shared by the GUI toolbars."""

from __future__ import annotations

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap

ICON_COLOR = "#e8e8e8"
ICON_SIZE = 16


def line_icon(kind: str, color: str = ICON_COLOR, size: int = ICON_SIZE) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = painter.pen()
    pen.setColor(QColor(color))
    pen.setWidthF(1.5)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    if kind == "plus":
        painter.drawLine(size // 2, 3, size // 2, size - 3)
        painter.drawLine(3, size // 2, size - 3, size // 2)
    elif kind == "minus":
        painter.drawLine(3, size // 2, size - 3, size // 2)
    elif kind == "undo":
        painter.drawLine(3, 4, 7, 2)
        painter.drawLine(3, 4, 7, 6)
        path = QPainterPath()
        path.moveTo(3, 4)
        path.cubicTo(7, 3, 13, 7, 10, 13)
        painter.drawPath(path)
    elif kind == "refresh":
        # Two clear arcs give a readable reload glyph at 16 px.
        painter.drawArc(QRectF(3, 3, 10, 10), 25 * 16, 135 * 16)
        painter.drawArc(QRectF(3, 3, 10, 10), 205 * 16, 135 * 16)
        painter.drawLine(12, 2, 14, 4)
        painter.drawLine(14, 4, 11, 5)
        painter.drawLine(4, 14, 2, 12)
        painter.drawLine(2, 12, 5, 11)
    elif kind == "folder":
        path = QPainterPath()
        path.moveTo(2, 5.2)
        path.lineTo(5.8, 5.2)
        path.lineTo(7, 6.6)
        path.lineTo(13.2, 6.6)
        path.quadTo(14, 6.6, 14, 7.4)
        path.lineTo(14, 12.4)
        path.quadTo(14, 13.2, 13.2, 13.2)
        path.lineTo(2.8, 13.2)
        path.quadTo(2, 13.2, 2, 12.4)
        path.closeSubpath()
        painter.drawPath(path)
    else:  # image
        painter.drawRoundedRect(2, 3, 12, 10, 2, 2)
        painter.drawEllipse(5, 5, 2, 2)
        path = QPainterPath()
        path.moveTo(3, 11)
        path.lineTo(7, 7)
        path.lineTo(9, 9)
        path.lineTo(11, 6.5)
        path.lineTo(13, 10)
        painter.drawPath(path)
    painter.end()
    return QIcon(pix)


def color_square_icon(color: str, size: int = 18) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(QColor(color))
    painter.setPen(QColor("#111111"))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
    painter.end()
    return QIcon(pix)
