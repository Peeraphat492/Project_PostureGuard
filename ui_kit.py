"""Shared design tokens and small Qt helpers used by the dashboard and the Active Break game."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

BLUE = "#2563EB"
VIOLET = "#7C3AED"
CYAN = "#06B6D4"
GOOD = "#22C55E"
WARN = "#F59E0B"
POOR = "#EF4444"
COIN = "#FBBF24"
TEXT = "#0F172A"
TEXT_2 = "#64748B"
MUTED = "#94A3B8"
PREVIEW_BG = "#0F172A"


def format_seconds(seconds: float, short: bool = False) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if short and h == 0:
        return f"{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_tone(widget: QWidget, tone: str, prop: str = "tone"):
    """Switch a QSS dynamic property; re-polishes only when the value actually changes."""
    if widget.property(prop) == tone:
        return
    widget.setProperty(prop, tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def make_label(text: str, name: str, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def make_badge(text: str, tone: str, name: str = "badge") -> QLabel:
    badge = QLabel(text)
    badge.setObjectName(name)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    set_tone(badge, tone)
    return badge


# ---------------------------------------------------------------------------
# Painted icons (crisp at any DPI, no image assets needed for PyInstaller)
# ---------------------------------------------------------------------------

def paint_glyph(p: QPainter, kind: str, color: QColor, stroke: float = 2.0):
    """Draw a line icon on a 24x24 unit grid."""
    pen = QPen(color, stroke, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    def poly(*pts):
        path = QPainterPath(QPointF(*pts[0]))
        for pt in pts[1:]:
            path.lineTo(QPointF(*pt))
        p.drawPath(path)

    if kind == "shield":
        path = QPainterPath(QPointF(12, 3))
        path.lineTo(19, 6)
        path.lineTo(19, 11.5)
        path.cubicTo(19, 15.8, 16, 19.2, 12, 20.8)
        path.cubicTo(8, 19.2, 5, 15.8, 5, 11.5)
        path.lineTo(5, 6)
        path.closeSubpath()
        p.drawPath(path)
        poly((9, 12), (11.3, 14.3), (15.3, 10))
    elif kind == "camera":
        path = QPainterPath()
        path.addRoundedRect(QRectF(3, 7, 13, 10), 2.5, 2.5)
        p.drawPath(path)
        poly((16, 10.5), (21, 8), (21, 16), (16, 13.5))
    elif kind == "warning":
        poly((12, 3.5), (21.5, 20), (2.5, 20), (12, 3.5))
        poly((12, 9.5), (12, 14))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QPointF(12, 17), 1.3, 1.3)
    elif kind == "stop":
        path = QPainterPath()
        path.addRoundedRect(QRectF(6, 6, 12, 12), 2.5, 2.5)
        p.drawPath(path)
    elif kind == "play":
        poly((8, 5.5), (19, 12), (8, 18.5), (8, 5.5))
    elif kind == "back":
        poly((14.5, 5.5), (8, 12), (14.5, 18.5))
    elif kind == "history":
        path = QPainterPath()
        path.arcMoveTo(QRectF(4, 4, 16, 16), 160)
        path.arcTo(QRectF(4, 4, 16, 16), 160, -300)   # clock face, open at the left
        p.drawPath(path)
        poly((2.8, 8.4), (4.5, 11.4), (7.6, 9.9))      # arrow: "back in time"
        poly((12, 8), (12, 12), (14.8, 13.8))         # clock hands


def draw_icon(kind: str, color: str, size: int = 18) -> QPixmap:
    scale = 3
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size * scale / 24.0, size * scale / 24.0)
    paint_glyph(p, kind, QColor(color))
    p.end()
    pm.setDevicePixelRatio(scale)
    return pm
