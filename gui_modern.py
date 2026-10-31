from __future__ import annotations

import logging
import os
import stat
import sys
import threading
import time
from dataclasses import replace

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QLibraryInfo, QPoint, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from camera import CameraSource
from config import CONFIG
from game.active_break import ActiveBreak
from game.active_break_view import ActiveBreakView
from game.body_pose import BodyPose
from pose_engine import PoseEngine, mirror_landmarks, side_overrides
from history import HistoryStore
from history_view import HistoryView
from notifications import BAD_RISKS, PopupNotifier, result_evidence
from posture_analyzer import METRICS, FeatureSet, PostureAnalyzer
from session import SessionTracker
from ui_kit import (
    BLUE, CYAN, GOOD, MUTED, POOR, PREVIEW_BG, TEXT, TEXT_2, VIOLET, WARN,
    draw_icon, format_seconds, make_badge, make_label, paint_glyph, set_tone,
)

log = logging.getLogger("posture_guard")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
FRAME_ERROR_LIMIT = 30  # consecutive failed frames (~1 s) before monitoring is stopped safely
LETTERBOX_BG = "#1E293B"  # dark navy around a camera frame narrower than its preview (theme slate-800)

# ---------------------------------------------------------------------------
# Layout tokens
# ---------------------------------------------------------------------------

PAGE_MARGIN = 20
GAP = 14
CARD_PAD = 16
PREVIEW_MIN = 160
# Previews use cover (fill) rendering: the frame is centre-cropped to the preview box. The box may be up to this
# much wider than the 4:3 frame, which bounds the crop to 1 - 1/1.16 = ~14% of the frame height (~7% top/bottom).
PREVIEW_WIDEN = 1.16

APP_QSS = """
* { outline: 0; }
QWidget#root, QWidget#page { background: #F8FAFC; }
QLabel { color: #0F172A; background: transparent; }
QToolTip {
    background: #0F172A; color: #F8FAFC; border: none;
    border-radius: 6px; padding: 6px 8px; font-size: 12px;
}

/* ---------- Header ---------- */
QFrame#topbar {
    background: #FFFFFF;
    border: none;
    border-bottom: 1px solid #E2E8F0;
}
QLabel#brandTitle { font-size: 17px; font-weight: 700; color: #0F172A; }
QLabel#brandSubtitle { font-size: 12px; color: #64748B; }

/* ---------- Scroll fallback (small windows only) ---------- */
QScrollArea#workspace { background: #F8FAFC; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 6px 2px 6px 0; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 4px; min-height: 40px; }
QScrollBar::handle:vertical:hover { background: #94A3B8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

/* ---------- Cards & text ---------- */
QFrame#card, QFrame#toolbar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}
QLabel#cardTitle { font-size: 14px; font-weight: 700; color: #0F172A; }
QLabel#cardCaption { font-size: 12px; color: #64748B; }
QLabel#fieldLabel { font-size: 12px; font-weight: 600; color: #64748B; }
QLabel#statLabel { font-size: 12px; color: #64748B; }
QLabel#statValue { font-size: 13px; font-weight: 700; color: #0F172A; }
QLabel#cameraNumber {
    background: #EFF6FF; color: #2563EB;
    border-radius: 8px; font-size: 11px; font-weight: 700;
}

/* ---------- Badges ---------- */
QLabel#badge { border-radius: 9px; padding: 4px 10px; font-size: 11px; font-weight: 700; }
QLabel#headerBadge { border-radius: 13px; padding: 6px 14px; font-size: 12px; font-weight: 600; }
QLabel#badge[tone="good"], QLabel#headerBadge[tone="good"] { background: #DCFCE7; color: #16A34A; }
QLabel#badge[tone="warning"], QLabel#headerBadge[tone="warning"] { background: #FEF3C7; color: #B45309; }
QLabel#badge[tone="poor"], QLabel#headerBadge[tone="poor"] { background: #FEE2E2; color: #DC2626; }
QLabel#badge[tone="neutral"], QLabel#headerBadge[tone="neutral"] { background: #F1F5F9; color: #64748B; }

/* ---------- Current posture ---------- */
QLabel#postureState { font-size: 20px; font-weight: 700; color: #0F172A; }
QLabel#postureState[tone="good"] { color: #16A34A; }
QLabel#postureState[tone="warning"] { color: #D97706; }
QLabel#postureState[tone="poor"] { color: #DC2626; }
QLabel#postureReason { font-size: 12px; color: #64748B; }
QLabel#durationPill {
    background: #F1F5F9; color: #475569;
    border-radius: 9px; padding: 6px 10px;
    font-size: 12px; font-weight: 600;
}

/* ---------- Posture score ---------- */
QLabel#scoreValue { font-size: 34px; font-weight: 700; color: #94A3B8; }
QLabel#scoreValue[tone="good"] { color: #16A34A; }
QLabel#scoreValue[tone="warning"] { color: #D97706; }
QLabel#scoreValue[tone="poor"] { color: #DC2626; }
QLabel#scoreSuffix { font-size: 13px; font-weight: 600; color: #94A3B8; padding-bottom: 6px; }
QLabel#componentName { font-size: 12px; color: #64748B; }
QLabel#componentValue { font-size: 12px; font-weight: 700; color: #0F172A; }

/* ---------- Calibration ---------- */
QLabel#detectStatus { font-size: 12px; font-weight: 600; color: #0F172A; }
QLabel#detectStatus[tone="good"] { color: #16A34A; }
QLabel#detectStatus[tone="warning"] { color: #B45309; }
QLabel#detectStatus[tone="poor"] { color: #DC2626; }
QLabel#detectPercent { font-size: 11px; font-weight: 700; color: #0891B2; }

/* ---------- Inputs ---------- */
QComboBox {
    combobox-popup: 0;   /* drop-down list below the box (not a menu placed over it) */
    background: #FFFFFF;
    color: #0F172A;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0 30px 0 12px;
    min-height: 36px;
    font-size: 13px;
    font-weight: 600;
}
QComboBox:hover { border-color: #CBD5E1; }
QComboBox:focus, QComboBox:on { border-color: #2563EB; }
QComboBox::drop-down { border: none; width: 28px; subcontrol-origin: padding; subcontrol-position: center right; }
QComboBox::down-arrow { image: none; width: 0; height: 0; }
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 4px;
    outline: 0;
    selection-background-color: #EFF6FF;
    selection-color: #1D4ED8;
}
QComboBox QAbstractItemView::item { min-height: 30px; padding: 0 8px; border: none; border-radius: 6px; color: #0F172A; }
QComboBox QAbstractItemView::item:hover { background: #F1F5F9; }
QComboBox QAbstractItemView::item:selected { background: #EFF6FF; color: #1D4ED8; }

/* ---------- Buttons ---------- */
QPushButton {
    background: #FFFFFF;
    color: #0F172A;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0 14px;
    min-height: 36px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover { background: #F8FAFC; border-color: #CBD5E1; }
QPushButton:pressed { background: #F1F5F9; border-color: #94A3B8; }
QPushButton:focus { border-color: #93C5FD; }
QPushButton#primaryButton { background: #2563EB; border-color: #2563EB; color: #FFFFFF; }
QPushButton#primaryButton:hover { background: #1D4ED8; border-color: #1D4ED8; }
QPushButton#primaryButton:pressed { background: #1E40AF; border-color: #1E40AF; }
QPushButton#dangerButton { background: #EF4444; border-color: #EF4444; color: #FFFFFF; }
QPushButton#dangerButton:hover { background: #DC2626; border-color: #DC2626; }
QPushButton#dangerButton:pressed { background: #B91C1C; border-color: #B91C1C; }
QPushButton#dangerButton:disabled { background: #FECACA; border-color: #FECACA; color: #FFFFFF; }
QPushButton#outlineButton { background: #FFFFFF; border-color: #E2E8F0; color: #0F172A; }
QPushButton#outlineButton:hover { background: #F8FAFC; border-color: #CBD5E1; }
QPushButton#outlineButton:pressed { background: #F1F5F9; border-color: #94A3B8; }

/* ---------- Progress ---------- */
QProgressBar {
    background: #E2E8F0;
    border: none;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk { background: #2563EB; border-radius: 3px; }
QProgressBar#scoreBar { min-height: 8px; max-height: 8px; border-radius: 4px; }
QProgressBar#scoreBar::chunk { background: #CBD5E1; border-radius: 4px; }
QProgressBar#scoreBar[tone="good"]::chunk { background: #22C55E; }
QProgressBar#scoreBar[tone="warning"]::chunk { background: #F59E0B; }
QProgressBar#scoreBar[tone="poor"]::chunk { background: #EF4444; }
QProgressBar#detectBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #06B6D4, stop:1 #2563EB);
}

/* ---------- Posture warning popup ---------- */
QDialog#postureAlert { background: transparent; }
QFrame#popupCard { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px; }
QLabel#popupTitle { font-size: 17px; font-weight: 700; color: #0F172A; }
QLabel#popupTitle[tone="warning"] { color: #C2410C; }
QLabel#popupTitle[tone="poor"] { color: #B91C1C; }
QLabel#popupProblem { font-size: 14px; font-weight: 700; color: #0F172A; }
QLabel#popupMessage { font-size: 13px; color: #475569; }
QLabel#popupMeta {
    background: #F1F5F9; color: #475569;
    border-radius: 8px; padding: 5px 10px;
    font-size: 12px; font-weight: 600;
}
QPushButton#popupPrimary { color: #FFFFFF; min-width: 92px; }
QPushButton#popupPrimary[tone="warning"] { background: #F97316; border-color: #F97316; }
QPushButton#popupPrimary[tone="warning"]:hover { background: #EA580C; border-color: #EA580C; }
QPushButton#popupPrimary[tone="warning"]:pressed { background: #C2410C; border-color: #C2410C; }
QPushButton#popupPrimary[tone="poor"] { background: #EF4444; border-color: #EF4444; }
QPushButton#popupPrimary[tone="poor"]:hover { background: #DC2626; border-color: #DC2626; }
QPushButton#popupPrimary[tone="poor"]:pressed { background: #B91C1C; border-color: #B91C1C; }

QLabel#popupTitle[tone="info"] { color: #1D4ED8; }
QPushButton#popupPrimary[tone="neutral"] { background: #FFFFFF; border-color: #E2E8F0; color: #0F172A; }
QPushButton#popupPrimary[tone="neutral"]:hover { background: #F8FAFC; border-color: #CBD5E1; }
QPushButton#popupBreak { background: #FFFFFF; border-color: #BBF7D0; color: #15803D; }
QPushButton#popupBreak:hover { background: #F0FDF4; border-color: #86EFAC; }
QPushButton#popupBreak[tone="solid"], QPushButton#breakButton { background: #22C55E; border-color: #22C55E; color: #FFFFFF; }
QPushButton#popupBreak[tone="solid"]:hover, QPushButton#breakButton:hover { background: #16A34A; border-color: #16A34A; }
QPushButton#breakButton:pressed { background: #15803D; border-color: #15803D; }
QPushButton#breakButton:disabled { background: #BBF7D0; border-color: #BBF7D0; }

/* ---------- Dialogs ---------- */
QMessageBox { background: #FFFFFF; }
QMessageBox QLabel { color: #0F172A; font-size: 13px; }
QMessageBox QPushButton { min-width: 84px; }
"""

# risk level -> (tone, badge text)
RISK_UI = {
    "GOOD": ("good", "Good"),
    "WARNING": ("warning", "Warning"),
    "POOR": ("poor", "Poor"),
    "N/A": ("neutral", "No data"),
}

# risk level -> (tone, accent, soft background, title, message) for the posture popup
POPUP_UI = {
    "WARNING": ("warning", "#F97316", "#FFF7ED", "Posture Warning", ""),
    "POOR": ("poor", "#EF4444", "#FEF2F2", "Posture Warning", ""),
    "BREAK": (
        "info", BLUE, "#EFF6FF", "Time for an Active Break",
        f"You've been sitting for a while. Play a quick "
        f"{max(1, round(CONFIG.active_break_total_seconds / 60))}-minute movement game to reset your body.",
    ),
}


# Posture popup: the most significant flagged indicator -> (headline, short correction tip)
POPUP_PROBLEMS = {
    "Forward head": ("Forward Head Detected", "Move your head back into a comfortable neutral position."),
    "Neck flexion": ("Head Tilted Down", "Lift your gaze and keep the screen close to eye level."),
    "Trunk forward lean": ("Slouching Detected", "Sit back against the chair and lengthen your back."),
    "Trunk recline": ("Reclining Too Far", "Sit a little more upright with your back supported."),
    "Hunched posture": ("Hunched Posture Detected", "Sit tall and let your shoulders relax back."),
    "Shoulder tilt": ("Uneven Shoulders Detected", "Relax both shoulders and keep them level."),
    "Head tilt": ("Head Tilted Sideways", "Bring your head back to level."),
    "Side lean": ("Leaning to One Side", "Center your weight evenly on both hips."),
    "Upper-body alignment": ("Leaning to One Side", "Center your head over your shoulders."),
}
POPUP_PROBLEM_DEFAULT = ("Incorrect Posture Detected", "Adjust your sitting position.")


def main_problem(result) -> str | None:
    """The flagged indicator that deviates most from its limit (None when nothing is flagged)."""
    flagged = [(m[2], name) for name, m in result.metrics.items() if m[3]]
    return max(flagged)[1] if flagged else None


SHORT_NAMES = {"Upper-body alignment": "Upper body", "Trunk forward lean": "Trunk lean"}


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

def make_card(name: str = "card", spacing: int = 10):
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(CARD_PAD, CARD_PAD - 2, CARD_PAD, CARD_PAD - 2)
    layout.setSpacing(spacing)
    return frame, layout


class LogoMark(QWidget):
    """Gradient brand mark: royal blue -> violet with a posture shield."""

    def __init__(self, size: int = 40):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0.0, QColor(BLUE))
        grad.setColorAt(1.0, QColor(VIOLET))
        path = QPainterPath()
        path.addRoundedRect(r, r.width() * 0.3, r.height() * 0.3)
        p.fillPath(path, grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(CYAN))
        p.drawEllipse(QPointF(r.width() * 0.8, r.height() * 0.2), r.width() * 0.07, r.width() * 0.07)
        glyph = r.width() * 0.62
        p.translate((r.width() - glyph) / 2, (r.height() - glyph) / 2)
        p.scale(glyph / 24.0, glyph / 24.0)
        paint_glyph(p, "shield", QColor("#FFFFFF"), 2.1)


class CameraSelector(QComboBox):
    """Combo box exposing the QSpinBox-style value()/setValue() API used by the camera logic."""

    def __init__(self, value: int, maximum: int = 10):
        super().__init__()
        self.setView(QListView())
        self.setItemDelegate(QStyledItemDelegate(self))   # honours the QSS item height/padding in list mode
        for index in range(maximum + 1):
            self.addItem(f"Camera {index}")
        self.setValue(value)
        self.setMinimumWidth(124)
        self.setMaxVisibleItems(maximum + 1)   # whole list, no scrollbar
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # The list lives in a separate popup window (QComboBoxPrivateContainer) that paints its own
        # rectangular white background/frame around the rounded list: make that window frameless
        # and transparent so only the styled, rounded list is visible.
        popup = self.view().parentWidget()
        popup.setWindowFlags(popup.windowFlags() | Qt.WindowType.FramelessWindowHint
                             | Qt.WindowType.NoDropShadowWindowHint)
        popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        popup.setStyleSheet(f"{popup.metaObject().className()} {{ background: transparent; border: none; }}")

    def showPopup(self):
        super().showPopup()
        self.view().doItemsLayout()  # row positions were cached before the list got its styled 30 px rows
        # Attach the list to the box: same left edge and width, a small gap below (or above, if Qt
        # had to open it upwards near the bottom of the screen).
        popup = self.view().parentWidget()
        top_left = self.mapToGlobal(QPoint(0, 0))
        below = popup.y() >= top_left.y()
        y = top_left.y() + self.height() + 4 if below else top_left.y() - popup.height() - 4
        popup.setGeometry(top_left.x(), y, self.width(), popup.height())

    def value(self) -> int:
        return self.currentIndex()

    def setValue(self, value: int):
        self.setCurrentIndex(max(0, min(int(value), self.count() - 1)))

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(TEXT_2), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        cx, cy = self.width() - 16, self.height() / 2
        path = QPainterPath(QPointF(cx - 4, cy - 2))
        path.lineTo(cx, cy + 2)
        path.lineTo(cx + 4, cy - 2)
        p.drawPath(path)


class CameraView(QLabel):
    """Dark, rounded preview surface. Still a QLabel, so setPixmap()/size() work as before."""

    RADIUS = 12

    def __init__(self, hint: str):
        super().__init__()
        self._hint = hint
        self._icon = draw_icon("camera", MUTED, 26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(240)
        self.setFixedHeight(300)

    # Pixmaps must never drive the layout size (avoids grow-only feedback loops). The width cap set by
    # _fit_previews (frame aspect at the current height) is the preferred width, so the centred view fills it.
    def sizeHint(self) -> QSize:
        cap = self.maximumWidth()
        return QSize(cap if cap < 16777215 else 480, self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(240, self.minimumHeight())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(rect, self.RADIUS, self.RADIUS)
        p.fillPath(path, QColor(PREVIEW_BG))

        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            p.setClipPath(path)
            p.fillPath(path, QColor(LETTERBOX_BG))  # side areas the frame does not cover
            # Cover the surface (centred, overflow clipped by the rounded path) even if the last frame was
            # cropped/scaled for a slightly different size, so no empty bars appear.
            size = pm.deviceIndependentSize().scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)
            target = QRectF(
                (rect.width() - size.width()) / 2,
                (rect.height() - size.height()) / 2,
                size.width(),
                size.height(),
            )
            p.drawPixmap(target, pm, QRectF(pm.rect()))
            return

        cx, cy = rect.width() / 2, rect.height() / 2 - 16
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1E293B"))
        p.drawEllipse(QPointF(cx, cy), 28, 28)
        icon_size = self._icon.deviceIndependentSize()
        p.drawPixmap(QPointF(cx - icon_size.width() / 2, cy - icon_size.height() / 2), self._icon)

        font = QFont(self.font())
        font.setPixelSize(13)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QColor("#CBD5E1"))
        p.drawText(QRectF(0, cy + 38, rect.width(), 20), Qt.AlignmentFlag.AlignCenter, "Camera preview will appear here")
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor(TEXT_2))
        p.drawText(QRectF(0, cy + 58, rect.width(), 18), Qt.AlignmentFlag.AlignCenter, self._hint)


class MetricTile(QWidget):
    """Compact stat row: colored dot, label, right-aligned value (updated via .value)."""

    def __init__(self, title: str, value: str = "0", accent: str = BLUE):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{accent}; border-radius:4px;")
        row.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(make_label(title, "statLabel"), 1)
        self.value = make_label(value, "statValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.value)


class PostureAlertDialog(QDialog):
    """Non-blocking, single-instance posture warning. Shown with show(), never exec(),
    so the camera timer keeps running while it is open."""

    snoozed = pyqtSignal()
    break_requested = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(
            parent,
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("postureAlert")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(480)
        self._risk = None
        self._shown_seconds = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 28)  # room for the soft shadow
        card = QFrame()
        card.setObjectName("popupCard")
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(15, 23, 42, 50))
        card.setGraphicsEffect(shadow)
        outer.addWidget(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 20)
        layout.setSpacing(16)

        head = QHBoxLayout()
        head.setSpacing(14)
        self._icon = QLabel()
        self._icon.setFixedSize(48, 48)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(6)
        self._title = make_label("", "popupTitle")
        self._problem = make_label("", "popupProblem", wrap=True)
        self._message = make_label("", "popupMessage", wrap=True)
        text.addWidget(self._title)
        text.addWidget(self._problem)
        text.addWidget(self._message)
        text.addSpacing(4)
        meta_row = QHBoxLayout()
        self._meta = make_label("", "popupMeta")
        meta_row.addWidget(self._meta)
        meta_row.addStretch(1)
        text.addLayout(meta_row)
        head.addLayout(text, 1)
        layout.addLayout(head)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)
        self._snooze_button = QPushButton("Snooze 1 min")
        self._snooze_button.setObjectName("outlineButton")
        self._snooze_button.clicked.connect(self._snooze)
        self._break_button = QPushButton("Active Break")
        self._break_button.setObjectName("popupBreak")
        self._break_button.clicked.connect(self._start_break)
        self._dismiss = QPushButton("Dismiss")
        self._dismiss.setObjectName("popupPrimary")
        self._dismiss.clicked.connect(self.close)
        for button in (self._snooze_button, self._break_button, self._dismiss):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            footer.addWidget(button)
        layout.addLayout(footer)

    @property
    def kind(self):
        """"WARNING" / "POOR" for posture warnings, "BREAK" for the sitting-too-long prompt."""
        return self._risk

    def set_state(self, risk: str, bad_seconds: float = 0.0, problem: str | None = None):
        """Switch WARNING/POOR/BREAK styling in place; refresh the posture problem and its duration."""
        if risk != self._risk and risk in POPUP_UI:
            self._risk = risk
            tone, accent, soft, title, message = POPUP_UI[risk]
            is_break = risk == "BREAK"
            self._title.setText(title)
            self._problem.setVisible(not is_break)
            if is_break:
                self._message.setText(message)
            self._icon.setPixmap(draw_icon("play" if is_break else "warning", accent, 26))
            self._icon.setStyleSheet(f"background:{soft}; border-radius:24px;")
            set_tone(self._title, tone)
            self._meta.setVisible(not is_break)
            self._snooze_button.setVisible(not is_break)
            self._break_button.setText("Start Active Break" if is_break else "Active Break")
            set_tone(self._break_button, "solid" if is_break else "")
            self._dismiss.setText("Not now" if is_break else "Dismiss")
            set_tone(self._dismiss, "neutral" if is_break else tone)
        if risk == "BREAK":
            return
        headline, tip = POPUP_PROBLEMS.get(problem, POPUP_PROBLEM_DEFAULT)
        if headline != self._problem.text():
            self._problem.setText(headline)
            self._message.setText(tip)
        seconds = int(bad_seconds)
        if seconds != self._shown_seconds:
            self._shown_seconds = seconds
            self._meta.setText(f"Bad posture for {format_seconds(seconds, short=True)}")

    def show_over(self, window: QWidget):
        self.adjustSize()
        area = window.frameGeometry() if window.isVisible() and not window.isMinimized() \
            else window.screen().availableGeometry()
        x = area.center().x() - self.width() // 2
        y = area.top() + max(40, (area.height() - self.height()) // 3)
        self.move(x, y)
        self.show()
        self.raise_()

    def _snooze(self):
        self.snoozed.emit()
        self.close()

    def _start_break(self):
        self.close()
        self.break_requested.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Guard — Dual Camera Monitor")
        self.resize(1440, 860)
        self.setMinimumSize(1024, 640)

        self.front_cam: CameraSource | None = None
        self.side_cam: CameraSource | None = None
        self.front_pose = PoseEngine()
        self.side_pose = PoseEngine(CONFIG.side_landmark_visibility_threshold, side_overrides())
        self.analyzer = PostureAnalyzer()
        self.session = SessionTracker()
        self.history = HistoryStore()  # local SQLite session history (errors are reported, never fatal)

        self.notifier = PopupNotifier()  # popup timing rules (the only notification UI)
        self.session_started = False
        self._camera_state_cache = None
        self._risk_cache = None
        self._monitored = False  # a reliable posture evaluation has happened since the cameras opened
        self._camera_busy = False  # guards against re-entrant Open/Stop (e.g. clicks during a dialog)
        self._opening = None       # (front, side, results, threads) while cameras open in the background
        self._break_after_open = False
        self._frame_errors = 0     # consecutive failed frames
        self._detect_errors = {"front": 0, "side": 0}
        self._last_error = None    # last logged error signature (full traceback is logged once per signature)
        self.active_break: ActiveBreak | None = None
        self._clock = time.monotonic  # game clock (replaceable in tests)

        self._build_ui()
        self._set_camera_state(False, False)
        self._set_risk("N/A")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(CONFIG.refresh_ms)
        # The open session is saved periodically (limits data loss if the app is killed), on Stop and on close.
        self.history_timer = QTimer(self)
        self.history_timer.timeout.connect(self._save_history)
        self.history_timer.start(int(CONFIG.history_autosave_seconds * 1000))

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCentralWidget(root)

        shell = QVBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_header())

        # Everything is sized to fit one screen; the scroll area only kicks in on very small windows.
        scroll = QScrollArea()
        scroll.setObjectName("workspace")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.installEventFilter(self)
        self._workspace_scroll = scroll
        self._stack = QStackedWidget()
        self._stack.addWidget(scroll)
        self.break_view = ActiveBreakView()
        self.break_view.end_requested.connect(lambda: self.active_break and self.active_break.finish_early(self._clock()))
        self.break_view.return_requested.connect(self._finish_active_break)
        self._stack.addWidget(self.break_view)
        self.history_view = HistoryView(self.history)
        self.history_view.back_requested.connect(self.close_history)
        self._stack.addWidget(self.history_view)
        shell.addWidget(self._stack, 1)

        page = QWidget()
        page.setObjectName("page")
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.setWidget(page)

        content = QVBoxLayout(page)
        content.setContentsMargins(PAGE_MARGIN, 16, PAGE_MARGIN, 16)
        content.setSpacing(GAP)
        self._toolbar = self._build_toolbar()
        content.addWidget(self._toolbar)

        cameras = QHBoxLayout()
        cameras.setSpacing(GAP)
        front_card, self.front_view, self.front_cam_status = self._camera_panel(
            "Front Camera", "Shoulders & lateral alignment", "01"
        )
        side_card, self.side_view, self.side_cam_status = self._camera_panel(
            "Side Camera", "Head & torso alignment", "02"
        )
        self._camera_cards = (front_card, side_card)
        cameras.addWidget(front_card, 1)
        cameras.addWidget(side_card, 1)
        content.addLayout(cameras)

        bottom = QWidget()
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(GAP)
        bottom_row.addWidget(self._build_posture_card(), 4)
        bottom_row.addWidget(self._build_score_card(), 3)
        bottom_row.addWidget(self._build_session_card(), 3)
        bottom_row.addWidget(self._build_durations_card(), 3)
        bottom.installEventFilter(self)
        self._bottom_row = bottom
        content.addWidget(bottom)
        content.addStretch(1)

        # One reusable popup instance (posture warning or break reminder), never two at once.
        self.posture_popup = PostureAlertDialog(self)
        self.posture_popup.finished.connect(lambda _result: self.notifier.closed())
        self.posture_popup.snoozed.connect(lambda: self.notifier.snooze(self._clock()))
        self.posture_popup.break_requested.connect(self.start_active_break)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("topbar")
        row = QHBoxLayout(header)
        row.setContentsMargins(PAGE_MARGIN, 10, PAGE_MARGIN, 10)
        row.setSpacing(12)
        row.addWidget(LogoMark(36))
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(make_label("Posture Guard", "brandTitle"))
        titles.addWidget(make_label("AI-powered real-time posture monitoring", "brandSubtitle"))
        row.addLayout(titles)
        row.addStretch(1)

        self.btn_break = self._button("Start Active Break", "breakButton", "play", "#FFFFFF",
                                      "Play a short movement game using the front camera")
        self.btn_break.clicked.connect(self.start_active_break)
        row.addWidget(self.btn_break, 0, Qt.AlignmentFlag.AlignVCenter)

        self.connection_badge = make_badge("●  Cameras offline", "poor", "headerBadge")
        row.addWidget(self.connection_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        return header

    def _build_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(CARD_PAD, 12, CARD_PAD, 12)
        row.setSpacing(8)

        self.front_index = CameraSelector(CONFIG.front_camera_index)
        self.side_index = CameraSelector(CONFIG.side_camera_index)
        for caption, selector in (("Front", self.front_index), ("Side", self.side_index)):
            row.addWidget(make_label(caption, "fieldLabel"))
            row.addWidget(selector)
            row.addSpacing(4)

        # Automatic detection status (no calibration step): readiness / camera guidance + data coverage.
        detect = QVBoxLayout()
        detect.setSpacing(5)
        self.detect_status = make_label("Open both cameras to start automatic monitoring", "detectStatus", wrap=True)
        detect.addWidget(self.detect_status)
        coverage_row = QHBoxLayout()
        coverage_row.setSpacing(8)
        self.detect_coverage = self._progress("detectBar")
        self.detect_coverage.setToolTip("Share of posture indicators that can be measured reliably right now")
        coverage_percent = make_label("0%", "detectPercent")
        coverage_percent.setFixedWidth(38)
        coverage_percent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.detect_coverage.valueChanged.connect(lambda v: coverage_percent.setText(f"{v}%"))
        coverage_row.addWidget(self.detect_coverage, 1, Qt.AlignmentFlag.AlignVCenter)
        coverage_row.addWidget(coverage_percent)
        detect.addLayout(coverage_row)
        detect_box = QWidget()
        detect_box.setLayout(detect)
        detect_box.setMinimumWidth(140)
        detect_box.layout().setContentsMargins(8, 0, 8, 0)
        row.addWidget(detect_box, 1)

        self.btn_open = self._button("Open Cameras", "primaryButton", "camera", "#FFFFFF",
                                     "Connect the front and side cameras")
        self.btn_open.clicked.connect(self.open_cameras)
        self.btn_history = self._button("History", "outlineButton", "history", TEXT,
                                        "View your previous monitoring sessions")
        self.btn_history.clicked.connect(self.show_history)
        self.btn_stop = self._button("Stop Cameras", "dangerButton", "stop", "#FFFFFF",
                                     "Stop both cameras and pause monitoring (session data is kept)")
        self.btn_stop.clicked.connect(self.stop_cameras)
        self.btn_stop.setEnabled(False)
        for button in (self.btn_open, self.btn_stop, self.btn_history):
            row.addWidget(button)
        return toolbar

    @staticmethod
    def _button(text: str, name: str, icon: str, icon_color: str, tooltip: str) -> QPushButton:
        button = QPushButton(f" {text}")
        button.setObjectName(name)
        button.setIcon(QIcon(draw_icon(icon, icon_color, 16)))
        button.setIconSize(QSize(16, 16))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip)
        return button

    def _camera_panel(self, title: str, subtitle: str, number: str):
        card, box = make_card()

        top = QHBoxLayout()
        top.setSpacing(10)
        index_badge = make_label(number, "cameraNumber")
        index_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        index_badge.setFixedSize(28, 28)
        top.addWidget(index_badge)
        top.addWidget(make_label(title, "cardTitle"))
        top.addWidget(make_label(subtitle, "cardCaption"), 1)
        status = make_badge("●  OFFLINE", "poor")
        top.addWidget(status)
        box.addLayout(top)

        view = CameraView("Press Open Cameras to start streaming")
        box.addWidget(view, 0, Qt.AlignmentFlag.AlignHCenter)
        return card, view, status

    def _build_posture_card(self) -> QFrame:
        card, layout = make_card(spacing=6)
        top = QHBoxLayout()
        top.addWidget(make_label("Current Posture", "cardTitle"), 1)
        self.risk_badge = make_badge("Inactive", "neutral")
        top.addWidget(self.risk_badge)
        layout.addLayout(top)
        layout.addSpacing(4)

        self.status = make_label("Not monitoring yet", "postureState", wrap=True)
        self.reason = make_label(
            "Open both cameras — posture detection starts automatically.", "postureReason", wrap=True
        )
        layout.addWidget(self.status)
        layout.addWidget(self.reason)
        layout.addStretch(1)

        duration_row = QHBoxLayout()
        self.duration = make_label("Bad posture duration  ·  00:00:00", "durationPill")
        duration_row.addWidget(self.duration)
        duration_row.addStretch(1)
        layout.addLayout(duration_row)
        return card

    def _build_score_card(self) -> QFrame:
        card, layout = make_card(spacing=8)
        top = QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(make_label("Posture Score", "cardTitle"), 1, Qt.AlignmentFlag.AlignTop)
        self.score = make_label("--", "scoreValue")
        set_tone(self.score, "neutral")
        top.addWidget(self.score, 0, Qt.AlignmentFlag.AlignBottom)
        top.addWidget(make_label("/100", "scoreSuffix"), 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(top)

        self.score_bar = self._progress("scoreBar")
        set_tone(self.score_bar, "neutral")
        layout.addWidget(self.score_bar)
        layout.addSpacing(4)

        self.component_bars = {}
        for name in ("Neck", "Back", "Shoulder", "Alignment"):
            row = QHBoxLayout()
            row.setSpacing(10)
            label = make_label(name, "componentName")
            label.setFixedWidth(64)
            bar = self._progress("componentBar")
            value = make_label("--", "componentValue")
            value.setFixedWidth(26)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label)
            row.addWidget(bar, 1, Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(value)
            layout.addLayout(row)
            self.component_bars[name] = (bar, value)
        layout.addStretch(1)
        return card

    def _build_session_card(self) -> QFrame:
        card, layout = make_card(spacing=9)
        layout.addWidget(make_label("Session", "cardTitle"))
        layout.addSpacing(2)
        self.metric_session = MetricTile("Monitoring time", "00:00:00", BLUE)
        self.metric_good = MetricTile("Good posture", "0%", GOOD)
        self.metric_warning = MetricTile("Warning", "0%", WARN)
        self.metric_poor = MetricTile("Poor", "0%", POOR)
        self.metric_average = MetricTile("Average score", "0", VIOLET)
        for tile in (self.metric_session, self.metric_good, self.metric_warning,
                     self.metric_poor, self.metric_average):
            layout.addWidget(tile)
        layout.addStretch(1)
        return card

    def _build_durations_card(self) -> QFrame:
        card, layout = make_card(spacing=9)
        self.class_times = make_label("Detected Postures", "cardTitle")
        layout.addWidget(self.class_times)
        layout.addSpacing(2)
        self.duration_values = {}
        for label_text, key, accent in (
            ("Forward head", "Forward Head", BLUE),
            ("Slouching", "Slouching", VIOLET),
            ("Uneven shoulder", "Uneven Shoulder", CYAN),
            ("Leaning", "Leaning Left/Right", WARN),
        ):
            tile = MetricTile(label_text, "00:00", accent)
            layout.addWidget(tile)
            self.duration_values[key] = tile.value
        layout.addStretch(1)
        return card

    @staticmethod
    def _progress(name: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setObjectName(name)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        return bar

    # ------------------------------------------------------ one-screen fit --

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize and obj in (self._workspace_scroll, self._bottom_row):
            self._fit_previews()
        return super().eventFilter(obj, event)

    @staticmethod
    def _height_for(widget: QWidget, width: int) -> int:
        return widget.heightForWidth(width) if widget.hasHeightForWidth() else widget.sizeHint().height()

    def _fit_previews(self):
        """Give the camera previews whatever height is left so the whole dashboard fits without scrolling."""
        scroll = self._workspace_scroll
        margins = scroll.widget().layout().contentsMargins()
        inner = scroll.width() - margins.left() - margins.right()
        column = (inner - GAP) // 2
        current = self.front_view.height()
        camera_chrome = self._height_for(self._camera_cards[0], column) - current
        others = (
            margins.top() + margins.bottom() + 2 * GAP
            # toolbar text wraps more once a scrollbar narrows the page: estimate at that width so a few extra
            # pixels can never make a scrollbar appear and then overflow for real (feedback loop)
            + self._height_for(self._toolbar, inner - scroll.verticalScrollBar().sizeHint().width())
            + camera_chrome
            + self._height_for(self._bottom_row, inner)
        )
        # Up to 4:3 so 4:3 webcams fill the surface; 16:9 frames get slim letterbox bars.
        max_height = (column - 2 * CARD_PAD - 2) * 3 / 4
        height = int(max(PREVIEW_MIN, min(scroll.height() - others, max_height)))
        if abs(height - current) > 1:
            for view in (self.front_view, self.side_view):
                view.setFixedHeight(height)
        else:
            height = current
        # At most slightly wider than the camera frame at this height (capture aspect 4:3 x PREVIEW_WIDEN), centred
        # in its card. The frame itself keeps its size and proportions; image scaling is unchanged.
        width = max(self.front_view.minimumWidth(),
                    round(height * CONFIG.frame_width / CONFIG.frame_height * PREVIEW_WIDEN))
        for view in (self.front_view, self.side_view):
            if view.maximumWidth() != width:
                view.setMaximumWidth(width)

    # -------------------------------------------------------- state styling --

    def _set_camera_state(self, front_ok: bool, side_ok: bool):
        state = (front_ok, side_ok)
        if self._camera_state_cache == state:
            return
        self._camera_state_cache = state

        for label, ok in ((self.front_cam_status, front_ok), (self.side_cam_status, side_ok)):
            label.setText("●  LIVE" if ok else "●  OFFLINE")
            set_tone(label, "good" if ok else "poor")

        if front_ok and side_ok:
            self.connection_badge.setText("●  Both cameras live")
            set_tone(self.connection_badge, "good")
        elif front_ok or side_ok:
            self.connection_badge.setText("●  1 of 2 cameras live")
            set_tone(self.connection_badge, "warning")
        else:
            self.connection_badge.setText("●  Cameras offline")
            set_tone(self.connection_badge, "poor")

    def _set_risk(self, risk: str):
        if risk == self._risk_cache:
            return
        self._risk_cache = risk
        tone, text = RISK_UI.get(risk, RISK_UI["N/A"])
        self.risk_badge.setText(f"●  {text}")
        set_tone(self.risk_badge, tone)
        set_tone(self.status, tone)

    def _update_notifications(self, result=None):
        """Popup notifications for one frame (result None = no usable frames). Timing rules live in
        PopupNotifier; this only opens, refreshes and closes the single popup."""
        now = self._clock()
        risk = result.risk_level if result is not None else "N/A"
        action = self.notifier.update(now, *result_evidence(result))
        popup = self.posture_popup
        if action == "close":
            self._close_popup()
        elif action == "show":
            QApplication.beep()
            popup.set_state(risk, self.notifier.bad_seconds, main_problem(result))
            popup.show_over(self)
        elif self.notifier.visible == "posture" and risk in BAD_RISKS:
            popup.set_state(risk, self.notifier.bad_seconds, main_problem(result))  # current problem + duration
        if self.session_started and self.notifier.break_popup_allowed() and self.session.should_break_remind():
            QApplication.beep()
            self._offer_active_break()

    def _close_popup(self):
        if self.posture_popup.isVisible():
            self.posture_popup.close()
        self.notifier.closed()

    # -------------------------------------------------------------- logic --

    def _release_cameras(self):
        """Stop the capture threads and release both VideoCapture devices."""
        for cam in (self.front_cam, self.side_cam):
            if cam is not None:
                cam.release()  # joins the capture thread, then releases the device
        self.front_cam = self.side_cam = None

    def _cameras_running(self) -> bool:
        return any(cam is not None and cam.cap is not None for cam in (self.front_cam, self.side_cam))

    def _update_camera_buttons(self):
        self.btn_stop.setEnabled(self._cameras_running() and not self._camera_busy)
        self.btn_open.setEnabled(not self._camera_busy)

    def open_cameras(self):
        """Open both cameras in parallel on background threads so the UI stays responsive."""
        if self._camera_busy:
            return
        if self.front_index.value() == self.side_index.value():
            QMessageBox.warning(self, "Camera configuration", "Front and side camera must use different camera indices.")
            return
        self._camera_busy = True
        self._update_camera_buttons()
        self._release_cameras()  # never two captures (or threads) on the same device
        self._set_detection("Initializing Cameras…", "warning")
        front, side = CameraSource(self.front_index.value()), CameraSource(self.side_index.value())
        results = {}

        def opener(key, cam):
            try:
                results[key] = cam.open()
            except Exception:  # a broken driver must not leave the UI stuck in "busy"
                results[key] = False

        threads = [threading.Thread(target=opener, args=(k, c), name=f"open-{k}", daemon=True)
                   for k, c in (("front", front), ("side", side))]
        for t in threads:
            t.start()
        self._opening = (front, side, results, threads)
        QTimer.singleShot(30, self._finish_open)

    def _finish_open(self):
        if self._opening is None:
            return
        front, side, results, threads = self._opening
        if any(t.is_alive() for t in threads):
            QTimer.singleShot(30, self._finish_open)
            return
        self._opening = None
        try:
            ok_front, ok_side = results.get("front", False), results.get("side", False)
            for cam, ok in ((front, ok_front), (side, ok_side)):
                if not ok:
                    cam.release()
            self.front_cam = front if ok_front else None
            self.side_cam = side if ok_side else None
            self._set_camera_state(ok_front, ok_side)
            if not (ok_front or ok_side):
                self._set_detection("Camera Error · could not open the front and side camera", "poor")
                QMessageBox.warning(
                    self, "Camera error",
                    "Could not open the cameras. Try indices such as 0/1 or 0/2.\n\n"
                    "On macOS, also allow camera access for Terminal, VS Code, or Python in Privacy & Security > Camera.",
                )
                return

            # No calibration. The first start begins a session; later starts resume it (stopped time excluded).
            if not self.session_started:
                self.session.reset()
                self.session_started = True
            else:
                self.session.resume()
            self._monitored = False
            self.front_pose.reset()
            self.side_pose.reset()
            self.analyzer.reset_smoothing()
            self._set_detection("Checking Body Landmarks…", "warning")
            self.status.setText("Checking body landmarks…")
            if not (ok_front and ok_side):
                failed = "front" if not ok_front else "side"
                self._set_detection(f"Camera Error · could not open the {failed} camera — "
                                    f"monitoring with the other camera", "poor")
                QMessageBox.warning(self, "Camera error",
                                    f"Could not open the {failed} camera. Monitoring continues with the other camera.")
        finally:
            self._camera_busy = False
            self._update_camera_buttons()
            if self._break_after_open:
                self._break_after_open = False
                if self.front_cam is not None:
                    self.start_active_break()

    def stop_cameras(self):
        """Stop streaming and analysis, release both devices, pause (not reset) the session."""
        if self._camera_busy or not self._cameras_running():
            return
        self._camera_busy = True
        self._update_camera_buttons()
        try:
            if self.active_break is not None:
                self._finish_active_break()
            self._release_cameras()
            self.front_pose.reset()
            self.side_pose.reset()
            self.analyzer.reset_smoothing()
            if self.session_started:
                self.session.pause()
                self._save_history()  # same session row; Open Cameras resumes it
            self._close_popup()
            self.notifier.reset_episode()
            self._set_camera_state(False, False)
            for view in (self.front_view, self.side_view):
                view.clear()  # back to the camera placeholder
            self._show_stopped()
        finally:
            self._camera_busy = False
            self._update_camera_buttons()

    def _show_stopped(self):
        self.status.setText("Monitoring stopped")
        self.reason.setText("Cameras are off. Your session data is kept — press Open Cameras to resume.")
        self.duration.setText("Bad posture duration  ·  00:00:00")
        self._set_risk("N/A")
        self.score.setText("--")
        self.score_bar.setValue(0)
        set_tone(self.score, "neutral")
        set_tone(self.score_bar, "neutral")
        for bar, label in self.component_bars.values():
            bar.setValue(0)
            label.setText("--")
        self.detect_coverage.setValue(0)
        self._set_detection("Stopped · session paused — press Open Cameras to resume", "neutral")
        self._show_session_time()

    @staticmethod
    def _annotate(frame, title: str, lines, risk: str):
        """Small translucent readout in the top-left corner (no full-width band over the head)."""
        scale = frame.shape[1] / 1280
        title_size, text_size = 0.62 * scale, 0.5 * scale
        line_h = max(14, int(22 * scale))
        pad = max(6, int(10 * scale))
        widths = [cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_size, 2)[0][0] + int(26 * scale)]
        widths += [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)[0][0] for t in lines]
        x0, y0 = pad, pad
        box_w = min(max(widths) + 2 * pad, frame.shape[1] - x0)
        box_h = min(pad * 2 + line_h * (len(lines) + 1), frame.shape[0] - y0)
        roi = frame[y0:y0 + box_h, x0:x0 + box_w]
        shade = roi.copy()
        shade[:] = (20, 14, 9)
        frame[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(roi, 0.35, shade, 0.65, 0)

        color = {
            "GOOD": (102, 211, 153),
            "WARNING": (36, 191, 251),
            "POOR": (133, 113, 251),
            "N/A": (184, 196, 216),
        }.get(risk, (184, 196, 216))
        y = y0 + pad + int(line_h * 0.8)
        cv2.circle(frame, (x0 + pad + int(6 * scale), y - int(6 * scale)), max(3, int(6 * scale)), color, -1, cv2.LINE_AA)
        cv2.putText(frame, title, (x0 + pad + int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, title_size,
                    (245, 248, 252), max(1, int(2 * scale)), cv2.LINE_AA)
        for line in lines:
            y += line_h
            cv2.putText(frame, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (207, 218, 235), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def _wrap(text: str, width: int = 36) -> list:
        lines, line = [], ""
        for word in text.split():
            if line and len(line) + 1 + len(word) > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        return lines + ([line] if line else [])

    def _readout(self, camera: str, features, result) -> list:
        """Measured indicators vs. their reference limits ('!' = flagged), or camera guidance."""
        if not features.valid:
            return self._wrap(features.issue or "Waiting for a reliable pose")
        lines = []
        for metric in METRICS:
            if metric.camera != camera or metric.name not in result.metrics or metric.key == "slouch_index":
                continue
            if metric.mode == "negative":
                continue  # recline shares the trunk reading; shown with its sign below
            value, limit, _, flagged = result.metrics[metric.name]
            name = SHORT_NAMES.get(metric.name, metric.name)
            lines.append(f"{name:13s} {value:+5.1f}/{limit:.0f} deg{' !' if flagged else ''}")
        if not lines:  # posture not evaluated (the other camera is not ready): show raw readings
            lines = [f"{k.replace('_deg', '').replace('_', ' ')[:14]:14s} {v:+5.1f}"
                     for k, v in features.values.items() if k.endswith("_deg")]
        if features.held:
            lines.append("(holding last reading)")
        elif features.issue:
            lines += self._wrap(features.issue)[:2]
        return lines

    @staticmethod
    def _to_pixmap(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q)

    @staticmethod
    def _cover_box(fw: int, fh: int, widget: QWidget):
        """Centred (x, y, w, h) pixel rectangle of a fw x fh frame with the widget's aspect ratio (cover / fill)."""
        if widget.width() <= 0 or widget.height() <= 0:
            return 0, 0, fw, fh
        target = widget.width() / widget.height()
        if fw / fh > target:   # frame wider than the box: trim left/right
            cw = max(1, round(fh * target))
            return (fw - cw) // 2, 0, cw, fh
        ch = max(1, round(fw / target))  # frame taller than the box: trim top/bottom
        return 0, (fh - ch) // 2, fw, ch

    @classmethod
    def _cover_crop(cls, frame, widget: QWidget):
        """Centre-crop a frame to the widget's aspect ratio (cover / fill preview).

        Only a centred sub-rectangle is kept, at the original pixel scale, so the skeleton already drawn on the
        frame stays exactly on the body; the readout box is drawn afterwards, so it is never cut off.
        """
        x, y, w, h = cls._cover_box(frame.shape[1], frame.shape[0], widget)
        return np.ascontiguousarray(frame[y:y + h, x:x + w])

    @staticmethod
    def _crop_landmarks(lms, box, fw: int, fh: int):
        """Landmarks re-expressed in a cropped frame (normalized to the crop; may fall outside 0..1)."""
        if lms is None:
            return None
        x, y, w, h = box
        return {name: replace(p, x=(p.x * fw - x) / w, y=(p.y * fh - y) / h, width=w, height=h)
                for name, p in lms.items()}

    def _fitted_pixmap(self, frame, widget: QWidget) -> QPixmap:
        """Scale a BGR frame to fit a widget at the screen's pixel density (sharp on Retina)."""
        dpr = widget.devicePixelRatioF()
        pixmap = self._to_pixmap(frame).scaled(
            widget.size() * dpr,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        pixmap.setDevicePixelRatio(dpr)
        return pixmap

    def _show_frame(self, view: QLabel, frame):
        view.setPixmap(self._fitted_pixmap(frame, view))

    def update_frame(self):
        """Timer slot. Never lets an exception escape into Qt: recoverable errors are logged and the
        tracking state is reset; repeated failures stop the cameras safely and tell the user."""
        try:
            self._process_frame()
            self._frame_errors = 0
        except Exception as exc:  # deliberately broad here only: this is the top of the real-time loop
            self._on_frame_error(exc)

    def _log_error(self, message: str, exc: BaseException):
        signature = f"{type(exc).__name__}: {exc}"
        if signature != self._last_error:
            log.error(message, exc_info=exc)  # full traceback once per distinct error
            self._last_error = signature
        else:
            log.warning("%s (repeated): %s", message, signature)

    def _on_frame_error(self, exc: Exception):
        self._frame_errors += 1
        self._log_error(f"Frame processing failed ({self._frame_errors} in a row)", exc)
        try:
            # A half-updated frame may have left tracking state inconsistent: start it fresh.
            self.front_pose.reset()
            self.side_pose.reset()
            self.analyzer.reset_smoothing()
            if self._frame_errors >= FRAME_ERROR_LIMIT:
                log.error("Stopping cameras after %d consecutive processing errors", self._frame_errors)
                self._frame_errors = 0
                self.stop_cameras()
                self._set_detection("Stopped after repeated processing errors — see logs/posture_guard.log", "poor")
                QMessageBox.warning(
                    self, "Monitoring stopped",
                    "Posture monitoring stopped because of a repeated internal error.\n\n"
                    "Your session data is kept. Details were written to logs/posture_guard.log. "
                    "You can press Open Cameras to try again.")
            else:
                self._set_detection("Processing error — recovering…", "poor")
        except Exception as cleanup_exc:  # never fail inside the error handler
            log.critical("Error while recovering from a frame error", exc_info=cleanup_exc)

    def _detect(self, camera: str, engine: PoseEngine, frame):
        """Run pose detection for one camera. A failure (e.g. a MediaPipe error) affects only this camera
        for this frame; it is logged, and only persistent failures are escalated."""
        try:
            _, landmarks = engine.process(frame)
            self._detect_errors[camera] = 0
            return landmarks, None
        except Exception as exc:
            self._detect_errors[camera] += 1
            if self._detect_errors[camera] >= FRAME_ERROR_LIMIT:
                raise  # persistent: let the frame-level handler stop safely
            self._log_error(f"{camera.capitalize()} camera pose detection failed", exc)
            engine.reset()
            return None, f"{camera.capitalize()} camera: pose detection error — retrying"

    def _process_frame(self):
        self._show_session_time()
        if self.front_cam is None and self.side_cam is None:
            return  # stopped (or never opened): no capture, no pose processing

        okf, ff = self.front_cam.read() if self.front_cam else (False, None)
        oks, sf = self.side_cam.read() if self.side_cam else (False, None)
        okf, oks = bool(okf) and ff is not None, bool(oks) and sf is not None
        if self.active_break is not None:
            # Same camera reads and pose engine as monitoring; posture checks and popups pause.
            self.session.tick("break")  # game time is not posture monitoring
            self._set_camera_state(okf, oks)
            self._update_active_break(ff if okf else None)
            return
        self._set_camera_state(okf, oks)
        if not (okf or oks):
            self.session.tick("insufficient")
            self.status.setText("Camera frame unavailable")
            self._set_risk("N/A")
            self._update_notifications(None)  # no data: pauses the warning timer, never warns
            self._set_detection("Camera Error · no frames from the front and side camera", "poor")
            return
        fresh = (okf and getattr(self.front_cam, "new_frame", True)) or (oks and getattr(self.side_cam, "new_frame", True))
        if not fresh:
            return  # nothing new from either camera since the last tick: don't re-run pose on old frames
        # Pose runs on the un-mirrored frames so MediaPipe's left/right labels are anatomically correct;
        # each camera has its own pose engine, landmark filter and feature smoothing. A camera without
        # frames (or with a detection error) is reported and the other camera keeps being evaluated.
        flm = slm = None
        if okf:
            flm, error = self._detect("front", self.front_pose, ff)
            front_features = FeatureSet(issue=error) if error else self.analyzer.extract_front(flm)
        else:
            front_features = FeatureSet(issue="Camera Error: no frames from the front camera")
        if oks:
            slm, error = self._detect("side", self.side_pose, sf)
            side_features = FeatureSet(issue=error) if error else self.analyzer.extract_side(slm)
        else:
            side_features = FeatureSet(issue="Camera Error: no frames from the side camera")

        rough = self.analyzer.classify(front_features, side_features, self.session.persistent_bad_seconds)
        rough_bad = bool(rough.labels) and rough.labels != ["Good Posture"]
        self.session.preclassify_tick(rough_bad)
        result = self.analyzer.classify(front_features, side_features, self.session.persistent_bad_seconds)

        # Session statistics only count valid (both-camera) monitoring; partial / insufficient time is
        # tracked separately and never counted as good or bad posture.
        state = "insufficient" if result.risk_level == "N/A" else ("partial" if result.partial else "monitoring")
        self.session.tick(state, result.labels, result.risk_level, result.score)

        if okf:
            ff = self.front_pose.draw_front(ff, flm, result.bad_regions_front)
            if CONFIG.mirror_front_view:
                ff = cv2.flip(ff, 1)
            ff = self._cover_crop(ff, self.front_view)
            ff = self._annotate(ff, "FRONT CAMERA", self._readout("front", front_features, result), result.risk_level)
            self._show_frame(self.front_view, ff)
        elif not self.front_view.pixmap().isNull():
            self.front_view.clear()
        if oks:
            sf = self.side_pose.draw_side(sf, slm, self.analyzer.side_choice, result.bad_regions_side)
            if CONFIG.mirror_side_view:
                sf = cv2.flip(sf, 1)
            sf = self._cover_crop(sf, self.side_view)
            sf = self._annotate(sf, "SIDE CAMERA", self._readout("side", side_features, result), result.risk_level)
            self._show_frame(self.side_view, sf)
        elif not self.side_view.pixmap().isNull():
            self.side_view.clear()

        self._update_dashboard(result)
        self._update_detection_status(result)
        self._update_notifications(result)

    def _show_session_time(self):
        """'Monitoring time' = valid posture analysis only; the tooltip explains where the rest went."""
        s = self.session
        if not self.session_started:
            self.metric_session.value.setText("00:00:00")
            return
        self.metric_session.value.setText(format_seconds(s.monitored_seconds))
        self.metric_session.setToolTip(
            f"Valid monitoring: {format_seconds(s.monitored_seconds)}\n"
            f"Session (cameras on): {format_seconds(s.elapsed)}\n"
            f"One camera only: {format_seconds(s.partial_seconds)}\n"
            f"Insufficient data: {format_seconds(s.insufficient_seconds)}\n"
            f"Active Break: {format_seconds(s.break_seconds)}")

    def _update_dashboard(self, r):
        self.status.setText(r.status)
        self._set_risk(r.risk_level)
        self.duration.setText(f"Bad posture duration  ·  {format_seconds(self.session.persistent_bad_seconds)}")
        self.reason.setText(" • ".join(r.reasons) if r.reasons else "-")

        if r.score is None:  # insufficient data: never shown as a perfect (or any) score
            self.score.setText("--")
            self.score_bar.setValue(0)
            set_tone(self.score, "neutral")
            set_tone(self.score_bar, "neutral")
        else:
            # The analyzer keeps the score inside its classification band, and the colour comes from the
            # classification itself, so number, colour and Good/Warning/Poor status always agree.
            score = int(r.score)
            tone = RISK_UI.get(r.risk_level, RISK_UI["N/A"])[0]
            self.score.setText(str(score))
            self.score_bar.setValue(score)
            set_tone(self.score, tone)
            set_tone(self.score_bar, tone)
        for name, (bar, label) in self.component_bars.items():
            value = r.components.get(name)
            bar.setValue(0 if value is None else int(value))
            label.setText("--" if value is None else str(int(value)))

        p = self.session.percentages()
        self._show_session_time()
        self.metric_good.value.setText(f"{p['Good Posture']}%")
        self.metric_warning.value.setText(f"{p['Warning']}%")
        self.metric_poor.value.setText(f"{p['Poor']}%")
        self.metric_average.value.setText(str(self.session.average_score))

        d = self.session.durations
        for key, label in self.duration_values.items():
            label.setText(format_seconds(d[key], True))

    # ------------------------------------------------------- active break --

    def start_active_break(self):
        if self.active_break is not None or self._camera_busy:
            return
        if self.front_cam is None or self.front_cam.cap is None:  # the game needs the front camera
            self._break_after_open = True  # cameras open in the background; start once they are ready
            self.open_cameras()
            return
        self._close_popup()
        self.notifier.reset_episode()  # no posture popups during the game; a new episode starts afterwards
        self.active_break = ActiveBreak(self._clock())
        self.btn_break.setEnabled(False)
        self._stack.setCurrentWidget(self.break_view)

    def _offer_active_break(self):
        """Sitting too long: ask (non-blocking) whether to start the game."""
        self.posture_popup.set_state("BREAK")
        self.posture_popup.show_over(self)
        self.notifier.opened("break", self._clock())

    def _update_active_break(self, frame):
        ctrl = self.active_break
        now = self._clock()
        pose, pixmap = None, None
        if frame is not None and ctrl.state != "results":
            _, landmarks = self.front_pose.process(frame)
            # The game is always shown as a mirror so "left" on screen is the player's left.
            frame = cv2.flip(self.front_pose.draw_front(frame, landmarks), 1)
            # Cover the game canvas (no side bars): show a centred crop of the mirrored frame. The game's body
            # coordinates are those of the displayed (cropped) frame, so hands, coins and bombs line up with
            # what the player sees; the crop is a uniform scale + shift, so the pose rules are unaffected.
            fh, fw = frame.shape[:2]
            box = self._cover_box(fw, fh, self.break_view.canvas)
            shown = self._crop_landmarks(mirror_landmarks(landmarks), box, fw, fh)
            pose = BodyPose.from_landmarks(shown, aspect=box[2] / box[3])
            pixmap = self._fitted_pixmap(np.ascontiguousarray(frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]),
                                         self.break_view.canvas)
        ctrl.update(now, pose)
        if ctrl.finished(now):
            self._finish_active_break()
            return
        self.break_view.update_view(ctrl, now, pixmap, pose)

    def _finish_active_break(self):
        if self.active_break is None:
            return
        self.active_break = None
        self.btn_break.setEnabled(True)
        self._stack.setCurrentIndex(0)
        self.session.mark_break_taken()
        self.analyzer.reset_smoothing()
        self.notifier.reset_episode()

    def _set_detection(self, text: str, tone: str):
        self.detect_status.setText(text)
        set_tone(self.detect_status, tone)

    def _update_detection_status(self, r):
        coverage = round(100 * len(r.metrics) / len(METRICS))
        self.detect_coverage.setValue(coverage)
        hint = r.guidance[0] if r.guidance else ""
        if r.risk_level == "N/A":
            state = "Insufficient Data" if self._monitored else "Checking Body Landmarks"
            self._set_detection(f"{state} · {hint or 'waiting for a reliable pose'}", "warning")
            return
        self._monitored = True
        if r.partial:
            self._set_detection(f"Partial Monitoring · {hint}", "warning")
        elif hint:
            self._set_detection(f"Auto Detection Active · Monitoring Posture · {hint}", "warning")
        else:
            self._set_detection("Auto Detection Active · Monitoring Posture", "good")

    # ------------------------------------------------------------ history --

    def _save_history(self, final: bool = False):
        """Upsert the current session (background write). Sessions are keyed by id, so pausing/resuming and
        repeated saves update one row."""
        if not self.session_started:
            return
        self.history.save(self.session.snapshot(), final=final)
        if not final and self._stack.currentWidget() is self.history_view:
            self.history.flush(0.5)
            self.history_view.refresh()

    def show_history(self):
        self._save_history()
        self.history.flush(1.0)  # include the latest values of the current session
        self.history_view.refresh(current_id=self.session.session_id if self.session_started else None)
        self._stack.setCurrentWidget(self.history_view)

    def close_history(self):
        if self._stack.currentWidget() is self.history_view:
            self._stack.setCurrentIndex(0)

    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self._opening is not None:
                front, side, _, threads = self._opening
                for t in threads:
                    t.join(timeout=5.0)
                front.release()
                side.release()
                self._opening = None
            self._release_cameras()
            self.front_pose.close()
            self.side_pose.close()
            self.history_timer.stop()
            self._save_history(final=True)
            self.history.close(timeout=3.0)  # waits for the final write
        finally:
            event.accept()


def _light_palette() -> QPalette:
    """Pin a light palette so macOS/Windows dark mode cannot invert dialogs and popups."""
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: "#F8FAFC",
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: "#FFFFFF",
        QPalette.ColorRole.AlternateBase: "#F1F5F9",
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: "#FFFFFF",
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: BLUE,
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.ToolTipBase: TEXT,
        QPalette.ColorRole.ToolTipText: "#F8FAFC",
        QPalette.ColorRole.PlaceholderText: MUTED,
        QPalette.ColorRole.Link: BLUE,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    return pal


def _app_font() -> QFont:
    font = QApplication.font()
    families = set(QFontDatabase.families())
    for family in ("Inter", "SF Pro Text", "Segoe UI Variable Text", "Segoe UI"):
        if family in families:
            font = QFont(family)
            break
    font.setPixelSize(13)
    return font


def _unhide_qt_plugins():
    """Clear the macOS 'hidden' file flag on Qt's plugins.

    On this machine something re-applies UF_HIDDEN to everything inside dot-folders such as
    .venv, and Qt silently skips hidden plugin files, failing with
    'Could not find the Qt platform plugin "cocoa"'.
    """
    if sys.platform != "darwin":
        return
    plugins = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    for dirpath, dirnames, filenames in os.walk(plugins):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            try:
                flags = os.lstat(path).st_flags
                if flags & stat.UF_HIDDEN:
                    os.chflags(path, flags & ~stat.UF_HIDDEN, follow_symlinks=False)
            except OSError:
                pass


def _setup_logging():
    """Diagnostics go to stderr and to logs/posture_guard.log (rotated)."""
    from logging.handlers import RotatingFileHandler
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(logging.WARNING)
    log.addHandler(stream)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(os.path.join(LOG_DIR, "posture_guard.log"),
                                           maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    except OSError as exc:
        log.warning("File logging unavailable (%s); logging to stderr only", exc)

    def excepthook(exc_type, exc, tb):
        # With a custom hook PyQt6 reports exceptions raised in Qt slots instead of aborting the app.
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = excepthook


def run():
    _unhide_qt_plugins()
    _setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Posture Guard")
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setFont(_app_font())
    app.setStyleSheet(APP_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
