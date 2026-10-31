"""Qt screens for the Active Break game: play screen (HUD + camera canvas + instructions) and results."""
from __future__ import annotations

import logging
import math
import os
from collections import Counter
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .active_break import PHASE_NAMES, ActiveBreak
from .body_pose import BodyPose
from .coin_game import BOMB_DANGER_FROM, BOMB_RADIUS, COIN_RADIUS, HAND_RADIUS, CoinGame
from config import CONFIG
from .stretch_game import StretchGame
from ui_kit import BLUE, COIN, CYAN, GOOD, POOR, PREVIEW_BG, TEXT, VIOLET, format_seconds, make_label, set_tone

# Diagnostic for real-camera checks (off by default): POSTURE_GUARD_DEBUG_HANDS=1 marks each wrist and the source of
# each hand marker (K = index/pinky knuckles, F = estimated along the forearm) and logs how often each source is used.
DEBUG_HANDS = os.environ.get("POSTURE_GUARD_DEBUG_HANDS") == "1"
HAND_LOG_SECONDS = 5.0
log = logging.getLogger("posture_guard")

GAME_QSS = """
QWidget#breakView, QWidget#breakPage { background: #FFFFFF; }
QFrame#gameCard { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; }
QLabel#modePill { border-radius: 13px; padding: 6px 14px; font-size: 13px; font-weight: 700; }
QLabel#modePill[tone="stretch"] { background: #F3E8FF; color: #7C3AED; }
QLabel#modePill[tone="coin"] { background: #EFF6FF; color: #2563EB; }
QLabel#modePill[tone="transition"] { background: #DCFCE7; color: #16A34A; }
QLabel#roundLabel { color: #64748B; font-size: 12px; font-weight: 600; }
QLabel#hudCaption { color: #94A3B8; font-size: 11px; font-weight: 600; }
QLabel#hudValue { color: #0F172A; font-size: 20px; font-weight: 700; }
QLabel#hudScore { color: #7C3AED; font-size: 24px; font-weight: 800; }
QLabel#panelCaption { color: #94A3B8; font-size: 11px; font-weight: 700; }
QLabel#instructionTitle { color: #0F172A; font-size: 22px; font-weight: 800; }
QLabel#instructionTitle[tone="danger"] { color: #DC2626; }
QLabel#instructionTitle[tone="done"] { color: #16A34A; }
QLabel#instructionHint { color: #64748B; font-size: 13px; }
QLabel#statName { color: #64748B; font-size: 13px; }
QLabel#statNum { color: #0F172A; font-size: 15px; font-weight: 700; }
QProgressBar#holdBar {
    background: #EDE9FE; border: none; border-radius: 5px;
    min-height: 10px; max-height: 10px; color: transparent;
}
QProgressBar#holdBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C3AED, stop:1 #2563EB);
}
QFrame#resultsCard { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 22px; }
QLabel#resultsTitle { color: #0F172A; font-size: 26px; font-weight: 800; }
QLabel#resultsSubtitle { color: #64748B; font-size: 14px; }
QLabel#resultsScoreCaption { color: #94A3B8; font-size: 12px; font-weight: 700; }
QLabel#resultsScore { color: #7C3AED; font-size: 56px; font-weight: 800; }
QFrame#resultTile { background: #F8FAFC; border: 1px solid #EEF2F7; border-radius: 14px; }
QLabel#resultValue { color: #0F172A; font-size: 26px; font-weight: 800; }
QLabel#resultName { color: #64748B; font-size: 12px; font-weight: 600; }
QLabel#resultsNote { color: #94A3B8; font-size: 12px; }
QPushButton#returnButton {
    background: #2563EB; border-color: #2563EB; color: #FFFFFF;
    min-height: 44px; padding: 0 24px; font-size: 14px; font-weight: 700;
}
QPushButton#returnButton:hover { background: #1D4ED8; border-color: #1D4ED8; }
QPushButton#returnButton:pressed { background: #1E40AF; border-color: #1E40AF; }
"""

MODE_TONES = {"stretch": "stretch", "coin": "coin", "transition": "transition"}
STRETCH_STATUS = {"detected": "POSE DETECTED — HOLD IT", "adjust": "ADJUST YOUR POSITION",
                  "not_visible": "BODY NOT FULLY VISIBLE"}


def _card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("gameCard")
    return frame


def _font(px: int, weight: QFont.Weight = QFont.Weight.Bold, base: Optional[QFont] = None) -> QFont:
    font = QFont(base) if base is not None else QFont()
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


# ---------------------------------------------------------------------------
# Stick-figure guide for the current stretch
# ---------------------------------------------------------------------------

class PoseGuide(QWidget):
    POSES = {
        # hands (left, right), tilt in degrees (negative = lean to the screen-left)
        "raise_arms": (((0.36, 0.06), (0.64, 0.06)), 0),
        "arms_wide": (((0.06, 0.40), (0.94, 0.40)), 0),
        "lean_left": (((0.38, 0.66), (0.62, 0.66)), -16),
        "lean_right": (((0.38, 0.66), (0.62, 0.66)), 16),
    }

    def __init__(self):
        super().__init__()
        self.kind = "raise_arms"
        self.setFixedHeight(150)

    def set_kind(self, kind: str):
        if kind != self.kind:
            self.kind = kind
            self.update()

    def paintEvent(self, event):
        hands, tilt = self.POSES.get(self.kind, self.POSES["raise_arms"])
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        bg = QPainterPath()
        bg.addRoundedRect(rect, 14, 14)
        p.fillPath(bg, QColor("#F5F3FF"))
        size = min(rect.width(), rect.height()) * 0.9
        ox, oy = rect.center().x() - size / 2, rect.top() + (rect.height() - size) / 2

        def pt(x, y):
            return QPointF(ox + x * size, oy + y * size)

        hip = pt(0.5, 0.80)
        p.translate(hip)
        p.rotate(tilt)
        p.translate(-hip)
        pen = QPen(QColor(VIOLET), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        neck, left_sh, right_sh = pt(0.5, 0.36), pt(0.38, 0.38), pt(0.62, 0.38)
        p.drawLine(neck, hip)
        p.drawLine(left_sh, right_sh)
        p.drawLine(left_sh, pt(*hands[0]))
        p.drawLine(right_sh, pt(*hands[1]))
        p.drawLine(hip, pt(0.40, 0.98))
        p.drawLine(hip, pt(0.60, 0.98))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(VIOLET))
        p.drawEllipse(pt(0.5, 0.24), size * 0.075, size * 0.075)


# ---------------------------------------------------------------------------
# Camera canvas with game overlays
# ---------------------------------------------------------------------------

class GameCanvas(QWidget):
    RADIUS = 14

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 220)
        self._pixmap: Optional[QPixmap] = None
        self._ctrl: Optional[ActiveBreak] = None
        self._pose: Optional[BodyPose] = None
        self._now = 0.0
        self._hand_stats: Counter = Counter()          # DEBUG_HANDS only
        self._hand_stats_since: Optional[float] = None

    def sizeHint(self) -> QSize:
        return QSize(800, 450)

    def set_scene(self, ctrl: ActiveBreak, now: float, pixmap: Optional[QPixmap], pose: Optional[BodyPose]):
        self._ctrl, self._now, self._pose = ctrl, now, pose
        if pixmap is not None:
            self._pixmap = pixmap
        if DEBUG_HANDS and ctrl.playing and pixmap is not None:
            self._log_hand_sources(now, pose)
        self.update()

    def _log_hand_sources(self, now: float, pose: Optional[BodyPose]):
        """Share of hand markers from knuckle landmarks / forearm estimates / missing, logged every few seconds."""
        stats = self._hand_stats
        stats.update(["no body"] if pose is None else list(pose.hand_sources) + ["missing"] * (2 - len(pose.hands)))
        if self._hand_stats_since is None:
            self._hand_stats_since = now
        if now - self._hand_stats_since >= HAND_LOG_SECONDS:
            total = sum(stats.values()) or 1
            log.info("Hand debug (%.0f s): %s", now - self._hand_stats_since,
                     ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in stats.most_common()))
            stats.clear()
            self._hand_stats_since = now

    # ---- mapping helpers
    def _image_rect(self) -> QRectF:
        """Where the displayed frame (and the game's 0..1 coordinates) lies: cover / fill, centred. The frame
        arrives already cropped to the canvas shape; any small mismatch is clipped, never shown as bars."""
        rect = QRectF(self.rect())
        if self._pixmap is None or self._pixmap.isNull():
            return rect
        size = self._pixmap.deviceIndependentSize().scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        return QRectF(rect.center().x() - size.width() / 2, rect.center().y() - size.height() / 2,
                      size.width(), size.height())

    @staticmethod
    def _map(ir: QRectF, x: float, y: float) -> QPointF:
        return QPointF(ir.left() + x * ir.width(), ir.top() + y * ir.height())

    def _pill(self, p: QPainter, center: QPointF, text: str, fg: str, bg: str, px: int = 16, border: Optional[str] = None):
        p.setFont(_font(px, QFont.Weight.Bold, self.font()))
        metrics = p.fontMetrics()
        w, h = metrics.horizontalAdvance(text) + 32, metrics.height() + 16
        box = QRectF(center.x() - w / 2, center.y() - h / 2, w, h)
        path = QPainterPath()
        path.addRoundedRect(box, h / 2, h / 2)
        p.fillPath(path, QColor(bg))
        if border:
            p.setPen(QPen(QColor(border), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.setPen(QColor(fg))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        return box

    def _outlined_text(self, p: QPainter, center: QPointF, text: str, px: int, color: QColor):
        font = _font(px, QFont.Weight.Black, self.font())
        path = QPainterPath()
        path.addText(0, 0, font, text)
        box = path.boundingRect()
        path.translate(center.x() - box.width() / 2 - box.left(), center.y() + box.height() / 2 - box.bottom())
        p.setPen(QPen(QColor(255, 255, 255, color.alpha()), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPath(path)

    # ---- painting
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, self.RADIUS, self.RADIUS)
        p.fillPath(frame, QColor(PREVIEW_BG))
        p.setClipPath(frame)
        ir = self._image_rect()
        if self._pixmap is not None and not self._pixmap.isNull():
            p.drawPixmap(ir, self._pixmap, QRectF(self._pixmap.rect()))
        else:
            p.setPen(QColor("#CBD5E1"))
            p.setFont(_font(14, QFont.Weight.DemiBold, self.font()))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Waiting for the front camera…")

        ctrl = self._ctrl
        if ctrl is None:
            return
        if self._pose is not None and ctrl.playing:
            self._paint_hands(p, ir)
        if ctrl.state == "stretch":
            self._paint_stretch(p, ir, ctrl.game)
        elif ctrl.state == "coin":
            self._paint_coin_game(p, ir, ctrl.game)
        elif ctrl.state == "transition":
            self._paint_transition(p, ir, ctrl)
        if ctrl.playing and self._pose is None:
            self._pill(p, QPointF(rect.center().x(), rect.bottom() - 34),
                       "Move into view — show your head and shoulders", "#92400E", "#FEF3C7", 14)

    def _paint_hands(self, p: QPainter, ir: QRectF):
        r = HAND_RADIUS * ir.height() * 0.55
        for hand in self._pose.hands:
            c = self._map(ir, *hand)
            p.setPen(QPen(QColor("#FFFFFF"), 3))
            p.setBrush(QColor(CYAN))
            p.drawEllipse(c, r, r)
        if DEBUG_HANDS:   # wrist dots, wrist -> hand lines and the source of each marker
            for wrist in self._pose.wrists:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#94A3B8"))
                p.drawEllipse(self._map(ir, *wrist), 6, 6)
            p.setFont(_font(14, QFont.Weight.Black, self.font()))
            for hand, source in zip(self._pose.hands, self._pose.hand_sources):
                c = self._map(ir, *hand)
                wrist = min(self._pose.wrists, key=lambda w: math.dist(w, hand), default=None)
                if wrist is not None:
                    p.setPen(QPen(QColor("#94A3B8"), 2))
                    p.drawLine(self._map(ir, *wrist), c)
                p.setPen(QColor("#0F172A"))
                p.drawText(c + QPointF(r + 4, -r), "K" if source == "knuckles" else "F")

    def _paint_stretch(self, p: QPainter, ir: QRectF, game: StretchGame):
        now = self._now
        matching = game.matching and not game.celebrating(now)
        box = self._pill(p, QPointF(ir.center().x(), ir.top() + 34), game.exercise.name,
                         TEXT, "#FFFFFF", 18, GOOD if matching else None)
        bar = QRectF(box.left() + 16, box.bottom() + 8, box.width() - 32, 8)
        track = QPainterPath()
        track.addRoundedRect(bar, 4, 4)
        p.fillPath(track, QColor(255, 255, 255, 140))
        progress = game.hold_progress(now)
        if progress > 0:
            fill = QPainterPath()
            fill.addRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * progress, bar.height()), 4, 4)
            p.fillPath(fill, QColor(GOOD if matching else VIOLET))
        if game.celebrating(now):
            color = QColor(GOOD if game.message_good else "#F59E0B")
            self._outlined_text(p, ir.center(), game.message, 64 if game.message_good else 40, color)

    def _paint_coin_game(self, p: QPainter, ir: QRectF, game: CoinGame):
        now = self._now
        h = ir.height()
        if game.coin is not None:
            c = self._map(ir, game.coin.x, game.coin.y)
            r = COIN_RADIUS * h
            pulse = 1 + 0.06 * math.sin(now * 6)
            glow = QRadialGradient(c, r * 1.9)
            glow.setColorAt(0, QColor(251, 191, 36, 120))
            glow.setColorAt(1, QColor(251, 191, 36, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(c, r * 1.9, r * 1.9)
            p.setBrush(QColor("#F59E0B"))
            p.drawEllipse(c, r * pulse, r * pulse)
            p.setBrush(QColor(COIN))
            p.drawEllipse(c, r * 0.8 * pulse, r * 0.8 * pulse)
            p.setPen(QPen(QColor("#B45309"), max(2.0, r * 0.12)))
            p.setFont(_font(int(r * 0.95), QFont.Weight.Black, self.font()))
            p.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, "$")

        bomb = game.bomb
        if bomb is not None:
            pos = self._map(ir, *bomb.position(now))
            r = BOMB_RADIUS * h
            danger = bomb.progress(now) >= BOMB_DANGER_FROM
            if danger:
                ring = r * (1.5 + 0.25 * math.sin(now * 18))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(239, 68, 68, 70))
                p.drawEllipse(pos, ring, ring)
            p.setPen(QPen(QColor("#7F1D1D"), max(2.0, r * 0.12)))
            p.drawLine(pos + QPointF(r * 0.5, -r * 0.6), pos + QPointF(r * 0.95, -r * 1.15))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(COIN))
            spark = r * (0.22 + 0.08 * math.sin(now * 25))
            p.drawEllipse(pos + QPointF(r * 0.95, -r * 1.15), spark, spark)
            p.setPen(QPen(QColor("#991B1B"), max(2.0, r * 0.1)))
            p.setBrush(QColor(POOR))
            p.drawEllipse(pos, r, r)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 150))
            p.drawEllipse(pos + QPointF(-r * 0.35, -r * 0.35), r * 0.22, r * 0.22)
            self._pill(p, QPointF(ir.center().x(), ir.top() + 34), game.instruction(),
                       "#FFFFFF", POOR if danger else "#F87171", 18)

        for fb in game.feedback:
            age = now - fb.at
            alpha = max(0, int(255 * (1 - age / 0.9)))
            c = self._map(ir, fb.x, fb.y) + QPointF(0, -age * 70 - 20)
            color = QColor(GOOD if fb.good else POOR)
            color.setAlpha(alpha)
            self._outlined_text(p, c, fb.text, 30, color)

    def _paint_transition(self, p: QPainter, ir: QRectF, ctrl: ActiveBreak):
        p.fillRect(QRectF(self.rect()), QColor(255, 255, 255, 228))
        c = QRectF(self.rect()).center()
        p.setPen(QColor(TEXT))
        p.setFont(_font(32, QFont.Weight.Black, self.font()))
        p.drawText(QRectF(0, c.y() - 90, self.width(), 44), Qt.AlignmentFlag.AlignCenter, ctrl.transition_title)
        p.setPen(QColor(VIOLET))
        p.setFont(_font(19, QFont.Weight.Bold, self.font()))
        p.drawText(QRectF(0, c.y() - 40, self.width(), 30), Qt.AlignmentFlag.AlignCenter, ctrl.transition_subtitle)
        remaining = max(1, math.ceil(ctrl.transition_remaining(self._now)))
        center = QPointF(c.x(), c.y() + 40)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#EFF6FF"))
        p.drawEllipse(center, 30, 30)
        p.setPen(QColor(BLUE))
        p.setFont(_font(24, QFont.Weight.Black, self.font()))
        p.drawText(QRectF(center.x() - 30, center.y() - 30, 60, 60), Qt.AlignmentFlag.AlignCenter, str(remaining))


# ---------------------------------------------------------------------------
# Whole Active Break screen
# ---------------------------------------------------------------------------

class ActiveBreakView(QWidget):
    end_requested = pyqtSignal()
    return_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("breakView")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(GAME_QSS)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._pages = QStackedWidget()
        outer.addWidget(self._pages)
        self._pages.addWidget(self._build_play_page())
        self._pages.addWidget(self._build_results_page())

    # ------------------------------------------------------------------ build

    def _build_play_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("breakPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        hud = _card()
        row = QHBoxLayout(hud)
        row.setContentsMargins(16, 12, 16, 12)
        row.setSpacing(18)
        self.mode_pill = make_label("Get Ready", "modePill")
        set_tone(self.mode_pill, "transition")
        row.addWidget(self.mode_pill)
        self.round_label = make_label("", "roundLabel")
        row.addWidget(self.round_label)
        row.addStretch(1)
        self.phase_time = self._hud_stat(row, "TIME LEFT", "0:30")
        self.total_time = self._hud_stat(row, "TOTAL", "2:00")
        self.score_value = self._hud_stat(row, "SCORE", "0", "hudScore")
        end = QPushButton("End Break")
        end.setObjectName("outlineButton")
        end.setCursor(Qt.CursorShape.PointingHandCursor)
        end.clicked.connect(self.end_requested.emit)
        row.addWidget(end)
        root.addWidget(hud)

        body = QHBoxLayout()
        body.setSpacing(14)
        canvas_card = _card()
        canvas_box = QVBoxLayout(canvas_card)
        canvas_box.setContentsMargins(12, 12, 12, 12)
        self.canvas = GameCanvas()
        canvas_box.addWidget(self.canvas)
        body.addWidget(canvas_card, 1)

        panel = _card()
        panel.setFixedWidth(300)
        side = QVBoxLayout(panel)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(10)
        side.addWidget(make_label("CURRENT INSTRUCTION", "panelCaption"))
        self.instruction = make_label("Get ready", "instructionTitle", wrap=True)
        self.hint = make_label("", "instructionHint", wrap=True)
        side.addWidget(self.instruction)
        side.addWidget(self.hint)
        side.addSpacing(4)
        self.guide = PoseGuide()
        side.addWidget(self.guide)
        self.progress_caption = make_label("HOLD", "panelCaption")
        side.addWidget(self.progress_caption)
        self.progress = QProgressBar()
        self.progress.setObjectName("holdBar")
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        side.addWidget(self.progress)
        side.addSpacing(6)
        self._stats = {}
        for key, name in (("stretches", "Stretches completed"), ("coins", "Coins collected"),
                          ("avoided", "Bombs avoided"), ("hits", "Bomb hits")):
            line = QHBoxLayout()
            line.addWidget(make_label(name, "statName"), 1)
            value = make_label("0", "statNum")
            line.addWidget(value)
            holder = QWidget()
            holder.setLayout(line)
            line.setContentsMargins(0, 0, 0, 0)
            side.addWidget(holder)
            self._stats[key] = (holder, value)
        side.addStretch(1)
        body.addWidget(panel)
        root.addLayout(body, 1)
        return page

    @staticmethod
    def _hud_stat(row: QHBoxLayout, caption: str, value: str, value_name: str = "hudValue"):
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(make_label(caption, "hudCaption"), 0, Qt.AlignmentFlag.AlignRight)
        label = make_label(value, value_name)
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        col.addWidget(label)
        row.addLayout(col)
        return label

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("breakPage")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.addStretch(1)
        card = QFrame()
        card.setObjectName("resultsCard")
        card.setMaximumWidth(640)
        box = QVBoxLayout(card)
        box.setContentsMargins(36, 32, 36, 30)
        box.setSpacing(10)
        for widget in (make_label("Active Break Complete!", "resultsTitle"),
                       make_label("Nice work moving your body. Here's how you did:", "resultsSubtitle"),
                       make_label("TOTAL SCORE", "resultsScoreCaption")):
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(widget)
        self.results_score = make_label("0", "resultsScore")
        self.results_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.results_score)
        grid = QGridLayout()
        grid.setSpacing(12)
        self._result_values = {}
        for i, (key, name, color) in enumerate((("stretches", "Stretches Completed", VIOLET),
                                                ("coins", "Coins Collected", "#D97706"),
                                                ("avoided", "Bombs Avoided", GOOD),
                                                ("hits", "Bomb Hits", POOR))):
            tile = QFrame()
            tile.setObjectName("resultTile")
            tile_box = QVBoxLayout(tile)
            tile_box.setContentsMargins(16, 12, 16, 12)
            tile_box.setSpacing(2)
            value = make_label("0", "resultValue")
            value.setStyleSheet(f"color:{color};")
            tile_box.addWidget(value)
            tile_box.addWidget(make_label(name, "resultName"))
            grid.addWidget(tile, i // 2, i % 2)
            self._result_values[key] = value
        box.addSpacing(6)
        box.addLayout(grid)
        box.addSpacing(10)
        self.return_button = QPushButton("Return to Monitoring")
        self.return_button.setObjectName("returnButton")
        self.return_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.return_button.clicked.connect(self.return_requested.emit)
        box.addWidget(self.return_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.results_note = make_label("", "resultsNote")
        self.results_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.results_note)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card, 4)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)
        return page

    # ------------------------------------------------------------------ update

    def update_view(self, ctrl: ActiveBreak, now: float, pixmap: Optional[QPixmap], pose: Optional[BodyPose]):
        if ctrl.state == "results":
            self._show_results(ctrl, now)
            return
        self._pages.setCurrentIndex(0)

        state = ctrl.state
        set_tone(self.mode_pill, MODE_TONES[state])
        if state == "transition":
            self.mode_pill.setText("Get Ready" if ctrl.phase_index < 0 else "Up Next")
            self.phase_time.setText(format_seconds(ctrl.transition_remaining(now), short=True))
        else:
            self.mode_pill.setText(PHASE_NAMES[state])
            self.phase_time.setText(format_seconds(math.ceil(ctrl.phase_remaining(now)), short=True))
        self.round_label.setText(ctrl.round_label)
        self.total_time.setText(format_seconds(math.ceil(ctrl.total_remaining(now)), short=True))
        self.score_value.setText(str(ctrl.score))

        game = ctrl.game
        stretch = isinstance(game, StretchGame)
        coin = isinstance(game, CoinGame)
        self.guide.setVisible(stretch)
        self.progress_caption.setVisible(state != "transition")
        self.progress.setVisible(state != "transition")
        for key, (holder, value) in self._stats.items():
            holder.setVisible((stretch and key == "stretches") or (coin and key != "stretches"))
        self._stats["stretches"][1].setText(str(ctrl.stretches_completed))
        self._stats["coins"][1].setText(str(ctrl.coins_collected))
        self._stats["avoided"][1].setText(str(ctrl.bombs_avoided))
        self._stats["hits"][1].setText(str(ctrl.bomb_hits))

        if stretch:
            self.instruction.setText(game.exercise.name)
            set_tone(self.instruction, "done" if game.celebrating(now) and game.message_good else "")
            self.hint.setText(f"{game.exercise.hint}, then hold it for {CONFIG.stretch_hold_seconds:g} seconds."
                              f"\nUp next: {game.next_exercise.name}")
            self.guide.set_kind(game.exercise.key)
            self.progress_caption.setText("HOLD THE POSE" if game.celebrating(now) else STRETCH_STATUS[game.status])
            self.progress.setValue(int(game.hold_progress(now) * 100))
        elif coin:
            self.instruction.setText(game.instruction())
            set_tone(self.instruction, "danger" if game.bomb is not None else "")
            self.hint.setText("Reach coins with either hand (+5). Bomb from the side: lean away. "
                              "Bomb from above: duck down.")
            self.progress_caption.setText("ROUND PROGRESS")
            elapsed = ctrl.phase.duration - ctrl.phase_remaining(now)
            self.progress.setValue(int(100 * elapsed / max(ctrl.phase.duration, 1e-6)))
        else:
            self.instruction.setText(ctrl.transition_title)
            set_tone(self.instruction, "")
            self.hint.setText(ctrl.transition_subtitle)
        self.canvas.set_scene(ctrl, now, pixmap, pose)

    def _show_results(self, ctrl: ActiveBreak, now: float):
        self._pages.setCurrentIndex(1)
        self.results_score.setText(str(ctrl.score))
        self._result_values["stretches"].setText(str(ctrl.stretches_completed))
        self._result_values["coins"].setText(str(ctrl.coins_collected))
        self._result_values["avoided"].setText(str(ctrl.bombs_avoided))
        self._result_values["hits"].setText(str(ctrl.bomb_hits))
        self.results_note.setText(f"Returning to monitoring automatically in {math.ceil(ctrl.results_remaining(now))}s")
