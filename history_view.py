"""Session History page: saved sessions (newest first), a per-session report, filters and a score trend.

Visual layer only. All numbers come from history.HistoryStore (the saved SessionTracker statistics).
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout,
    QWidget,
)

from config import CONFIG
from history import FILTERS, HistoryStore, SessionRecord, filter_since, summarize
from posture_analyzer import score_band
from ui_kit import BLUE, CYAN, GOOD, POOR, VIOLET, WARN, draw_icon, format_seconds, make_label, set_tone

GOOD_MIN = int(score_band("GOOD")[0])       # 80: same bands as the live Posture Score
WARNING_MIN = int(score_band("WARNING")[0])  # 60

ACCENT = "#007AFF"
INK = "#172033"
INK_2 = "#637083"      # secondary text (>= 4.5:1 on white)
LINE = "#E5EAF1"
GRID = "#EEF1F5"
ISSUE_COLORS = {"Forward Head": BLUE, "Slouching": VIOLET, "Uneven Shoulder": CYAN, "Leaning": WARN}
BACK_LINK_H, BACK_LINK_GAP = 18, 6   # header: back-link line height and gap to the title
SHORT_FILTERS = {"all": "All", "today": "Today", "7d": "7 Days", "30d": "30 Days"}

HISTORY_QSS = f"""
QWidget#historyView, QWidget#historyPage {{ background: #F5F7FA; }}
QScrollArea#historyScroll {{ background: transparent; border: none; }}
QWidget#historyScrollBody {{ background: transparent; }}
QFrame#hCard {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; }}
QFrame#hTile {{ background: #F5F7FA; border: none; border-radius: 12px; }}
QLabel {{ color: {INK}; background: transparent; }}

QLabel#hTitle {{ font-size: 29px; font-weight: 600; color: {INK}; }}
QLabel#hSubtitle {{ font-size: 12px; color: {INK_2}; }}
QLabel#hSection {{ font-size: 15px; font-weight: 600; color: {INK}; }}
QLabel#hCaption {{ font-size: 12px; color: {INK_2}; }}
QLabel#hCardLabel {{ font-size: 12px; font-weight: 500; color: {INK_2}; }}
QLabel#hMetric {{ font-size: 30px; font-weight: 600; color: {INK}; }}
QLabel#hMetricUnit {{ font-size: 14px; font-weight: 500; color: {INK_2}; }}
QLabel#hTileValue {{ font-size: 15px; font-weight: 600; color: {INK}; }}
QLabel#hTileLabel {{ font-size: 12px; color: {INK_2}; }}
QLabel#hBody {{ font-size: 13px; color: {INK}; }}
QLabel#hBodyStrong {{ font-size: 13px; font-weight: 600; color: {INK}; }}
QLabel#hValue {{ font-size: 13px; font-weight: 600; color: {INK}; }}
QLabel#hMuted {{ font-size: 12px; color: {INK_2}; }}
QLabel#hDetailTitle {{ font-size: 20px; font-weight: 600; color: {INK}; }}
QLabel#hInsight {{ font-size: 14px; font-weight: 600; color: {INK}; }}
QLabel#hEmptyTitle {{ font-size: 17px; font-weight: 600; color: {INK}; }}
QLabel#hEmptyText {{ font-size: 13px; color: {INK_2}; }}

QPushButton#hBack {{
    background: transparent; border: none; padding: 0; text-align: left; min-height: 18px; max-height: 18px;
    font-size: 12px; font-weight: 500; color: {INK_2};
}}
QPushButton#hBack:hover {{ color: {ACCENT}; }}
QPushButton#hBack:pressed {{ color: #0062CC; }}
QFrame#hSegmented {{ background: #EDF0F5; border: none; border-radius: 9px; }}
QPushButton#hSegment {{
    background: transparent; border: 1px solid transparent; border-radius: 7px;
    min-height: 26px; min-width: 58px; padding: 0 12px; font-size: 12px; font-weight: 500; color: {INK_2};
}}
QPushButton#hSegment:hover {{ color: {INK}; }}
QPushButton#hSegment:checked {{ background: #FFFFFF; border-color: {LINE}; color: {INK}; font-weight: 600; }}

QFrame#hRow {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#hRow:hover {{ border-color: #D3DBE6; }}
QFrame#hRow[tone="selected"] {{ background: #F2F8FF; border: 1px solid {ACCENT}; }}
QLabel#hRowDate {{ font-size: 13px; font-weight: 600; color: {INK}; }}
QLabel#hRowMeta {{ font-size: 12px; color: {INK_2}; }}

QLabel#hBadge {{ border-radius: 8px; padding: 2px 8px; font-size: 12px; font-weight: 600; }}
QLabel#hBadge[tone="good"] {{ background: #E8F8EE; color: #15803D; }}
QLabel#hBadge[tone="warning"] {{ background: #FEF4E2; color: #B45309; }}
QLabel#hBadge[tone="poor"] {{ background: #FDECEC; color: #B91C1C; }}
QLabel#hBadge[tone="neutral"] {{ background: #EEF1F5; color: {INK_2}; }}
QLabel#hBadge[tone="live"] {{ background: #E6F1FF; color: #0062CC; }}
QLabel#hBadge[tone="done"] {{ background: #EEF1F5; color: #475467; }}

QFrame#hDivider {{ background: {GRID}; border: none; min-height: 1px; max-height: 1px; }}
QPushButton#hDetailsToggle {{
    background: transparent; border: none; padding: 0; font-size: 12px; font-weight: 500; color: {ACCENT};
    text-align: left;
}}
QPushButton#hDetailsToggle:hover {{ color: #0062CC; }}
QFrame#hInsightBox {{ background: #F5F9FF; border: none; border-radius: 12px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #CED6E0; border-radius: 3px; min-height: 36px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def score_tone(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    return "good" if round(score) >= GOOD_MIN else ("warning" if round(score) >= WARNING_MIN else "poor")


TONE_TEXT = {"good": "Good", "warning": "Warning", "poor": "Poor", "neutral": "No data"}

def _when(ts: float) -> datetime:
    return datetime.fromtimestamp(ts)  # local time


def _card(name: str = "hCard", margins=(18, 16, 18, 16), spacing: int = 10):
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return frame, layout


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("hDivider")
    return line


def _badge(text: str, tone: str) -> QWidget:
    badge = make_label(text, "hBadge")
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    set_tone(badge, tone)
    return badge


def _duration(seconds: float) -> str:
    """Readable duration: '25 min 4 s', '1 h 05 min', '42 s'."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s} s" if s else f"{m} min"
    return f"{s} s"


def insight(r: SessionRecord) -> tuple:
    """(headline, detail) built only from the stored values; neutral when the data is insufficient."""
    if r.monitored_seconds <= 0 or r.score is None:
        return ("Not enough reliable posture data was recorded in this session.",
                "Scores and durations appear once both cameras have measured your posture.")
    top = r.top_issue
    if not top:
        return "No posture issues were detected during this session.", ""
    share = 100 * top[1] / r.monitored_seconds
    return (f"{top[0]} was the most frequently detected issue during this session.",
            f"It was present for about {share:.0f}% of the monitored time.")


# ---------------------------------------------------------------------------
# painted components
# ---------------------------------------------------------------------------

class DistributionBar(QWidget):
    """Segmented Good / Warning / Poor bar (thin, rounded, small gaps between segments)."""

    def __init__(self, height: int = 8):
        super().__init__()
        self.values = (0.0, 0.0, 0.0)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_values(self, good: float, warning: float, poor: float):
        self.values = (good or 0.0, warning or 0.0, poor or 0.0)
        self.setToolTip(f"Good {self.values[0]:.0f}% · Warning {self.values[1]:.0f}% · Poor {self.values[2]:.0f}%")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        p.fillPath(clip, QColor(GRID))
        total = sum(self.values)
        if total <= 0:
            return
        p.setClipPath(clip)
        parts = [(v, c) for v, c in zip(self.values, (GOOD, WARN, POOR)) if v > 0]
        gap = 2.0 if len(parts) > 1 else 0.0
        width = rect.width() - gap * (len(parts) - 1)
        x = rect.left()
        for value, color in parts:
            w = width * value / total
            p.fillRect(QRectF(x, rect.top(), w, rect.height()), QColor(color))
            x += w + gap


class TrendChart(QWidget):
    """Average score per session, oldest to newest, on a 0-100 scale with the Good (80) / Warning (60) lines."""

    def __init__(self):
        super().__init__()
        self.points: List[tuple] = []   # (started_at, score) actually stored
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_points(self, points):
        self.points = list(points)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(11)
        p.setFont(font)
        left, right, top, bottom = 28, 8, 6, 20
        area = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)

        def y_of(score):
            return area.bottom() - area.height() * max(0.0, min(100.0, score)) / 100

        for value in (0, 50, 100):
            p.setPen(QPen(QColor(GRID), 1))
            p.drawLine(QPointF(area.left(), y_of(value)), QPointF(area.right(), y_of(value)))
        for value, color in ((GOOD_MIN, GOOD), (WARNING_MIN, WARN)):   # the app's score bands
            c = QColor(color)
            c.setAlpha(110)
            p.setPen(QPen(c, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(area.left(), y_of(value)), QPointF(area.right(), y_of(value)))
        p.setPen(QColor(INK_2))
        for value in (0, WARNING_MIN, GOOD_MIN, 100):
            p.drawText(QRectF(0, y_of(value) - 7, left - 6, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(value))

        if not self.points:
            return
        n = len(self.points)
        xs = [area.center().x()] if n == 1 else [area.left() + 6 + (area.width() - 12) * i / (n - 1) for i in range(n)]
        if n > 1:
            path = QPainterPath(QPointF(xs[0], y_of(self.points[0][1])))
            for x, (_, score) in zip(xs[1:], self.points[1:]):
                path.lineTo(QPointF(x, y_of(score)))
            p.setPen(QPen(QColor(ACCENT), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for x, (_, score) in zip(xs, self.points):
            p.setPen(QPen(QColor(ACCENT), 2))
            p.setBrush(QColor("#FFFFFF"))
            p.drawEllipse(QPointF(x, y_of(score)), 3.5, 3.5)
        p.setPen(QColor(INK_2))
        if n == 1:
            score = self.points[0][1]
            p.drawText(QRectF(area.left(), area.bottom() + 3, area.width(), 16), Qt.AlignmentFlag.AlignCenter,
                       f"{_when(self.points[0][0]):%d %b}")
            p.setPen(QColor(INK))
            bold = QFont(font)
            bold.setWeight(QFont.Weight.DemiBold)
            p.setFont(bold)
            p.drawText(QRectF(xs[0] - 30, y_of(score) - 24, 60, 16), Qt.AlignmentFlag.AlignCenter, str(round(score)))
        else:
            for x, (ts, _), align in ((xs[0], self.points[0], Qt.AlignmentFlag.AlignLeft),
                                      (xs[-1], self.points[-1], Qt.AlignmentFlag.AlignRight)):
                box = QRectF(x - 80 if align == Qt.AlignmentFlag.AlignRight else x - 4, area.bottom() + 3, 84, 16)
                p.drawText(box, align | Qt.AlignmentFlag.AlignVCenter, f"{_when(ts):%d %b}")


class SessionRow(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, r: SessionRecord, live: bool):
        super().__init__()
        self.setObjectName("hRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.session_id = r.id
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        start = _when(r.started_at)
        top.addWidget(make_label(f"{start:%a %d %b} · {start:%H:%M}", "hRowDate"))
        if live:
            top.addWidget(_badge("Live", "live"))
        top.addStretch(1)
        pill = _badge(f"{r.score}" if r.score is not None else "--", score_tone(r.average_score))
        pill.setToolTip(f"Average Posture Score · {TONE_TEXT[score_tone(r.average_score)]}")
        top.addWidget(pill)
        layout.addLayout(top)

        top_issue = r.top_issue
        self.meta = make_label(f"{_duration(r.monitored_seconds)}  ·  "
                               f"{top_issue[0] if top_issue else 'No issues detected'}", "hRowMeta")
        layout.addWidget(self.meta)
        bar = DistributionBar(4)
        bar.set_values(r.good_percent, r.warning_percent, r.poor_percent)
        layout.addSpacing(2)
        layout.addWidget(bar)
        self.setToolTip(f"{start:%A %d %B %Y, %H:%M}\nMonitored {format_seconds(r.monitored_seconds)}\n"
                        f"Good {r.good_percent:.0f}% · Warning {r.warning_percent:.0f}% · Poor {r.poor_percent:.0f}%")

    def set_selected(self, selected: bool):
        set_tone(self, "selected" if selected else "")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.session_id)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

class HistoryView(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, store: HistoryStore):
        super().__init__()
        self.store = store
        self.setObjectName("historyView")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(HISTORY_QSS)
        self.filter = "all"
        self.records: List[SessionRecord] = []
        self.rows: List[SessionRow] = []
        self.selected_id: Optional[str] = None
        self.current_id: Optional[str] = None

        # Whole page scrolls only when the window is very small; at normal sizes everything fits.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("historyScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        page = QWidget()
        page.setObjectName("historyPage")
        scroll.setWidget(page)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(18)
        layout.addLayout(self._build_header())
        layout.addLayout(self._build_overview())
        self.body = QStackedWidget()
        self.body.addWidget(self._build_content())
        self.empty = self._message_page()
        self.body.addWidget(self.empty)
        layout.addWidget(self.body, 1)

    # ------------------------------------------------------------------ A. header

    def _build_header(self) -> QVBoxLayout:
        # Editorial header. Row 1: quiet back link. Row 2: title + subtitle, with the filters centred on them.
        header = QVBoxLayout()
        header.setContentsMargins(4, 0, 4, 8)   # subtle side inset; 8 px extra breathing room below
        header.setSpacing(BACK_LINK_GAP)
        self.btn_back = QPushButton("←  Back to Dashboard")
        self.btn_back.setObjectName("hBack")
        self.btn_back.setFixedHeight(BACK_LINK_H)
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_back.clicked.connect(self.back_requested.emit)
        header.addWidget(self.btn_back, 0, Qt.AlignmentFlag.AlignLeft)
        row = QHBoxLayout()
        row.setSpacing(16)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(make_label("Session History", "hTitle"))
        titles.addWidget(make_label("Your posture monitoring activity", "hSubtitle"))
        row.addLayout(titles)
        row.addStretch(1)

        segmented = QFrame()
        segmented.setObjectName("hSegmented")
        seg = QHBoxLayout(segmented)
        seg.setContentsMargins(3, 3, 3, 3)
        seg.setSpacing(2)
        self.filter_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, full in FILTERS:
            button = QPushButton(SHORT_FILTERS[key])
            button.setObjectName("hSegment")
            button.setToolTip(full)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=key: self.set_filter(k))
            group.addButton(button)
            seg.addWidget(button)
            self.filter_buttons[key] = button
        self.filter_buttons["all"].setChecked(True)
        row.addWidget(segmented, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addLayout(row)
        return header

    # ------------------------------------------------------------------ B. overview

    def _build_overview(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)
        self.stat_values = {}

        score_card, s = _card(spacing=6)
        s.addWidget(make_label("Average Posture Score", "hCardLabel"))
        line = QHBoxLayout()
        line.setSpacing(4)
        self.stat_values["score"] = make_label("--", "hMetric")
        line.addWidget(self.stat_values["score"], 0, Qt.AlignmentFlag.AlignBottom)
        unit = make_label("/100", "hMetricUnit")
        unit.setContentsMargins(0, 0, 0, 6)   # sit on the number's baseline
        line.addWidget(unit, 0, Qt.AlignmentFlag.AlignBottom)
        line.addSpacing(8)
        self.score_class = _badge("No data", "neutral")
        line.addWidget(self.score_class, 0, Qt.AlignmentFlag.AlignVCenter)
        line.addStretch(1)
        s.addLayout(line)
        s.addWidget(make_label("Monitoring-time weighted average", "hCaption"))
        row.addWidget(score_card, 1)

        for key, title, caption in (("time", "Total Monitoring", "Valid posture monitoring time"),
                                    ("count", "Total Sessions", "Recorded sessions")):
            card, c = _card(spacing=6)
            c.addWidget(make_label(title, "hCardLabel"))
            value = make_label("--", "hMetric")
            c.addWidget(value)
            c.addWidget(make_label(caption, "hCaption"))
            row.addWidget(card, 1)
            self.stat_values[key] = value
        return row

    # ------------------------------------------------------------------ C. content

    def _build_content(self) -> QWidget:
        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(16)
        list_card, lc = _card(margins=(14, 14, 8, 14), spacing=10)
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 6, 0)
        header.addWidget(make_label("Sessions", "hSection"))
        header.addStretch(1)
        self.list_title = make_label("", "hMuted")
        header.addWidget(self.list_title)
        lc.addLayout(header)
        self.list_scroll = QScrollArea()
        self.list_scroll.setObjectName("historyScroll")
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder.setObjectName("historyScrollBody")
        self.list_layout = QVBoxLayout(holder)
        self.list_layout.setContentsMargins(0, 0, 6, 0)
        self.list_layout.setSpacing(8)
        self.list_layout.addStretch(1)
        self.list_scroll.setWidget(holder)
        self.list_scroll.setMinimumHeight(150)
        lc.addWidget(self.list_scroll, 1)
        self.list_card = list_card
        left.addWidget(list_card, 3)

        trend_card, tc = _card(spacing=6)
        trend_head = QHBoxLayout()
        trend_head.addWidget(make_label("Score Trend", "hSection"))
        trend_head.addStretch(1)
        trend_head.addWidget(make_label("Average score per session", "hMuted"))
        tc.addLayout(trend_head)
        self.chart = TrendChart()
        tc.addWidget(self.chart, 1)
        self.chart_note = make_label("", "hMuted")
        self.chart_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tc.addWidget(self.chart_note)
        self.trend_card = trend_card
        trend_card.setMinimumHeight(230)
        left.addWidget(trend_card, 2)
        row.addLayout(left, 35)

        self.detail_stack = QStackedWidget()
        self.detail_stack.addWidget(self._build_details())
        self.no_selection = self._small_message("Select a session", "Choose a session on the left to see its report.")
        self.detail_stack.addWidget(self.no_selection)
        row.addWidget(self.detail_stack, 65)
        return content

    def _build_details(self) -> QWidget:
        card, outer = _card(margins=(22, 20, 22, 20), spacing=0)
        d = QVBoxLayout()
        d.setSpacing(12)
        outer.addLayout(d)

        # A. session header
        head = QHBoxLayout()
        head.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.d_title = make_label("", "hDetailTitle")
        self.d_sub = make_label("", "hMuted")
        titles.addWidget(self.d_title)
        titles.addWidget(self.d_sub)
        head.addLayout(titles, 1)
        self.d_status = _badge("Completed", "done")
        head.addWidget(self.d_status, 0, Qt.AlignmentFlag.AlignTop)
        d.addLayout(head)

        # B. overview tiles
        strip = QFrame()
        strip.setObjectName("hTile")
        tiles = QHBoxLayout(strip)
        tiles.setContentsMargins(14, 9, 14, 9)
        tiles.setSpacing(18)
        self.d_tiles = {}
        for i, (key, name) in enumerate((("monitored", "Monitoring"), ("score", "Score"), ("length", "Session"))):
            tiles.addWidget(make_label(name, "hTileLabel"))
            value = make_label("--", "hTileValue")
            tiles.addWidget(value)
            if i < 2:
                tiles.addStretch(1)
            self.d_tiles[key] = value
        d.addWidget(strip)

        # C. distribution
        d.addWidget(make_label("Posture Distribution", "hSection"))
        self.d_bar = DistributionBar(10)
        d.addWidget(self.d_bar)
        legend = QHBoxLayout()
        legend.setSpacing(6)
        self.d_percent = {}
        for i, (key, color) in enumerate((("Good", GOOD), ("Warning", WARN), ("Poor", POOR))):
            if i:
                legend.addSpacing(16)
            legend.addWidget(self._dot(color, 8))
            legend.addWidget(make_label(key, "hBody"))
            value = make_label("--", "hValue")
            legend.addWidget(value)
            self.d_percent[key] = value
        legend.addStretch(1)
        d.addLayout(legend)

        # D. issues
        issues_head = QHBoxLayout()
        issues_head.addWidget(make_label("Detected Posture Issues", "hSection"))
        issues_head.addStretch(1)
        issues_head.addWidget(make_label("may overlap", "hMuted"))
        d.addLayout(issues_head)
        issues = QVBoxLayout()
        issues.setSpacing(0)
        self.d_issues, self.d_issue_names = {}, {}
        for i, name in enumerate(("Forward Head", "Slouching", "Uneven Shoulder", "Leaning")):
            if i:
                issues.addWidget(_divider())
            line = QHBoxLayout()
            line.setContentsMargins(0, 6, 0, 6)
            line.setSpacing(10)
            line.addWidget(self._dot(ISSUE_COLORS[name], 8))
            label = make_label(name, "hBody")
            line.addWidget(label)
            line.addStretch(1)
            value = make_label("--", "hValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            line.addWidget(value)
            issues.addLayout(line)
            self.d_issues[name] = value
            self.d_issue_names[name] = label
        d.addLayout(issues)

        # E. insight
        box, b = _card("hInsightBox", margins=(14, 12, 14, 12), spacing=4)
        self.d_insight = make_label("", "hInsight", wrap=True)
        self.d_summary = make_label("", "hMuted", wrap=True)   # secondary sentence
        b.addWidget(self.d_insight)
        b.addWidget(self.d_summary)
        d.addWidget(box)
        self.d_details_toggle = QPushButton()
        self.d_details_toggle.setObjectName("hDetailsToggle")
        self.d_details_toggle.setCheckable(True)
        self.d_details_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.d_details_toggle.toggled.connect(self._toggle_details)
        d.addWidget(self.d_details_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.d_excluded = make_label("", "hMuted", wrap=True)
        self.d_excluded.hide()
        d.addWidget(self.d_excluded)
        d.addStretch(1)
        return card

    @staticmethod
    def _dot(color: str, size: int) -> QFrame:
        dot = QFrame()
        dot.setFixedSize(size, size)
        dot.setStyleSheet(f"background:{color}; border:none; border-radius:{size // 2}px;")
        return dot

    def _message_page(self) -> QWidget:
        page, layout = _card(margins=(32, 32, 32, 32), spacing=8)
        layout.addStretch(1)
        icon = make_label("", "hEmptyIcon")
        icon.setPixmap(draw_icon("history", "#9AA6B6", 36))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)
        layout.addSpacing(4)
        self.empty_title = make_label("No sessions yet", "hEmptyTitle")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_text = make_label("", "hEmptyText", wrap=True)
        self.empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_text.setFixedWidth(440)     # short, readable lines (wraps to its height)
        layout.addWidget(self.empty_title)
        centered = QHBoxLayout()   # (an alignment flag would stop the label from growing to its wrapped height)
        centered.addStretch(1)
        centered.addWidget(self.empty_text)
        centered.addStretch(1)
        layout.addLayout(centered)
        layout.addStretch(1)
        return page

    @staticmethod
    def _small_message(title: str, text: str) -> QWidget:
        page, layout = _card(margins=(24, 24, 24, 24), spacing=6)
        layout.addStretch(1)
        for value, name in ((title, "hEmptyTitle"), (text, "hEmptyText")):
            label = make_label(value, name, wrap=True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ data binding (unchanged logic)

    def set_filter(self, key: str):
        self.filter = key
        self.filter_buttons[key].setChecked(True)
        self.refresh()

    def refresh(self, current_id: Optional[str] = None):
        """Reload from the database (keeps the selection when that session is still listed)."""
        if current_id is not None:
            self.current_id = current_id
        self.records = self.store.sessions(filter_since(self.filter))
        totals = summarize(self.records)
        avg = totals["average_score"]
        self.stat_values["score"].setText("--" if avg is None else str(avg))
        tone = score_tone(avg)
        self.score_class.setText(TONE_TEXT[tone])
        set_tone(self.score_class, tone)
        self.stat_values["time"].setText(format_seconds(totals["monitored_seconds"]))
        self.stat_values["count"].setText(str(totals["count"]))
        self.chart.set_points([(r.started_at, r.average_score) for r in reversed(self.records)
                               if r.average_score is not None])
        enough = len(self.chart.points) >= 2
        self.chart_note.setText("" if enough else "More sessions needed to display trends.")
        self.chart_note.setVisible(not enough)

        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        for r in self.records:
            row = SessionRow(r, live=r.id == self.current_id)
            row.clicked.connect(self.select)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self.rows.append(row)
        self.list_title.setText(f"{len(self.records)} · {dict(FILTERS)[self.filter]}")
        # Few sessions: the list card hugs its rows and Score Trend takes the freed height.
        # Many sessions: the list keeps a 3:2 share of the column and scrolls.
        margins = self.list_card.layout().contentsMargins()
        natural = (margins.top() + margins.bottom() + self.list_card.layout().spacing() + 22
                   + self.list_layout.sizeHint().height())
        self.list_card.setMaximumHeight(max(210, natural))

        if not self.store.available and not self.records:
            for value in self.stat_values.values():   # unknown, not zero
                value.setText("--")
            self._show_message("Session history is unavailable",
                               "Posture monitoring still works, but sessions can't be saved or shown right now. "
                               f"Data file: {self.store.path}. Details are in logs/posture_guard.log.")
        elif not self.records:
            if self.filter == "all":
                self._show_message("No sessions yet",
                                   "Open the cameras to start monitoring. Sessions are saved automatically once "
                                   f"{_min_text()} of valid posture monitoring has been recorded.")
            else:
                self._show_message("No sessions in this period", "Try another filter to see earlier sessions.")
        else:
            self.body.setCurrentIndex(0)
            ids = [r.id for r in self.records]
            self.select(self.selected_id if self.selected_id in ids else ids[0])

    def _show_message(self, title: str, text: str):
        self.empty_title.setText(title)
        self.empty_text.setText(text)
        self.body.setCurrentWidget(self.empty)

    def select(self, session_id: str):
        record = next((r for r in self.records if r.id == session_id), None)
        if record is None:
            self.detail_stack.setCurrentWidget(self.no_selection)
            return
        self.selected_id = session_id
        for row in self.rows:
            row.set_selected(row.session_id == session_id)
        self.detail_stack.setCurrentIndex(0)
        self._show_details(record)

    def _show_details(self, r: SessionRecord):
        start = _when(r.started_at)
        self.d_title.setText(f"{start:%A, %d %B %Y}")
        live = r.id == self.current_id
        if live:
            sub = f"Started {start:%H:%M}"
        elif r.recovered:
            sub = f"Started {start:%H:%M} · saved before the app closed unexpectedly"
        elif r.ended_at:
            end = _when(r.ended_at)   # name the day when the session ran past midnight
            sub = f"Started {start:%H:%M} · ended {end:%H:%M}" if end.date() == start.date() \
                else f"Started {start:%H:%M} · ended {end:%a %d %b, %H:%M}"
        else:
            sub = f"Started {start:%H:%M}"
        self.d_sub.setText(sub)
        self.d_status.setText("In Progress" if live else "Completed")
        set_tone(self.d_status, "live" if live else "done")

        self.d_tiles["monitored"].setText(format_seconds(r.monitored_seconds))
        self.d_tiles["score"].setText("--" if r.score is None else str(r.score))
        self.d_tiles["length"].setText(format_seconds(r.session_seconds))
        self.d_bar.set_values(r.good_percent, r.warning_percent, r.poor_percent)
        for key, value in (("Good", r.good_percent), ("Warning", r.warning_percent), ("Poor", r.poor_percent)):
            self.d_percent[key].setText(f"{value:.1f}%")
        top = r.top_issue
        for name, seconds in r.issues:
            self.d_issues[name].setText(format_seconds(seconds))
            self.d_issue_names[name].setObjectName("hBodyStrong" if top and top[0] == name else "hBody")
            self.d_issue_names[name].style().unpolish(self.d_issue_names[name])
            self.d_issue_names[name].style().polish(self.d_issue_names[name])
        headline, detail = insight(r)
        self.d_insight.setText(headline)
        self.d_summary.setText(detail)
        self.d_summary.setVisible(bool(detail))
        excluded = [(name, s) for name, s in (("one camera only", r.partial_seconds),
                                                ("insufficient data", r.insufficient_seconds),
                                                ("Active Break", r.break_seconds),
                                                ("cameras stopped", r.stopped_seconds)) if s and s >= 1]
        self.d_excluded.setText("Not counted as monitoring: " + " · ".join(f"{n} {_duration(s)}" for n, s in excluded)
                                if excluded else "")
        self.d_details_toggle.setVisible(bool(excluded))
        self._toggle_details(self.d_details_toggle.isChecked())

    def _toggle_details(self, shown: bool):
        has = bool(self.d_excluded.text())
        self.d_excluded.setVisible(shown and has)
        self.d_details_toggle.setText(("▾  Hide details" if shown else "▸  Details") if has else "")


def _min_text() -> str:
    seconds = int(CONFIG.history_min_monitored_seconds)
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    return f"{seconds // 60} minute{'s' if seconds >= 120 else ''}"
