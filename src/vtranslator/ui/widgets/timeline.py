"""Multi-track Video Timeline Widget."""

from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QSlider
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from ...domain.segments import SubtitleSegment
from ...domain.characters import Character
from ..i18n import tr


class TrackHeaderPanel(QWidget):
    """Track header column on the left."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(130)
        self.setStyleSheet("background-color: #12151E; border-right: 1px solid #232736;")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)

        lbl_ruler_space = QLabel(tr("timeline.tracks"))
        lbl_ruler_space.setFixedHeight(28)
        lbl_ruler_space.setObjectName("mutedLabel")
        layout.addWidget(lbl_ruler_space)

        self.t_video = self._create_header_item(tr("timeline.video"), "#1F6FEB")
        self.t_video.setFixedHeight(44)
        layout.addWidget(self.t_video)

        self.t_sub = self._create_header_item(tr("timeline.subtitle"), "#8957E5")
        self.t_sub.setFixedHeight(46)
        layout.addWidget(self.t_sub)

        self.t_tts = self._create_header_item(tr("timeline.dubbing"), "#238636")
        self.t_tts.setFixedHeight(44)
        layout.addWidget(self.t_tts)

        self.t_orig = self._create_header_item(tr("timeline.original_audio"), "#D29922")
        self.t_orig.setFixedHeight(36)
        layout.addWidget(self.t_orig)

        layout.addStretch()

    def _create_header_item(self, title: str, color_hex: str) -> QLabel:
        lbl = QLabel(f" {title}")
        lbl.setObjectName("trackHeader")
        lbl.setStyleSheet(f"border-left: 3px solid {color_hex}; margin-top: 3px; margin-bottom: 3px;")
        return lbl


class CapcutTimelineCanvas(QWidget):
    """Multi-track timeline canvas."""

    seek_requested = Signal(int)
    segment_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.segments: List[SubtitleSegment] = []
        self.characters: List[Character] = []
        self.duration_ms: int = 60000
        self.current_time_ms: int = 0
        self.pixels_per_second: float = 40.0
        self.is_dragging_playhead: bool = False
        self.selected_seg_id: Optional[str] = None

        self.setMinimumHeight(210)
        self.setStyleSheet("background-color: #0D1117;")

    def set_duration(self, duration_ms: int) -> None:
        self.duration_ms = max(1000, duration_ms)
        self._update_width()
        self.update()

    def set_zoom(self, px_per_sec: float) -> None:
        self.pixels_per_second = max(10.0, min(200.0, px_per_sec))
        self._update_width()
        self.update()

    def _update_width(self) -> None:
        w = int((self.duration_ms / 1000.0) * self.pixels_per_second) + 200
        self.setMinimumWidth(max(800, w))

    def set_current_time(self, current_time_ms: int) -> None:
        self.current_time_ms = current_time_ms
        self.update()

    def set_data(self, segments: List[SubtitleSegment], characters: List[Character]) -> None:
        self.segments = segments
        self.characters = characters
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        ruler_h = 28
        tr1_y, tr1_h = 32, 40
        tr2_y, tr2_h = 76, 42
        tr3_y, tr3_h = 122, 40
        tr4_y, tr4_h = 166, 32

        # Track backgrounds
        for y, h, bg_color in [
            (tr1_y, tr1_h, "#10141D"), (tr2_y, tr2_h, "#131722"),
            (tr3_y, tr3_h, "#10161E"), (tr4_y, tr4_h, "#12141C")
        ]:
            painter.fillRect(QRectF(0, y, w, h), QBrush(QColor(bg_color)))
            painter.setPen(QPen(QColor("#21262D"), 1))
            painter.drawLine(0, y + h, w, y + h)

        # Time ruler
        painter.fillRect(QRectF(0, 0, w, ruler_h), QBrush(QColor("#161B22")))
        painter.setPen(QPen(QColor("#30363D"), 1))
        painter.drawLine(0, ruler_h, w, ruler_h)

        total_secs = int(self.duration_ms / 1000.0) + 1
        painter.setFont(QFont("Arial", 9))

        step = 5 if self.pixels_per_second < 30 else (2 if self.pixels_per_second < 70 else 1)
        for s in range(0, total_secs, step):
            x = int(s * self.pixels_per_second)
            painter.setPen(QPen(QColor("#6E7681"), 1))
            painter.drawLine(x, ruler_h - 10, x, ruler_h)

            time_txt = f"{s // 60:02d}:{s % 60:02d}"
            painter.setPen(QColor("#8B949E"))
            painter.drawText(x + 4, ruler_h - 12, time_txt)

            painter.setPen(QPen(QColor(33, 38, 45, 120), 1, Qt.DotLine))
            painter.drawLine(x, ruler_h, x, self.height())

        # Video track
        vid_len_px = int((self.duration_ms / 1000.0) * self.pixels_per_second)
        vid_rect = QRectF(0, tr1_y + 2, vid_len_px, tr1_h - 4)
        painter.fillRect(vid_rect, QBrush(QColor("#1F6FEB")))
        painter.setPen(QPen(QColor("#388BFD"), 1))
        painter.drawRoundedRect(vid_rect, 4, 4)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(vid_rect.adjusted(10, 0, -10, 0), Qt.AlignLeft | Qt.AlignVCenter,
                         "Video Track")

        # Subtitle blocks
        char_map = {c.id: c for c in self.characters}
        for seg in self.segments:
            x1 = int(seg.start_sec * self.pixels_per_second)
            x2 = int(seg.end_sec * self.pixels_per_second)
            seg_w = max(10, x2 - x1)

            c = char_map.get(seg.character_id)
            char_color = c.color if c else "#8957E5"

            block_rect = QRectF(x1, tr2_y + 2, seg_w, tr2_h - 4)
            is_sel = (seg.segment_id == self.selected_seg_id)

            bg_sub = QColor(char_color)
            bg_sub.setAlpha(200 if is_sel else 140)
            painter.fillRect(block_rect, QBrush(bg_sub))

            border_pen = QPen(QColor("#FFFFFF" if is_sel else char_color), 2 if is_sel else 1)
            painter.setPen(border_pen)
            painter.drawRoundedRect(block_rect, 4, 4)

            painter.fillRect(QRectF(x1, tr2_y + 2, 4, tr2_h - 4), QBrush(QColor(char_color)))

            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            txt = seg.translated_text or seg.original_text
            display_txt = f"[{c.name if c else 'Char'}] {txt}"
            painter.drawText(block_rect.adjusted(8, 2, -4, -2),
                             Qt.AlignLeft | Qt.AlignVCenter, display_txt)

        # TTS blocks (Track 3) - Drawn at effective speech timing
        for seg in self.segments:
            if seg.tts_audio_path:
                x1 = int((seg.effective_tts_start_ms / 1000.0) * self.pixels_per_second)
                x2 = int((seg.effective_tts_end_ms / 1000.0) * self.pixels_per_second)
                seg_w = max(10, x2 - x1)

                tts_rect = QRectF(x1, tr3_y + 2, seg_w, tr3_h - 4)
                painter.fillRect(tts_rect, QBrush(QColor(35, 134, 54, 180)))
                painter.setPen(QPen(QColor("#2EA043"), 1))
                painter.drawRoundedRect(tts_rect, 4, 4)

                painter.setPen(QPen(QColor("#FFFFFF"), 1))
                for wx in range(int(x1 + 6), int(x1 + seg_w - 6), 4):
                    h_wave = (wx % 12) + 6
                    painter.drawLine(wx, int(tr3_y + 20 - h_wave / 2),
                                     wx, int(tr3_y + 20 + h_wave / 2))

                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(tts_rect.adjusted(6, 0, -4, 0),
                                 Qt.AlignLeft | Qt.AlignVCenter, seg.voice_id or "")

        # Playhead
        play_x = int((self.current_time_ms / 1000.0) * self.pixels_per_second)

        poly = QPolygonF([
            QPointF(play_x - 7, 0),
            QPointF(play_x + 7, 0),
            QPointF(play_x + 7, ruler_h - 10),
            QPointF(play_x, ruler_h),
            QPointF(play_x - 7, ruler_h - 10),
        ])
        painter.setBrush(QBrush(QColor("#FF3B30")))
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawPolygon(poly)

        painter.setPen(QPen(QColor("#FF3B30"), 2))
        painter.drawLine(play_x, ruler_h, play_x, self.height())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            click_x = event.position().x()
            click_y = event.position().y()

            # Check segment click on subtitle track (Track 2)
            seg_clicked = False
            if 76 <= click_y <= 118:
                t_sec = click_x / self.pixels_per_second
                for s in self.segments:
                    if s.start_sec <= t_sec <= s.end_sec:
                        self.selected_seg_id = s.segment_id
                        self.segment_selected.emit(s.segment_id)
                        seg_clicked = True
                        break

            # Always update playhead position
            self.is_dragging_playhead = True
            target_ms = int((click_x / self.pixels_per_second) * 1000.0)
            self.seek_requested.emit(max(0, min(self.duration_ms, target_ms)))
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.is_dragging_playhead:
            click_x = event.position().x()
            target_ms = int((click_x / self.pixels_per_second) * 1000.0)
            self.seek_requested.emit(max(0, min(self.duration_ms, target_ms)))
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.is_dragging_playhead = False


class TimelineWidget(QWidget):
    """Complete timeline widget with toolbar, track headers and scrollable canvas."""

    seek_requested = Signal(int)
    play_pause_requested = Signal()
    segment_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Controls Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(34)
        toolbar.setStyleSheet("background-color: #161B22; border-bottom: 1px solid #30363D;")
        h_tb = QHBoxLayout(toolbar)
        h_tb.setContentsMargins(10, 0, 10, 0)
        h_tb.setSpacing(12)

        self.lbl_timecode = QLabel("00:00 / 00:00")
        self.lbl_timecode.setObjectName("timecodeLabel")
        h_tb.addWidget(self.lbl_timecode)

        h_tb.addStretch()

        h_tb.addWidget(QLabel(tr("timeline.zoom")))
        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setFixedSize(24, 24)
        btn_zoom_out.clicked.connect(self._zoom_out)
        h_tb.addWidget(btn_zoom_out)

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(15, 120)
        self.slider_zoom.setValue(40)
        self.slider_zoom.setFixedWidth(100)
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider_changed)
        h_tb.addWidget(self.slider_zoom)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedSize(24, 24)
        btn_zoom_in.clicked.connect(self._zoom_in)
        h_tb.addWidget(btn_zoom_in)

        layout.addWidget(toolbar)

        # Main timeline area
        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(0)

        self.header_panel = TrackHeaderPanel()
        split.addWidget(self.header_panel)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.canvas = CapcutTimelineCanvas()
        self.canvas.seek_requested.connect(self.seek_requested.emit)
        self.canvas.segment_selected.connect(self.segment_selected.emit)
        self.scroll.setWidget(self.canvas)

        split.addWidget(self.scroll, stretch=1)
        layout.addLayout(split)

    def set_duration(self, dur_ms: int) -> None:
        self.canvas.set_duration(dur_ms)
        self._update_timecode_display()

    def set_current_time(self, cur_ms: int) -> None:
        self.canvas.set_current_time(cur_ms)
        self._update_timecode_display()

    def set_data(self, segments: List[SubtitleSegment], characters: List[Character]) -> None:
        self.canvas.set_data(segments, characters)

    def _update_timecode_display(self) -> None:
        cur_s = self.canvas.current_time_ms // 1000
        dur_s = self.canvas.duration_ms // 1000
        self.lbl_timecode.setText(
            f"{cur_s // 60:02d}:{cur_s % 60:02d} / {dur_s // 60:02d}:{dur_s % 60:02d}"
        )

    def _on_zoom_slider_changed(self, val: int) -> None:
        self.canvas.set_zoom(val)

    def _zoom_in(self) -> None:
        self.slider_zoom.setValue(min(120, self.slider_zoom.value() + 15))

    def _zoom_out(self) -> None:
        self.slider_zoom.setValue(max(15, self.slider_zoom.value() - 15))
