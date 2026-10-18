"""Video Preview Widget with Interactive Subtitle ROI Box & Smooth Live Playback."""

import os
from typing import Optional, Callable, List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QGroupBox
from PySide6.QtCore import Qt, QUrl, Signal, QSizeF, QRectF, QPointF
from PySide6.QtGui import QTransform, QPainter, QColor, QPen, QBrush, QFont, QCursor, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, QGraphicsPixmapItem
from ...domain.enums import FlipMode, CoverMode
from ...domain.models import BlurCoverSettings, SubtitleStyleSettings, LogoSettings
from ...domain.segments import SubtitleSegment
from ..i18n import tr


def ass_to_hex(ass_col: str) -> str:
    """Convert ASS color (&HAABBGGRR or &HBBGGRR) to RGB hex (#RRGGBB)."""
    c = str(ass_col).replace("&H", "").replace("&h", "")
    if len(c) == 8:
        b, g, r = c[2:4], c[4:6], c[6:8]
        return f"#{r}{g}{b}".upper()
    elif len(c) == 6:
        b, g, r = c[0:2], c[2:4], c[4:6]
        return f"#{r}{g}{b}".upper()
    return "#FFFFFF"


class SingleInteractiveRoiItem(QGraphicsRectItem):
    """
    Single draggable ROI box used for both:
    - OCR subtitle extraction region
    - Old subtitle blur/cover region during render
    """

    def __init__(self, x: float = 0, y: float = 0, w: float = 600, h: float = 120,
                 on_changed: Optional[Callable] = None):
        super().__init__(x, y, w, h)
        self.on_changed = on_changed
        self.setAcceptHoverEvents(True)
        self.cover_mode = CoverMode.BLUR
        self.solid_color = "#000000"
        self._drag_type = None
        self._start_scene_pos = QPointF()
        self._start_rect = QRectF(x, y, w, h)
        self.setZValue(120)

    def set_roi(self, x: float, y: float, w: float, h: float,
                mode: CoverMode = CoverMode.BLUR, color: str = "#000000") -> None:
        self.setRect(QRectF(x, y, max(30, w), max(15, h)))
        self.cover_mode = mode
        self.solid_color = color
        self.update()

    def _get_handle_under_mouse(self, pos: QPointF) -> Optional[str]:
        r = self.rect()
        margin = 14.0
        corners = {
            "BR": QRectF(r.right() - margin, r.bottom() - margin, margin * 2, margin * 2),
            "BL": QRectF(r.left() - margin, r.bottom() - margin, margin * 2, margin * 2),
            "TR": QRectF(r.right() - margin, r.top() - margin, margin * 2, margin * 2),
            "TL": QRectF(r.left() - margin, r.top() - margin, margin * 2, margin * 2),
        }
        for name, rect in corners.items():
            if rect.contains(pos):
                return name
        if r.contains(pos):
            return "MOVE"
        return None

    def hoverMoveEvent(self, event) -> None:
        handle = self._get_handle_under_mouse(event.pos())
        cursors = {
            "BR": Qt.SizeFDiagCursor, "TL": Qt.SizeFDiagCursor,
            "BL": Qt.SizeBDiagCursor, "TR": Qt.SizeBDiagCursor,
            "MOVE": Qt.SizeAllCursor,
        }
        self.setCursor(QCursor(cursors.get(handle, Qt.ArrowCursor)))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_type = self._get_handle_under_mouse(event.pos())
            self._start_scene_pos = event.scenePos()
            self._start_rect = self.rect()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_type:
            delta = event.scenePos() - self._start_scene_pos
            dx, dy = delta.x(), delta.y()
            r = self._start_rect
            new_x, new_y = r.x(), r.y()
            new_w, new_h = r.width(), r.height()

            if self._drag_type == "MOVE":
                new_x += dx
                new_y += dy
            elif self._drag_type == "BR":
                new_w = max(30, r.width() + dx)
                new_h = max(15, r.height() + dy)
            elif self._drag_type == "BL":
                new_x = min(r.right() - 30, r.x() + dx)
                new_w = r.right() - new_x
                new_h = max(15, r.height() + dy)
            elif self._drag_type == "TR":
                new_y = min(r.bottom() - 15, r.y() + dy)
                new_w = max(30, r.width() + dx)
                new_h = r.bottom() - new_y
            elif self._drag_type == "TL":
                new_x = min(r.right() - 30, r.x() + dx)
                new_y = min(r.bottom() - 15, r.y() + dy)
                new_w = r.right() - new_x
                new_h = r.bottom() - new_y

            self.setRect(QRectF(new_x, new_y, new_w, new_h))
            if self.on_changed:
                self.on_changed(int(new_x), int(new_y), int(new_w), int(new_h))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_type = None
        event.accept()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        r = self.rect()
        painter.setRenderHint(QPainter.Antialiasing)

        if self.cover_mode == CoverMode.SOLID:
            color = QColor(self.solid_color)
            color.setAlpha(220)
            painter.fillRect(r, QBrush(color))
            border_pen = QPen(QColor("#00E5FF"), 2, Qt.SolidLine)
        else:
            blur_bg = QColor("#007AFF")
            blur_bg.setAlpha(70)
            painter.fillRect(r, QBrush(blur_bg))
            border_pen = QPen(QColor("#00E5FF"), 2, Qt.DashLine)

        painter.setPen(border_pen)
        painter.drawRect(r)

        # Corner handles
        handle_sz = 10.0
        painter.setBrush(QBrush(QColor("#FFD600")))
        painter.setPen(QPen(QColor("#000000"), 1))
        for pt in [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]:
            painter.drawRect(QRectF(pt.x() - handle_sz / 2, pt.y() - handle_sz / 2,
                                    handle_sz, handle_sz))

        # Badge label
        w_int, h_int = int(r.width()), int(r.height())
        label = f"SUB REGION ({w_int}×{h_int} px)"
        painter.setFont(QFont("Arial", 9, QFont.Bold))

        badge_w = min(r.width(), 240)
        badge_y = r.top() - 22 if r.top() >= 24 else r.top() + 4
        badge_rect = QRectF(r.left() + 2, badge_y, badge_w, 20)
        painter.fillRect(badge_rect, QBrush(QColor(0, 0, 0, 200)))
        painter.setPen(QColor("#00E5FF"))
        painter.drawText(badge_rect.adjusted(6, 1, -6, -1), Qt.AlignLeft | Qt.AlignVCenter, label)


class VideoPreviewWidget(QWidget):
    position_changed = Signal(int)
    duration_changed = Signal(int)
    roi_coordinates_changed = Signal(int, int, int, int)
    ocr_roi_changed = Signal(int, int, int, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.1)

        self.tts_player = QMediaPlayer(self)
        self.tts_audio_output = QAudioOutput(self)
        self.tts_player.setAudioOutput(self.tts_audio_output)
        self.tts_audio_output.setVolume(1.0)

        self.bgm_player = QMediaPlayer(self)
        self.bgm_audio_output = QAudioOutput(self)
        self.bgm_player.setAudioOutput(self.bgm_audio_output)
        self.bgm_audio_output.setVolume(0.85)
        self.isolated_bgm_path: Optional[str] = None

        self.preview_segments: List[SubtitleSegment] = []
        self._current_playing_seg_id: Optional[str] = None
        self.master_preview_audio_path: Optional[str] = None
        self.logo_settings: Optional[LogoSettings] = None
        self.subtitle_style: SubtitleStyleSettings = SubtitleStyleSettings()
        self._last_sub_text: str = ""

        self.scene = QGraphicsScene(self)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.player.setVideoOutput(self.video_item)

        self.roi_item = SingleInteractiveRoiItem(on_changed=self._on_roi_item_changed)
        self.scene.addItem(self.roi_item)

        # Logo watermark overlay item (rendered above video, below subtitles)
        self.logo_item = QGraphicsPixmapItem()
        self.logo_item.setZValue(130)
        self.logo_item.setVisible(False)
        self.scene.addItem(self.logo_item)

        self.sub_overlay_item = QGraphicsTextItem()
        self.sub_overlay_item.setDefaultTextColor(QColor("#FFFFFF"))
        self.sub_overlay_item.setFont(QFont("Arial", 16, QFont.Bold))
        self.sub_overlay_item.setZValue(150)
        self.sub_overlay_item.setVisible(False)
        self.scene.addItem(self.sub_overlay_item)

        self.view = QGraphicsView(self.scene, self)
        self.view.setFixedSize(400, 260)
        self.view.setStyleSheet(
            "background-color: #000000; border: 1px solid #30363D; border-radius: 6px;"
        )
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.current_zoom: float = 1.0
        self.current_flip: FlipMode = FlipMode.OFF

        self.btn_ocr_box = None
        self._init_ui()
        self.btn_ocr_box = self.btn_toggle_roi
        self._connect_signals()

    def _init_ui(self) -> None:
        self.setFixedWidth(416)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Header bar
        h_head = QHBoxLayout()
        lbl_head = QLabel("🎬 Xem trước (400×260)")
        lbl_head.setObjectName("panelTitle")
        h_head.addWidget(lbl_head)
        h_head.addStretch()

        self.btn_toggle_roi = QPushButton("⛶ Vùng chữ")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.setChecked(True)
        self.btn_toggle_roi.setFixedHeight(24)
        self.btn_toggle_roi.setToolTip(tr("preview.roi_toggle"))
        self.btn_toggle_roi.toggled.connect(self._toggle_roi_visibility)
        h_head.addWidget(self.btn_toggle_roi)
        layout.addLayout(h_head)

        # Video View strictly locked at 400x260
        layout.addWidget(self.view, alignment=Qt.AlignCenter)

        # Playback controls
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton(tr("preview.play"))
        self.play_button.setFixedWidth(68)
        self.play_button.setFixedHeight(28)
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("timecodeLabel")
        self.time_label.setFixedWidth(82)
        controls.addWidget(self.time_label)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self.seek)
        controls.addWidget(self.seek_slider, stretch=1)

        layout.addLayout(controls)

        # Information Card
        grp_info = QGroupBox("📊 Thông số phát")
        v_info = QVBoxLayout(grp_info)
        v_info.setContentsMargins(8, 8, 8, 8)
        v_info.setSpacing(4)

        self.lbl_info_res = QLabel("Độ phân giải: Chưa nạp video")
        self.lbl_info_res.setObjectName("mutedLabel")
        v_info.addWidget(self.lbl_info_res)

        self.lbl_info_audio = QLabel("Âm thanh nguồn: 10%")
        self.lbl_info_audio.setObjectName("mutedLabel")
        v_info.addWidget(self.lbl_info_audio)

        lbl_hint = QLabel("💡 Kéo 4 góc viền vàng trên màn hình để đặt vùng phụ đề.")
        lbl_hint.setObjectName("mutedLabel")
        v_info.addWidget(lbl_hint)

        layout.addWidget(grp_info)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.video_item.nativeSizeChanged.connect(self._on_native_size_changed)

    def _on_roi_item_changed(self, x: int, y: int, w: int, h: int) -> None:
        self.roi_coordinates_changed.emit(x, y, w, h)
        self.ocr_roi_changed.emit(x, y, w, h)

    def _toggle_roi_visibility(self, visible: bool) -> None:
        self.roi_item.setVisible(visible)
        self.scene.update()

    def set_roi_settings(self, settings: BlurCoverSettings) -> None:
        w = settings.width if settings.width > 0 else 600
        h = settings.height if settings.height > 0 else 120
        self.roi_item.set_roi(settings.x, settings.y, w, h, settings.mode, settings.solid_color)
        self.roi_item.setVisible(settings.enabled or self.btn_toggle_roi.isChecked())

    def get_roi(self) -> dict:
        r = self.roi_item.rect()
        rect = self.video_item.boundingRect()
        vw = int(rect.width()) if rect.width() > 0 else 1280
        vh = int(rect.height()) if rect.height() > 0 else 720
        return {
            "x": int(max(0, r.x())),
            "y": int(max(0, r.y())),
            "width": int(max(30, r.width())),
            "height": int(max(15, r.height())),
            "scene_width": vw,
            "scene_height": vh,
        }

    def get_ocr_roi(self) -> dict:
        return self.get_roi()

    def set_ocr_roi_visible(self, visible: bool) -> None:
        self.btn_toggle_roi.setChecked(visible)
        self.roi_item.setVisible(visible)
        self.scene.update()

    def set_ocr_roi_preset(self, preset: str) -> None:
        rect = self.video_item.boundingRect()
        vw = rect.width() if rect.width() > 0 else 1280
        vh = rect.height() if rect.height() > 0 else 720

        # Map preset names from combo data values to actual geometry
        presets = {
            "bottom": (0.85, 0.22, 0.78),      # 85% wide, 22% tall, from 78% Y to 100% Y
            "center": (0.75, 0.20, 0.45),       # 75% wide, 20% tall, at 45% Y
            "full_bottom": (0.96, 0.26, 0.74),  # 96% wide, 26% tall, from 74% Y to 100% Y
        }
        w_pct, h_pct, y_pct = presets.get(preset, (0.85, 0.22, 0.78))
        bw = int(vw * w_pct)
        bh = int(vh * h_pct)
        bx = int((vw - bw) / 2.0)
        by = int(vh * y_pct)

        self.roi_item.set_roi(bx, by, bw, bh)
        self.roi_item.setVisible(True)
        self.btn_toggle_roi.setChecked(True)
        self.roi_coordinates_changed.emit(bx, by, bw, bh)
        self.ocr_roi_changed.emit(bx, by, bw, bh)

    def apply_roi_preset(self, preset: str) -> None:
        """Alias for set_ocr_roi_preset."""
        self.set_ocr_roi_preset(preset)

    def _on_native_size_changed(self, size: QSizeF) -> None:
        if size.width() > 0 and size.height() > 0:
            old_rect = self.video_item.boundingRect()
            old_w = old_rect.width()
            old_h = old_rect.height()
            self.video_item.setSize(size)
            self.scene.setSceneRect(0, 0, size.width(), size.height())
            self.view.fitInView(self.video_item, Qt.KeepAspectRatio)
            self.apply_transform(self.current_zoom, self.current_flip)
            if hasattr(self, "lbl_info_res"):
                self.lbl_info_res.setText(f"Độ phân giải: {int(size.width())}×{int(size.height())}")
            if self.logo_settings:
                self.set_logo_settings(self.logo_settings)
            # Tự động co giãn khung quét OCR/Blur tỷ lệ thuận theo độ phân giải gốc của video
            if old_w > 0 and old_h > 0 and (abs(old_w - size.width()) > 5 or abs(old_h - size.height()) > 5):
                r = self.roi_item.rect()
                sx = size.width() / old_w
                sy = size.height() / old_h
                new_x = int(r.x() * sx)
                new_y = int(r.y() * sy)
                new_w = int(r.width() * sx)
                new_h = int(r.height() * sy)
                self.roi_item.set_roi(new_x, new_y, new_w, new_h)
                self.roi_coordinates_changed.emit(new_x, new_y, new_w, new_h)
                self.ocr_roi_changed.emit(new_x, new_y, new_w, new_h)

    def load_video(self, video_path: str, width: int = 0, height: int = 0) -> None:
        vw = width if width > 0 else 1280
        vh = height if height > 0 else 720

        self.video_item.setSize(QSizeF(vw, vh))
        self.scene.setSceneRect(0, 0, vw, vh)
        self.view.fitInView(self.video_item, Qt.KeepAspectRatio)

        # Default ROI at bottom of video (covers to bottom edge)
        bw = int(vw * 0.85)
        bh = int(vh * 0.22)
        bx = int((vw - bw) / 2.0)
        by = int(vh * 0.78)
        self.roi_item.set_roi(bx, by, bw, bh, CoverMode.BLUR)
        self.roi_item.setVisible(True)
        self.roi_coordinates_changed.emit(bx, by, bw, bh)

        if self.logo_settings:
            self.set_logo_settings(self.logo_settings)
        if self._last_sub_text:
            self._update_live_subtitle(self._last_sub_text)

        self.player.stop()
        self.bgm_player.stop()
        self.bgm_player.setSource(QUrl())
        self.isolated_bgm_path = None
        self.audio_output.setVolume(1.0)
        self.tts_player.stop()
        self._current_playing_seg_id = None
        self.master_preview_audio_path = None
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.pause()

        if hasattr(self, "lbl_info_res"):
            self.lbl_info_res.setText(f"Độ phân giải: {int(vw)}×{int(vh)}")

    def set_master_preview_audio(self, audio_wav_path: str) -> None:
        if audio_wav_path and os.path.exists(audio_wav_path):
            self.master_preview_audio_path = audio_wav_path
            self.tts_player.setSource(QUrl.fromLocalFile(audio_wav_path))

    def set_background_preview_audio(self, bgm_path: str) -> None:
        """Nạp rãnh nhạc nền & hiệu ứng âm thanh không lời (BGM/SFX)."""
        if bgm_path and os.path.exists(bgm_path):
            self.isolated_bgm_path = bgm_path
            self.bgm_player.setSource(QUrl.fromLocalFile(bgm_path))
            # Tắt tiếng video gốc chứa giọng cũ để chỉ nghe BGM + Lồng tiếng mới
            self.audio_output.setVolume(0.0)

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.tts_player.pause()
            if self.bgm_player.playbackState() == QMediaPlayer.PlayingState:
                self.bgm_player.pause()
            self.play_button.setText(tr("preview.play"))
        else:
            self.player.play()
            if self.tts_player.source().isValid():
                self.tts_player.play()
            if self.bgm_player.source().isValid():
                self.bgm_player.play()
            self.play_button.setText(tr("preview.pause"))

    def seek(self, position_ms: int) -> None:
        self.player.setPosition(position_ms)
        if self.master_preview_audio_path and self.tts_player.source().isValid():
            self.tts_player.setPosition(position_ms)
        else:
            self.tts_player.stop()
            self._current_playing_seg_id = None
        if self.isolated_bgm_path and self.bgm_player.source().isValid():
            self.bgm_player.setPosition(position_ms)

    def _on_position_changed(self, pos: int) -> None:
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(pos)
        dur = self.player.duration()
        self.time_label.setText(f"{self._format_time(pos)} / {self._format_time(dur)}")
        self.position_changed.emit(pos)

        # Sync BGM audio position
        if self.isolated_bgm_path and self.bgm_player.source().isValid():
            if abs(self.bgm_player.position() - pos) > 300:
                self.bgm_player.setPosition(pos)

        # Find active segment (synchronized with speech timing)
        active_seg = None
        for seg in self.preview_segments:
            if seg.effective_tts_start_ms <= pos <= seg.effective_tts_end_ms:
                active_seg = seg  # Cập nhật tới phân đoạn mới nhất trong trường hợp thoại chồng lấn

        # Live subtitle overlay
        active_sub_text = ""
        if active_seg:
            active_sub_text = active_seg.translated_text or active_seg.original_text
        self._update_live_subtitle(active_sub_text)

        # TTS audio sync
        if self.master_preview_audio_path and self.tts_player.source().isValid():
            if abs(self.tts_player.position() - pos) > 300:
                self.tts_player.setPosition(pos)
        elif active_seg and active_seg.tts_audio_path and os.path.exists(active_seg.tts_audio_path):
            if self._current_playing_seg_id != active_seg.segment_id:
                self._current_playing_seg_id = active_seg.segment_id
                offset_ms = max(0, pos - active_seg.effective_tts_start_ms)
                self.tts_player.setSource(QUrl.fromLocalFile(active_seg.tts_audio_path))
                self.tts_player.setPosition(offset_ms)
                if self.player.playbackState() == QMediaPlayer.PlayingState:
                    self.tts_player.play()

    def set_logo_settings(self, settings: Optional[LogoSettings]) -> None:
        """Hiển thị logo trực tiếp trên màn hình xem trước (Live Preview)."""
        self.logo_settings = settings
        if not settings or not settings.enabled or not settings.path or not os.path.exists(settings.path):
            self.logo_item.setVisible(False)
            return

        pixmap = QPixmap(settings.path)
        if pixmap.isNull():
            self.logo_item.setVisible(False)
            return

        rect = self.video_item.boundingRect()
        vw = rect.width() if rect.width() > 0 else 1280
        vh = rect.height() if rect.height() > 0 else 720

        scale_ratio = max(0.03, min(0.60, settings.scale_percent / 100.0))
        target_w = max(24, int(vw * scale_ratio))
        scaled_pixmap = pixmap.scaledToWidth(target_w, Qt.SmoothTransformation)
        self.logo_item.setPixmap(scaled_pixmap)
        self.logo_item.setOpacity(max(0.1, min(1.0, settings.opacity)))

        lw = scaled_pixmap.width()
        lh = scaled_pixmap.height()
        mx = getattr(settings, "margin_x", 24)
        my = getattr(settings, "margin_y", 24)

        pos = (settings.position or "top_right").lower()
        if "top_left" in pos:
            lx = mx
            ly = my
        elif "bottom_left" in pos:
            lx = mx
            ly = max(0, vh - lh - my)
        elif "bottom_right" in pos:
            lx = max(0, vw - lw - mx)
            ly = max(0, vh - lh - my)
        else:  # top_right
            lx = max(0, vw - lw - mx)
            ly = my

        self.logo_item.setPos(lx, ly)
        self.logo_item.setVisible(True)

    def set_subtitle_style(self, style: Optional[SubtitleStyleSettings]) -> None:
        """Cập nhật kiểu dáng phông chữ, vị trí và hiển thị 1 dòng cho xem trước."""
        if style:
            self.subtitle_style = style
        if self._last_sub_text:
            self._update_live_subtitle(self._last_sub_text)

    def _update_live_subtitle(self, text: str) -> None:
        self._last_sub_text = text
        if not text.strip():
            self.sub_overlay_item.setVisible(False)
            return

        style = self.subtitle_style or SubtitleStyleSettings()
        is_single = getattr(style, "single_line", True)
        display_text = " ".join(text.split()) if is_single else text

        font_name = getattr(style, "font_name", "Arial")
        font_size = getattr(style, "font_size", 22)
        primary_hex = ass_to_hex(getattr(style, "primary_color", "&H00FFFFFF"))
        outline_hex = ass_to_hex(getattr(style, "outline_color", "&H00000000"))
        outline_w = getattr(style, "outline_width", 2)
        align = getattr(style, "alignment", 2)
        margin_v = getattr(style, "margin_v", 30)

        v_rect = self.video_item.boundingRect()
        vw = v_rect.width() if v_rect.width() > 0 else 1280
        vh = v_rect.height() if v_rect.height() > 0 else 720
        preview_font_size = max(14, int(font_size * (vw / 1280.0)))

        shadows = []
        if outline_w > 0:
            for dx in range(-outline_w, outline_w + 1):
                for dy in range(-outline_w, outline_w + 1):
                    if dx != 0 or dy != 0:
                        shadows.append(f"{dx}px {dy}px 0px {outline_hex}")
        shadow_css = f"text-shadow: {', '.join(shadows)};" if shadows else ""

        self.sub_overlay_item.setHtml(
            f'<div align="center" style="font-family: \'{font_name}\';'
            f' font-size: {preview_font_size}px; font-weight: bold;'
            f' color: {primary_hex}; {shadow_css}'
            f' background-color: rgba(0,0,0,0.50);'
            f' padding: 4px 16px; border-radius: 6px;">'
            f'{display_text}</div>'
        )
        self.sub_overlay_item.setVisible(True)

        t_rect = self.sub_overlay_item.boundingRect()
        x = (vw - t_rect.width()) / 2.0
        if align == 8:  # Top
            y = max(10, margin_v)
        elif align == 5:  # Middle
            y = (vh - t_rect.height()) / 2.0
        else:  # Bottom (default)
            y = max(10, vh - t_rect.height() - margin_v)

        self.sub_overlay_item.setPos(x, y)

    def _on_duration_changed(self, dur: int) -> None:
        self.seek_slider.setRange(0, dur)
        self.time_label.setText(f"00:00 / {self._format_time(dur)}")
        self.duration_changed.emit(dur)

    @staticmethod
    def _format_time(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60:02d}:{s % 60:02d}"

    def set_original_volume(self, volume: float) -> None:
        effective_volume = max(0.0, min(1.0, float(volume)))
        self.audio_output.setVolume(effective_volume)
        self.lbl_info_audio.setText(f"Âm thanh nguồn: {round(effective_volume * 100)}%")

    def set_dubbing_volume(self, volume: float) -> None:
        self.tts_audio_output.setVolume(max(0.0, min(1.0, float(volume))))

    def set_preview_segments(self, segments: List[SubtitleSegment]) -> None:
        self.preview_segments = segments

    def apply_transform(self, zoom_ratio: float, flip_mode: FlipMode) -> None:
        self.current_zoom = zoom_ratio
        self.current_flip = flip_mode

        rect = self.video_item.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        h_scale = -1.0 if flip_mode in (FlipMode.HORIZONTAL, FlipMode.BOTH) else 1.0
        v_scale = -1.0 if flip_mode in (FlipMode.VERTICAL, FlipMode.BOTH) else 1.0

        transform = QTransform()
        transform.translate(cx, cy)
        transform.scale(h_scale * zoom_ratio, v_scale * zoom_ratio)
        transform.translate(-cx, -cy)
        self.video_item.setTransform(transform)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.video_item.boundingRect().width() > 0:
            self.view.fitInView(self.video_item, Qt.KeepAspectRatio)
