"""Video Frame, Audio Separation, Subtitle Blur & Logo Watermark Editor Widget."""

import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QRadioButton,
    QButtonGroup, QLabel, QSpinBox, QSlider, QGroupBox, QPushButton,
    QLineEdit, QComboBox, QFileDialog, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal
from ...domain.models import BlurCoverSettings, TransformSettings, LogoSettings
from ...domain.enums import CoverMode, FlipMode
from ..i18n import tr


class BlurEditorWidget(QWidget):
    settings_changed = Signal(BlurCoverSettings)
    transform_changed = Signal(int, FlipMode)  # zoom_percent, flip_mode
    logo_changed = Signal(LogoSettings)
    separate_audio_requested = Signal()
    play_audio_requested = Signal(str)  # filepath

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.settings = BlurCoverSettings()
        self.transform_settings = TransformSettings()
        self.logo_settings = LogoSettings()
        self.current_vocals_path = ""
        self.current_bgm_path = ""
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        v_main = QVBoxLayout(container)
        v_main.setContentsMargins(8, 8, 8, 8)
        v_main.setSpacing(12)

        # ══════════════════════════════════════════
        # 1. Audio stems
        # ══════════════════════════════════════════
        grp_sep = QGroupBox("🎵 Tách giọng nói và nhạc nền")
        v_sp = QVBoxLayout(grp_sep)

        lbl_sep_desc = QLabel(
            "Tạo rãnh giọng nói và rãnh nền để nhận diện, biên tập và ghép lại khi xuất. "
            "Chất lượng phụ thuộc phương pháp tách; phương pháp đang dùng sẽ hiển thị bên dưới."
        )
        lbl_sep_desc.setWordWrap(True)
        lbl_sep_desc.setObjectName("mutedLabel")
        v_sp.addWidget(lbl_sep_desc)

        h_sep_act = QHBoxLayout()
        self.btn_separate = QPushButton("✂️ Tách giọng và nhạc nền")
        self.btn_separate.setObjectName("accentButton")
        self.btn_separate.setFixedHeight(34)
        self.btn_separate.clicked.connect(self.separate_audio_requested.emit)
        h_sep_act.addWidget(self.btn_separate)
        v_sp.addLayout(h_sep_act)

        self.lbl_sep_status = QLabel("Chưa tách âm thanh (Sẽ tách tự động khi Render Nhanh)")
        self.lbl_sep_status.setObjectName("badgeOff")
        v_sp.addWidget(self.lbl_sep_status)

        h_play_sep = QHBoxLayout()
        self.btn_play_vocals = QPushButton("▶ Nghe Giọng Sạch (Vocals)")
        self.btn_play_vocals.setEnabled(False)
        self.btn_play_vocals.clicked.connect(lambda: self.play_audio_requested.emit(self.current_vocals_path))
        h_play_sep.addWidget(self.btn_play_vocals)

        self.btn_play_bgm = QPushButton("▶ Nghe Nhạc Nền & SFX (BGM)")
        self.btn_play_bgm.setEnabled(False)
        self.btn_play_bgm.clicked.connect(lambda: self.play_audio_requested.emit(self.current_bgm_path))
        h_play_sep.addWidget(self.btn_play_bgm)
        v_sp.addLayout(h_play_sep)

        v_main.addWidget(grp_sep)

        # ══════════════════════════════════════════
        # 2. PHÓNG TO & LẬT VIDEO (Zoom & Flip)
        # ══════════════════════════════════════════
        grp_zoom_flip = QGroupBox("🔍 Phóng to & Lật Video (Zoom & Flip)")
        v_zf = QVBoxLayout(grp_zoom_flip)

        # Zoom Slider
        h_zoom = QHBoxLayout()
        h_zoom.addWidget(QLabel("Tỷ lệ thu phóng:"))
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(50, 400)
        self.slider_zoom.setValue(100)
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider_changed)
        h_zoom.addWidget(self.slider_zoom)

        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setFixedWidth(50)
        self.lbl_zoom_val.setObjectName("accentLabel")
        h_zoom.addWidget(self.lbl_zoom_val)
        v_zf.addLayout(h_zoom)

        # Preset buttons for zoom
        h_presets = QHBoxLayout()
        for p in [50, 75, 100, 125, 150, 200]:
            btn = QPushButton(f"{p}%")
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda chk=False, val=p: self.set_zoom(val))
            h_presets.addWidget(btn)
        v_zf.addLayout(h_presets)

        # Flip Mode Radios
        v_zf.addSpacing(4)
        lbl_flip = QLabel("Chế độ lật khung hình:")
        v_zf.addWidget(lbl_flip)

        h_flip = QHBoxLayout()
        self.rb_flip_off = QRadioButton("Bình thường")
        self.rb_flip_h = QRadioButton("Lật ngang ↔")
        self.rb_flip_v = QRadioButton("Lật dọc ↕")
        self.rb_flip_both = QRadioButton("Cả hai ⤡")
        self.rb_flip_off.setChecked(True)

        self.bg_flip = QButtonGroup(self)
        self.bg_flip.addButton(self.rb_flip_off, 0)
        self.bg_flip.addButton(self.rb_flip_h, 1)
        self.bg_flip.addButton(self.rb_flip_v, 2)
        self.bg_flip.addButton(self.rb_flip_both, 3)

        self.rb_flip_off.toggled.connect(self._on_flip_changed)
        self.rb_flip_h.toggled.connect(self._on_flip_changed)
        self.rb_flip_v.toggled.connect(self._on_flip_changed)
        self.rb_flip_both.toggled.connect(self._on_flip_changed)

        h_flip.addWidget(self.rb_flip_off)
        h_flip.addWidget(self.rb_flip_h)
        h_flip.addWidget(self.rb_flip_v)
        h_flip.addWidget(self.rb_flip_both)
        v_zf.addLayout(h_flip)

        v_main.addWidget(grp_zoom_flip)

        # ══════════════════════════════════════════
        # 3. XÓA CHỮ GỐC / LÀM MỜ (Blur Subtitles)
        # ══════════════════════════════════════════
        grp_blur = QGroupBox("🪄 Xóa chữ gốc & Làm mờ phụ đề cũ")
        v_bl = QVBoxLayout(grp_blur)

        self.cb_enable = QCheckBox("<b>Bật làm mờ / che vùng phụ đề cũ</b>")
        self.cb_enable.setChecked(self.settings.enabled)
        self.cb_enable.toggled.connect(self._on_enable_toggled)
        v_bl.addWidget(self.cb_enable)

        # Mode: Blur or Solid
        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("Phương thức:"))
        self.rb_blur = QRadioButton("Làm mờ (Blur)")
        self.rb_blur.setChecked(True)
        self.rb_solid = QRadioButton("Băng che đơn sắc (Solid)")
        bg_mode = QButtonGroup(self)
        bg_mode.addButton(self.rb_blur, 0)
        bg_mode.addButton(self.rb_solid, 1)
        self.rb_blur.toggled.connect(self._on_mode_changed)
        self.rb_solid.toggled.connect(self._on_mode_changed)
        h_mode.addWidget(self.rb_blur)
        h_mode.addWidget(self.rb_solid)
        v_bl.addLayout(h_mode)

        # Coordinates
        h_coords = QHBoxLayout()
        h_coords.addWidget(QLabel("Tọa độ X:"))
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 4000)
        self.spin_x.valueChanged.connect(self._on_geometry_changed)
        h_coords.addWidget(self.spin_x)

        h_coords.addWidget(QLabel("Y:"))
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 4000)
        self.spin_y.setValue(880)
        self.spin_y.valueChanged.connect(self._on_geometry_changed)
        h_coords.addWidget(self.spin_y)
        v_bl.addLayout(h_coords)

        h_size = QHBoxLayout()
        h_size.addWidget(QLabel("Rộng W:"))
        self.spin_w = QSpinBox()
        self.spin_w.setRange(10, 4000)
        self.spin_w.setValue(1920)
        self.spin_w.valueChanged.connect(self._on_geometry_changed)
        h_size.addWidget(self.spin_w)

        h_size.addWidget(QLabel("Cao H:"))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(10, 2000)
        self.spin_h.setValue(160)
        self.spin_h.valueChanged.connect(self._on_geometry_changed)
        h_size.addWidget(self.spin_h)
        v_bl.addLayout(h_size)

        # Blur radius
        self.h_radius_layout = QHBoxLayout()
        self.h_radius_layout.addWidget(QLabel("Độ mờ (Radius):"))
        self.slider_radius = QSlider(Qt.Horizontal)
        self.slider_radius.setRange(1, 100)
        self.slider_radius.setValue(20)
        self.slider_radius.valueChanged.connect(self._on_radius_changed)
        self.h_radius_layout.addWidget(self.slider_radius)
        self.label_rad_val = QLabel("20")
        self.h_radius_layout.addWidget(self.label_rad_val)
        v_bl.addLayout(self.h_radius_layout)

        v_main.addWidget(grp_blur)

        # ══════════════════════════════════════════
        # 4. CHÈN LOGO / WATERMARK VÀO VIDEO
        # ══════════════════════════════════════════
        grp_logo = QGroupBox("🏷️ Chèn Logo / Watermark vào Video")
        v_lg = QVBoxLayout(grp_logo)

        self.cb_logo_enable = QCheckBox("<b>Bật chèn Logo vào video thành phẩm</b>")
        self.cb_logo_enable.setChecked(False)
        self.cb_logo_enable.toggled.connect(self._on_logo_changed)
        v_lg.addWidget(self.cb_logo_enable)

        # Logo file path
        h_logo_file = QHBoxLayout()
        self.edit_logo_path = QLineEdit()
        self.edit_logo_path.setPlaceholderText("Đường dẫn tệp ảnh logo (PNG/JPG)...")
        self.edit_logo_path.textChanged.connect(self._on_logo_changed)
        h_logo_file.addWidget(self.edit_logo_path)

        btn_browse_logo = QPushButton("📁 Chọn ảnh...")
        btn_browse_logo.clicked.connect(self._browse_logo_file)
        h_logo_file.addWidget(btn_browse_logo)
        v_lg.addLayout(h_logo_file)

        # Logo position
        h_pos = QHBoxLayout()
        h_pos.addWidget(QLabel("Vị trí hiển thị:"))
        self.combo_logo_pos = QComboBox()
        self.combo_logo_pos.addItem("Góc trên bên phải (Top-Right)", "top_right")
        self.combo_logo_pos.addItem("Góc trên bên trái (Top-Left)", "top_left")
        self.combo_logo_pos.addItem("Góc dưới bên phải (Bottom-Right)", "bottom_right")
        self.combo_logo_pos.addItem("Góc dưới bên trái (Bottom-Left)", "bottom_left")
        self.combo_logo_pos.currentIndexChanged.connect(self._on_logo_changed)
        h_pos.addWidget(self.combo_logo_pos, stretch=1)
        v_lg.addLayout(h_pos)

        # Logo scale & opacity
        h_lg_scale = QHBoxLayout()
        h_lg_scale.addWidget(QLabel("Kích thước logo (% video):"))
        self.slider_logo_scale = QSlider(Qt.Horizontal)
        self.slider_logo_scale.setRange(4, 30)
        self.slider_logo_scale.setValue(12)
        self.slider_logo_scale.valueChanged.connect(self._on_logo_scale_changed)
        h_lg_scale.addWidget(self.slider_logo_scale)
        self.lbl_logo_scale = QLabel("12%")
        self.lbl_logo_scale.setFixedWidth(40)
        h_lg_scale.addWidget(self.lbl_logo_scale)
        v_lg.addLayout(h_lg_scale)

        h_lg_op = QHBoxLayout()
        h_lg_op.addWidget(QLabel("Độ rõ nét (Opacity):"))
        self.slider_logo_op = QSlider(Qt.Horizontal)
        self.slider_logo_op.setRange(20, 100)
        self.slider_logo_op.setValue(85)
        self.slider_logo_op.valueChanged.connect(self._on_logo_op_changed)
        h_lg_op.addWidget(self.slider_logo_op)
        self.lbl_logo_op = QLabel("85%")
        self.lbl_logo_op.setFixedWidth(40)
        h_lg_op.addWidget(self.lbl_logo_op)
        v_lg.addLayout(h_lg_op)

        lbl_hint_logo = QLabel("💡 Logo hiển thị trực tiếp trên màn hình Xem trước (Preview). Bạn có thể kéo thanh trượt kích thước và độ mờ để căn chỉnh vừa mắt trước khi xuất video.")
        lbl_hint_logo.setWordWrap(True)
        lbl_hint_logo.setStyleSheet("color: #00E5FF; font-size: 11px; margin-top: 4px;")
        v_lg.addWidget(lbl_hint_logo)

        v_main.addWidget(grp_logo)
        v_main.addStretch()

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    # ── Separation Status Handler ──
    def set_separation_status(
        self,
        vocals_path: Optional[str],
        bgm_path: Optional[str],
        method: Optional[str] = None,
    ) -> None:
        self.current_vocals_path = vocals_path or ""
        self.current_bgm_path = bgm_path or ""
        has_both = bool(vocals_path and bgm_path and os.path.exists(vocals_path) and os.path.exists(bgm_path))
        if has_both:
            v_size = os.path.getsize(vocals_path) / (1024 * 1024)
            b_size = os.path.getsize(bgm_path) / (1024 * 1024)
            if method == "demucs":
                self.lbl_sep_status.setText(
                    f"Đã tách bằng Demucs · Giọng {v_size:.1f} MB · Nhạc nền {b_size:.1f} MB"
                )
                self.lbl_sep_status.setObjectName("badgeOk")
            elif method == "filter_approx":
                self.lbl_sep_status.setText(
                    f"Lọc DSP xấp xỉ · có thể còn lẫn giọng/nhạc · {v_size:.1f}/{b_size:.1f} MB"
                )
                self.lbl_sep_status.setObjectName("badgeWarn")
            else:
                self.lbl_sep_status.setText(
                    f"Đã nạp rãnh âm thanh đã lưu · {v_size:.1f}/{b_size:.1f} MB · phương pháp chưa rõ"
                )
                self.lbl_sep_status.setObjectName("badgeOff")
            self.btn_play_vocals.setEnabled(True)
            self.btn_play_bgm.setEnabled(True)
        else:
            self.lbl_sep_status.setText(
                "Chưa có rãnh tách · Demucs cần cài; bộ lọc DSP dự phòng chỉ xấp xỉ"
            )
            self.lbl_sep_status.setObjectName("badgeOff")
            self.btn_play_vocals.setEnabled(False)
            self.btn_play_bgm.setEnabled(False)

        self.lbl_sep_status.style().unpolish(self.lbl_sep_status)
        self.lbl_sep_status.style().polish(self.lbl_sep_status)

    # ── Zoom & Flip Handlers ──
    def set_zoom(self, percent: int) -> None:
        self.slider_zoom.setValue(percent)

    def _on_zoom_slider_changed(self, val: int) -> None:
        self.lbl_zoom_val.setText(f"{val}%")
        self.transform_settings.zoom_percent = val
        self.transform_changed.emit(val, self.transform_settings.flip_mode)

    def _on_flip_changed(self) -> None:
        if self.rb_flip_h.isChecked():
            mode = FlipMode.HORIZONTAL
        elif self.rb_flip_v.isChecked():
            mode = FlipMode.VERTICAL
        elif hasattr(self, "rb_flip_both") and self.rb_flip_both.isChecked():
            mode = FlipMode.BOTH
        else:
            mode = FlipMode.OFF
        self.transform_settings.flip_mode = mode
        self.transform_changed.emit(self.transform_settings.zoom_percent, mode)

    def set_transform_settings(self, settings: TransformSettings) -> None:
        self.transform_settings = settings.model_copy(deep=True)
        controls = (self.slider_zoom, self.rb_flip_off, self.rb_flip_h, self.rb_flip_v, self.rb_flip_both)
        for widget in controls:
            widget.blockSignals(True)
        self.slider_zoom.setValue(settings.zoom_percent)
        {
            FlipMode.OFF: self.rb_flip_off,
            FlipMode.HORIZONTAL: self.rb_flip_h,
            FlipMode.VERTICAL: self.rb_flip_v,
            FlipMode.BOTH: self.rb_flip_both,
        }.get(settings.flip_mode, self.rb_flip_off).setChecked(True)
        self.lbl_zoom_val.setText(f"{settings.zoom_percent}%")
        for widget in controls:
            widget.blockSignals(False)

    def set_logo_settings(self, settings: LogoSettings) -> None:
        self.logo_settings = settings.model_copy(deep=True)
        controls = (
            self.cb_logo_enable, self.edit_logo_path, self.combo_logo_pos,
            self.slider_logo_scale, self.slider_logo_op,
        )
        for widget in controls:
            widget.blockSignals(True)
        self.cb_logo_enable.setChecked(settings.enabled)
        self.edit_logo_path.setText(settings.path)
        pos_index = self.combo_logo_pos.findData(settings.position)
        if pos_index >= 0:
            self.combo_logo_pos.setCurrentIndex(pos_index)
        self.slider_logo_scale.setValue(settings.scale_percent)
        self.slider_logo_op.setValue(int(round(settings.opacity * 100)))
        self.lbl_logo_scale.setText(f"{settings.scale_percent}%")
        self.lbl_logo_op.setText(f"{int(round(settings.opacity * 100))}%")
        for widget in controls:
            widget.blockSignals(False)

    # ── Blur Handlers ──
    def set_settings(self, settings: BlurCoverSettings) -> None:
        self.settings = settings
        self.cb_enable.blockSignals(True)
        self.rb_blur.blockSignals(True)
        self.rb_solid.blockSignals(True)
        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_w.blockSignals(True)
        self.spin_h.blockSignals(True)
        self.slider_radius.blockSignals(True)
        self.cb_enable.setChecked(settings.enabled)
        if settings.mode == CoverMode.BLUR:
            self.rb_blur.setChecked(True)
        else:
            self.rb_solid.setChecked(True)
        self.spin_x.setValue(settings.x)
        self.spin_y.setValue(settings.y)
        self.spin_w.setValue(settings.width)
        self.spin_h.setValue(settings.height)
        self.slider_radius.setValue(settings.blur_radius)
        self.label_rad_val.setText(str(settings.blur_radius))
        self.slider_radius.setEnabled(settings.mode == CoverMode.BLUR)
        for widget in (
            self.cb_enable, self.rb_blur, self.rb_solid,
            self.spin_x, self.spin_y, self.spin_w, self.spin_h, self.slider_radius,
        ):
            widget.blockSignals(False)

    def update_coordinates_from_preview(self, x: int, y: int, w: int, h: int) -> None:
        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_w.blockSignals(True)
        self.spin_h.blockSignals(True)

        self.spin_x.setValue(max(0, x))
        self.spin_y.setValue(max(0, y))
        self.spin_w.setValue(max(10, w))
        self.spin_h.setValue(max(10, h))
        self.settings.x = max(0, x)
        self.settings.y = max(0, y)
        self.settings.width = max(10, w)
        self.settings.height = max(10, h)

        self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(False)
        self.spin_w.blockSignals(False)
        self.spin_h.blockSignals(False)

    def _on_enable_toggled(self, checked: bool) -> None:
        self.settings.enabled = checked
        self.settings_changed.emit(self.settings)

    def _on_mode_changed(self) -> None:
        self.settings.mode = CoverMode.BLUR if self.rb_blur.isChecked() else CoverMode.SOLID
        is_blur = self.rb_blur.isChecked()
        self.slider_radius.setEnabled(is_blur)
        self.settings_changed.emit(self.settings)

    def _on_geometry_changed(self) -> None:
        self.settings.x = self.spin_x.value()
        self.settings.y = self.spin_y.value()
        self.settings.width = self.spin_w.value()
        self.settings.height = self.spin_h.value()
        self.settings_changed.emit(self.settings)

    def _on_radius_changed(self, val: int) -> None:
        self.settings.blur_radius = val
        self.label_rad_val.setText(str(val))
        self.settings_changed.emit(self.settings)

    # ── Logo Handlers ──
    def _browse_logo_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Chọn tệp hình ảnh Logo", "", "Images (*.png *.jpg *.jpeg *.webp *.svg)"
        )
        if f:
            self.edit_logo_path.setText(f)
            self.cb_logo_enable.setChecked(True)
            self._on_logo_changed()

    def _on_logo_scale_changed(self, val: int) -> None:
        self.lbl_logo_scale.setText(f"{val}%")
        self.logo_settings.scale_percent = val
        self._on_logo_changed()

    def _on_logo_op_changed(self, val: int) -> None:
        self.lbl_logo_op.setText(f"{val}%")
        self.logo_settings.opacity = val / 100.0
        self._on_logo_changed()

    def _on_logo_changed(self) -> None:
        self.logo_settings.enabled = self.cb_logo_enable.isChecked()
        self.logo_settings.path = self.edit_logo_path.text().strip()
        self.logo_settings.position = self.combo_logo_pos.currentData() or "top_right"
        self.logo_settings.scale_percent = self.slider_logo_scale.value()
        self.logo_settings.opacity = self.slider_logo_op.value() / 100.0
        self.logo_changed.emit(self.logo_settings)
