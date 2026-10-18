"""Application settings and API configuration dialog."""

import os
import shutil
from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, QMessageBox, QGroupBox,
    QProgressBar, QSlider, QSpinBox, QCheckBox, QColorDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QColor
from ...app.settings import AppSettings
from ...services.keychain_service import KeychainService
from ...services.cache_service import CacheService
from ...services.nine_router_service import NineRouterService
from ...ai.firered_asr import FireRedAsrManager
from ...tts.model_manager import VieNeuModelManager
from ...domain.enums import LanguageCode
from ..i18n import tr, I18n


def ass_to_hex(ass_col: str) -> str:
    """Convert ASS format (&HAABBGGRR) to standard RGB hex (#RRGGBB)."""
    c = ass_col.replace("&H", "").replace("&h", "")
    if len(c) == 8:
        b, g, r = c[2:4], c[4:6], c[6:8]
        return f"#{r}{g}{b}".upper()
    elif len(c) == 6:
        b, g, r = c[0:2], c[2:4], c[4:6]
        return f"#{r}{g}{b}".upper()
    return "#FFFFFF"


def hex_to_ass(hex_col: str) -> str:
    """Convert standard RGB hex (#RRGGBB) to ASS format (&H00BBGGRR)."""
    h = hex_col.lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}".upper()
    return "&H00FFFFFF"


class ModelDownloadWorker(QThread):
    progress = Signal(int, str)
    finished_success = Signal()
    failed = Signal(str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager

    def run(self):
        try:
            self.manager.download_model(
                progress_callback=lambda p, msg: self.progress.emit(p, msg)
            )
            self.finished_success.emit()
        except Exception as e:
            self.failed.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: AppSettings,
        keychain: KeychainService,
        cache_service: Optional[CacheService] = None,
        initial_tab: int = 0,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.settings = settings
        self.keychain = keychain
        self.cache_service = cache_service
        self.initial_tab = initial_tab
        self._selected_vieneu_dir = self.settings.vieneu_model_dir or ""
        self.model_manager = VieNeuModelManager(custom_path=self._selected_vieneu_dir or None)
        self.firered_manager = FireRedAsrManager(self.settings.firered_model_dir)
        self._download_worker: Optional[ModelDownloadWorker] = None

        self.current_primary_hex = ass_to_hex(self.settings.primary_color)
        self.current_outline_hex = ass_to_hex(self.settings.outline_color)

        self.setWindowTitle(tr("settings.title") + " - vTranslate Studio")
        self.resize(700, 620)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        tabs = QTabWidget()

        # ══════════════════════════════════════════
        # Tab 1: Chung & Dọn dẹp Cache (General)
        # ══════════════════════════════════════════
        tab_general = QWidget()
        v_gen = QVBoxLayout(tab_general)
        v_gen.setSpacing(12)

        grp_app = QGroupBox("Cấu hình Dự án & Giao diện")
        v_app = QVBoxLayout(grp_app)

        h_lang = QHBoxLayout()
        h_lang.addWidget(QLabel(tr("settings.language_label")))
        self.combo_lang = QComboBox()
        self.combo_lang.addItem("Tiếng Việt (Mặc định)", LanguageCode.VI)
        self.combo_lang.addItem("English", LanguageCode.EN)
        self.combo_lang.addItem("日本語", LanguageCode.JA)
        idx_l = self.combo_lang.findData(self.settings.language)
        if idx_l >= 0:
            self.combo_lang.setCurrentIndex(idx_l)
        h_lang.addWidget(self.combo_lang)
        v_app.addLayout(h_lang)

        h_out = QHBoxLayout()
        h_out.addWidget(QLabel(tr("settings.output_dir_label")))
        self.edit_output_dir = QLineEdit(self.settings.output_dir)
        h_out.addWidget(self.edit_output_dir)
        btn_browse_out = QPushButton(tr("settings.browse"))
        btn_browse_out.clicked.connect(self._browse_output_dir)
        h_out.addWidget(btn_browse_out)
        v_app.addLayout(h_out)

        v_gen.addWidget(grp_app)

        # Quản lý & Xoá Cache
        grp_cache = QGroupBox("Quản lý Bộ nhớ đệm (Cache & Temporary Files)")
        v_cache = QVBoxLayout(grp_cache)

        lbl_cache_info = QLabel(
            "Bộ nhớ đệm lưu kết quả phân tích, âm thanh TTS và các tệp trung gian. "
            "Dọn dẹp sẽ xóa các tệp cache để giải phóng dung lượng; lần xử lý sau sẽ tạo lại dữ liệu cần thiết."
        )
        lbl_cache_info.setWordWrap(True)
        lbl_cache_info.setObjectName("mutedLabel")
        v_cache.addWidget(lbl_cache_info)

        h_cache_act = QHBoxLayout()
        self.btn_clear_cache_dialog = QPushButton("🗑️ Xoá sạch toàn bộ Cache ngay")
        self.btn_clear_cache_dialog.setObjectName("dangerButton")
        self.btn_clear_cache_dialog.setFixedHeight(34)
        self.btn_clear_cache_dialog.clicked.connect(self._on_clear_cache_clicked)
        h_cache_act.addWidget(self.btn_clear_cache_dialog)
        h_cache_act.addStretch()
        v_cache.addLayout(h_cache_act)

        v_gen.addWidget(grp_cache)
        v_gen.addStretch()
        tabs.addTab(tab_general, "⚙️ Chung & Cache")

        # ══════════════════════════════════════════
        # Tab 2: Trí tuệ nhân tạo (AI & APIs)
        # ══════════════════════════════════════════
        tab_ai = QWidget()
        v_ai = QVBoxLayout(tab_ai)
        v_ai.setSpacing(10)

        # 1. 9router Local & AI Model
        grp_trans = QGroupBox("9router Local & AI Dịch thuật / Kịch bản")
        v_t = QVBoxLayout(grp_trans)

        h_router_top = QHBoxLayout()
        btn_fill_9router = QPushButton("⚡ Tự động kết nối 9router Local")
        btn_fill_9router.setObjectName("accentButton")
        btn_fill_9router.clicked.connect(self._auto_fill_9router)
        h_router_top.addWidget(btn_fill_9router)

        self.lbl_9router_status = QLabel("")
        self.lbl_9router_status.setObjectName("badgeOff")
        h_router_top.addWidget(self.lbl_9router_status)
        h_router_top.addStretch()
        v_t.addLayout(h_router_top)

        h_model = QHBoxLayout()
        h_model.addWidget(QLabel("Mô hình AI:"))
        self.combo_ai_model = QComboBox()
        common_models = [
            ("ag/gemini-3.8-flash-low (9router)", "ag/gemini-3.8-flash-low"),
            ("ag/gemini-3.8-flash (9router)", "ag/gemini-3.8-flash"),
            ("ag/gemini-3.8-flash-medium (9router)", "ag/gemini-3.8-flash-medium"),
            ("ag/gemini-3.7-flash-low (9router)", "ag/gemini-3.7-flash-low"),
            ("cx/gpt-5.6-terra (9router)", "cx/gpt-5.6-terra"),
            ("cx/gpt-5.6-luna (9router)", "cx/gpt-5.6-luna"),
            ("gemini-3.5-flash-lite (Direct Gemini API)", "gemini-3.5-flash-lite"),
            ("tokenharbor/deepseek-v4.1-flash:free (9router)", "tokenharbor/deepseek-v4.1-flash:free"),
        ]
        for label, m in common_models:
            self.combo_ai_model.addItem(label, m)

        # Tìm model hiện tại
        curr_m = self.settings.vision_model
        matched_idx = -1
        for i in range(self.combo_ai_model.count()):
            if self.combo_ai_model.itemData(i) == curr_m or self.combo_ai_model.itemText(i) == curr_m:
                matched_idx = i
                break
        if matched_idx >= 0:
            self.combo_ai_model.setCurrentIndex(matched_idx)
        else:
            self.combo_ai_model.addItem(curr_m, curr_m)
            self.combo_ai_model.setCurrentIndex(self.combo_ai_model.count() - 1)

        h_model.addWidget(self.combo_ai_model, stretch=1)

        self.btn_test_model = QPushButton("⚡ Đo tốc độ")
        self.btn_test_model.clicked.connect(self._test_ai_model_speed)
        h_model.addWidget(self.btn_test_model)

        self.lbl_model_test_status = QLabel("")
        h_model.addWidget(self.lbl_model_test_status)
        v_t.addLayout(h_model)

        h_url = QHBoxLayout()
        h_url.addWidget(QLabel(tr("settings.api_endpoint")))
        self.edit_trans_url = QLineEdit(self.keychain.get_secret("translation_api_url") or "http://127.0.0.1:20128/v1")
        h_url.addWidget(self.edit_trans_url)
        v_t.addLayout(h_url)

        h_key = QHBoxLayout()
        h_key.addWidget(QLabel(tr("settings.api_key") + ":"))
        self.edit_trans_key = QLineEdit()
        self.edit_trans_key.setEchoMode(QLineEdit.Password)
        self.edit_trans_key.setText(self.keychain.get_secret("translation_api_key"))
        h_key.addWidget(self.edit_trans_key)
        v_t.addLayout(h_key)
        v_ai.addWidget(grp_trans)

        # 2. Speech-to-Text ASR
        grp_asr = QGroupBox(tr("settings.stt"))
        v_asr = QVBoxLayout(grp_asr)

        h_engine = QHBoxLayout()
        h_engine.addWidget(QLabel(tr("settings.asr_engine")))
        self.combo_asr_engine = QComboBox()
        self.combo_asr_engine.addItem(tr("settings.engine_firered"), "firered")
        self.combo_asr_engine.addItem(tr("settings.engine_whisper"), "whisper")
        idx_engine = self.combo_asr_engine.findData(self.settings.asr_engine)
        if idx_engine >= 0:
            self.combo_asr_engine.setCurrentIndex(idx_engine)
        h_engine.addWidget(self.combo_asr_engine, stretch=1)
        v_asr.addLayout(h_engine)

        h_fr_dl = QHBoxLayout()
        self.btn_download_firered = QPushButton(tr("settings.firered_download"))
        self.btn_download_firered.setObjectName("accentButton")
        self.btn_download_firered.clicked.connect(self._download_firered_model)
        h_fr_dl.addWidget(self.btn_download_firered)
        self.lbl_firered_status = QLabel("")
        self.lbl_firered_status.setObjectName("mutedLabel")
        h_fr_dl.addWidget(self.lbl_firered_status, stretch=1)
        v_asr.addLayout(h_fr_dl)

        self.edit_firered_dir = QLineEdit(self.settings.firered_model_dir)
        self.edit_firered_dir.setVisible(False)

        self.prog_firered = QProgressBar()
        self.prog_firered.setRange(0, 100)
        self.prog_firered.setVisible(False)
        v_asr.addWidget(self.prog_firered)
        v_ai.addWidget(grp_asr)

        v_ai.addStretch()
        tabs.addTab(tab_ai, "🤖 Trí tuệ nhân tạo (AI)")

        # ══════════════════════════════════════════
        # Tab 3: Lồng tiếng & Âm lượng (TTS & Audio)
        # ══════════════════════════════════════════
        tab_audio = QWidget()
        v_aud = QVBoxLayout(tab_audio)
        v_aud.setSpacing(12)

        # Local cloned-voice TTS
        grp_tts = QGroupBox("✨ Lồng tiếng cục bộ bằng VieNeu")
        v_tts = QVBoxLayout(grp_tts)
        v_tts.setSpacing(8)
        lbl_tts_requirements = QLabel(
            "VieNeu-TTS v3 Turbo ONNX chạy ngay trên máy và clone từ 10 mẫu WAV đi kèm; không gọi Gemini TTS hay cần API key. "
            "Có thể chọn sắc thái theo từng câu trên bảng phụ đề; mức thể hiện phụ thuộc mô hình."
        )
        lbl_tts_requirements.setWordWrap(True)
        lbl_tts_requirements.setObjectName("mutedLabel")
        v_tts.addWidget(lbl_tts_requirements)
        v_aud.addWidget(grp_tts)

        grp_vn = QGroupBox(tr("settings.vieneu_title"))
        v_g = QVBoxLayout(grp_vn)
        self.lbl_vn_info = QLabel("")
        self.lbl_vn_info.setObjectName("mutedLabel")
        v_g.addWidget(self.lbl_vn_info)

        self.prog_vieneu = QProgressBar()
        self.prog_vieneu.setRange(0, 100)
        self.prog_vieneu.setVisible(False)
        v_g.addWidget(self.prog_vieneu)

        h_vn_path = QHBoxLayout()
        self.edit_vieneu_dir = QLineEdit(self._selected_vieneu_dir)
        self.edit_vieneu_dir.setReadOnly(True)
        self.edit_vieneu_dir.setPlaceholderText("Để trống để tự tìm mô hình VieNeu đã cài")
        h_vn_path.addWidget(self.edit_vieneu_dir, stretch=1)
        btn_choose_vn = QPushButton(tr("settings.select_vieneu_dir"))
        btn_choose_vn.clicked.connect(self._browse_vieneu_dir)
        h_vn_path.addWidget(btn_choose_vn)
        v_g.addLayout(h_vn_path)

        self.btn_download_vieneu = QPushButton("Tải VieNeu-TTS v3 Turbo (ONNX · khoảng 640 MB)")
        self.btn_download_vieneu.setObjectName("accentButton")
        self.btn_download_vieneu.clicked.connect(self._download_vieneu_model)
        v_g.addWidget(self.btn_download_vieneu)
        v_aud.addWidget(grp_vn)

        grp_mix = QGroupBox("Hòa âm Lồng tiếng & Tốc độ nói")
        v_mix = QVBoxLayout(grp_mix)

        # Âm lượng gốc
        h_orig = QHBoxLayout()
        h_orig.addWidget(QLabel("Âm lượng video gốc:"))
        self.slider_orig_vol = QSlider(Qt.Horizontal)
        self.slider_orig_vol.setRange(0, 100)
        self.slider_orig_vol.setValue(int(self.settings.original_volume * 100))
        self.lbl_orig_val = QLabel(f"{int(self.settings.original_volume * 100)}%")
        self.lbl_orig_val.setFixedWidth(45)
        self.slider_orig_vol.valueChanged.connect(lambda v: self.lbl_orig_val.setText(f"{v}%"))
        h_orig.addWidget(self.slider_orig_vol)
        h_orig.addWidget(self.lbl_orig_val)
        v_mix.addLayout(h_orig)

        # Âm lượng lồng tiếng AI
        h_dub = QHBoxLayout()
        h_dub.addWidget(QLabel("Âm lượng thuyết minh AI:"))
        self.slider_dub_vol = QSlider(Qt.Horizontal)
        self.slider_dub_vol.setRange(50, 200)
        self.slider_dub_vol.setValue(int(self.settings.dubbing_volume * 100))
        self.lbl_dub_val = QLabel(f"{int(self.settings.dubbing_volume * 100)}%")
        self.lbl_dub_val.setFixedWidth(45)
        self.slider_dub_vol.valueChanged.connect(lambda v: self.lbl_dub_val.setText(f"{v}%"))
        h_dub.addWidget(self.slider_dub_vol)
        h_dub.addWidget(self.lbl_dub_val)
        v_mix.addLayout(h_dub)

        # Ngưỡng co giãn tốc độ tối đa
        h_tempo = QHBoxLayout()
        h_tempo.addWidget(QLabel("Giới hạn tăng tốc tối đa (Fit-to-Slot):"))
        self.slider_tempo = QSlider(Qt.Horizontal)
        self.slider_tempo.setRange(100, 140)
        self.slider_tempo.setValue(int(self.settings.max_tempo_stretch * 100))
        self.lbl_tempo_val = QLabel(f"{self.settings.max_tempo_stretch:.2f}x")
        self.lbl_tempo_val.setFixedWidth(45)
        self.slider_tempo.valueChanged.connect(lambda v: self.lbl_tempo_val.setText(f"{v/100:.2f}x"))
        h_tempo.addWidget(self.slider_tempo)
        h_tempo.addWidget(self.lbl_tempo_val)
        v_mix.addLayout(h_tempo)

        lbl_tempo_hint = QLabel("💡 Khuyên dùng 1.15x - 1.18x: Giọng lồng tiếng nhanh nhẹn, tự nhiên, rõ từng chữ. Tránh đặt > 1.25x vì sẽ gây méo tiếng như sóc chuột.")
        lbl_tempo_hint.setWordWrap(True)
        lbl_tempo_hint.setStyleSheet("color: #888888; font-size: 11px; margin-top: 2px;")
        v_mix.addWidget(lbl_tempo_hint)

        v_aud.addWidget(grp_mix)
        v_aud.addStretch()
        tabs.addTab(tab_audio, "🎙️ Lồng tiếng & Âm thanh")

        # ══════════════════════════════════════════
        # Tab 4: Kiểu dáng & Mẫu Phông Chữ (Subtitle Style & Live Preview)
        # ══════════════════════════════════════════
        tab_sub = QWidget()
        v_sub = QVBoxLayout(tab_sub)
        v_sub.setSpacing(10)

        grp_style = QGroupBox("Cấu hình Phông chữ Phụ đề Tiếng Việt")
        v_st = QVBoxLayout(grp_style)

        # 1. Phông chữ đa dạng & Cỡ chữ
        h_font = QHBoxLayout()
        h_font.addWidget(QLabel("Phông chữ:"))
        self.combo_font = QComboBox()

        curated_fonts = [
            "Arial", "Roboto", "Be Vietnam Pro", "Montserrat", "Open Sans",
            "Noto Sans", "Inter", "Segoe UI", "Tahoma", "Verdana",
            "Oswald", "Playfair Display", "Times New Roman", "Impact", "Comic Sans MS"
        ]
        all_families = QFontDatabase.families()
        seen_fonts = set()
        clean_fonts = []
        for f in curated_fonts:
            if f not in seen_fonts:
                clean_fonts.append(f)
                seen_fonts.add(f)
        for f in all_families:
            if f not in seen_fonts and not f.startswith("."):
                clean_fonts.append(f)
                seen_fonts.add(f)

        for f in clean_fonts:
            self.combo_font.addItem(f, f)

        idx_f = self.combo_font.findText(self.settings.font_name)
        if idx_f >= 0:
            self.combo_font.setCurrentIndex(idx_f)
        self.combo_font.currentIndexChanged.connect(self._update_font_preview)
        h_font.addWidget(self.combo_font, stretch=1)

        h_font.addWidget(QLabel("Cỡ chữ:"))
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(14, 52)
        self.spin_font_size.setValue(self.settings.font_size)
        self.spin_font_size.valueChanged.connect(self._update_font_preview)
        h_font.addWidget(self.spin_font_size)
        v_st.addLayout(h_font)

        # 2. Màu chữ & Màu viền chữ (Color Pickers)
        h_colors = QHBoxLayout()
        h_colors.addWidget(QLabel("Màu chữ chính:"))
        self.btn_pick_text_color = QPushButton(f"■ {self.current_primary_hex}")
        self.btn_pick_text_color.setStyleSheet(
            f"background-color: {self.current_primary_hex}; color: #000; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_pick_text_color.clicked.connect(self._pick_text_color)
        h_colors.addWidget(self.btn_pick_text_color)

        h_colors.addWidget(QLabel("Màu viền chữ:"))
        self.btn_pick_outline_color = QPushButton(f"■ {self.current_outline_hex}")
        self.btn_pick_outline_color.setStyleSheet(
            f"background-color: {self.current_outline_hex}; color: #FFF; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_pick_outline_color.clicked.connect(self._pick_outline_color)
        h_colors.addWidget(self.btn_pick_outline_color)
        v_st.addLayout(h_colors)

        # 3. Độ dày viền chữ & Lề dưới
        h_margin = QHBoxLayout()
        h_margin.addWidget(QLabel("Độ dày viền đen (Outline):"))
        self.spin_outline = QSpinBox()
        self.spin_outline.setRange(0, 8)
        self.spin_outline.setValue(self.settings.outline_width)
        self.spin_outline.valueChanged.connect(self._update_font_preview)
        h_margin.addWidget(self.spin_outline)

        h_margin.addWidget(QLabel("Khoảng cách lề (px):"))
        self.spin_margin_v = QSpinBox()
        self.spin_margin_v.setRange(10, 160)
        self.spin_margin_v.setValue(self.settings.margin_v)
        h_margin.addWidget(self.spin_margin_v)
        v_st.addLayout(h_margin)

        # 4. Vị trí hiển thị (Alignment) & Tự động gom thành 1 dòng
        h_pos = QHBoxLayout()
        h_pos.addWidget(QLabel("Vị trí hiển thị:"))
        self.combo_alignment = QComboBox()
        self.combo_alignment.addItem("Đáy màn hình (Bottom Center - Chuẩn)", 2)
        self.combo_alignment.addItem("Giữa màn hình (Middle Center)", 5)
        self.combo_alignment.addItem("Đỉnh màn hình (Top Center)", 8)
        curr_align = getattr(self.settings, "alignment", 2)
        idx_align = self.combo_alignment.findData(curr_align)
        if idx_align >= 0:
            self.combo_alignment.setCurrentIndex(idx_align)
        h_pos.addWidget(self.combo_alignment, stretch=1)
        v_st.addLayout(h_pos)

        self.cb_single_line = QCheckBox("Tự động gom phụ đề thành 1 dòng duy nhất (Single-line - Loại bỏ ngắt dòng)")
        self.cb_single_line.setChecked(getattr(self.settings, "single_line", True))
        v_st.addWidget(self.cb_single_line)

        v_sub.addWidget(grp_style)

        # 4. Live Visual Subtitle Preview Box
        grp_preview = QGroupBox("👁️ Khung Xem Trước Mẫu Phông Chữ Trực Quan")
        v_pv = QVBoxLayout(grp_preview)

        self.preview_container = QWidget()
        self.preview_container.setFixedHeight(110)
        self.preview_container.setStyleSheet(
            "background-color: #0E0F14; border: 2px solid #282B37; border-radius: 8px;"
        )
        v_pc = QVBoxLayout(self.preview_container)
        v_pc.setAlignment(Qt.AlignCenter)

        self.lbl_font_preview = QLabel("Đây là mẫu phụ đề tiếng Việt (Sample Subtitle)")
        self.lbl_font_preview.setAlignment(Qt.AlignCenter)
        self.lbl_font_preview.setWordWrap(True)
        v_pc.addWidget(self.lbl_font_preview)

        v_pv.addWidget(self.preview_container)
        v_sub.addWidget(grp_preview)

        v_sub.addStretch()
        tabs.addTab(tab_sub, "🎨 Kiểu dáng Phụ đề")

        # ══════════════════════════════════════════
        # Tab 5: Xuất video & FFmpeg (Render)
        # ══════════════════════════════════════════
        tab_ffmpeg = QWidget()
        v_ff = QVBoxLayout(tab_ffmpeg)
        v_ff.setSpacing(12)

        grp_render = QGroupBox("Cấu hình Xuất Video (MP4)")
        v_rn = QVBoxLayout(grp_render)

        h_preset = QHBoxLayout()
        h_preset.addWidget(QLabel("Tốc độ mã hóa (Preset):"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["ultrafast", "fast", "medium", "slow"])
        idx_p = self.combo_preset.findText(self.settings.render_preset)
        if idx_p >= 0:
            self.combo_preset.setCurrentIndex(idx_p)
        h_preset.addWidget(self.combo_preset, stretch=1)

        h_preset.addWidget(QLabel("Chất lượng CRF:"))
        self.spin_crf = QSpinBox()
        self.spin_crf.setRange(16, 28)
        self.spin_crf.setValue(self.settings.render_crf)
        h_preset.addWidget(self.spin_crf)
        v_rn.addLayout(h_preset)

        self.cb_burn_sub = QCheckBox("Tự động chèn phụ đề cứng tiếng Việt vào video xuất")
        self.cb_burn_sub.setChecked(self.settings.burn_subtitles)
        v_rn.addWidget(self.cb_burn_sub)

        v_ff.addWidget(grp_render)

        grp_bin = QGroupBox("Đường dẫn Hệ thống FFmpeg")
        v_b = QVBoxLayout(grp_bin)

        h_ff = QHBoxLayout()
        h_ff.addWidget(QLabel("FFmpeg Path:"))
        self.edit_ffmpeg = QLineEdit(self.settings.ffmpeg_path)
        h_ff.addWidget(self.edit_ffmpeg)
        btn_test_ff = QPushButton(tr("settings.test_connection"))
        btn_test_ff.clicked.connect(self._test_ffmpeg)
        h_ff.addWidget(btn_test_ff)
        v_b.addLayout(h_ff)

        h_ffp = QHBoxLayout()
        h_ffp.addWidget(QLabel("FFprobe Path:"))
        self.edit_ffprobe = QLineEdit(self.settings.ffprobe_path)
        h_ffp.addWidget(self.edit_ffprobe)
        v_b.addLayout(h_ffp)

        self.label_ff_status = QLabel("")
        self.label_ff_status.setObjectName("mutedLabel")
        v_b.addWidget(self.label_ff_status)
        v_ff.addWidget(grp_bin)

        v_ff.addStretch()
        tabs.addTab(tab_ffmpeg, "🎬 Render & FFmpeg")

        layout.addWidget(tabs)
        self.tabs = tabs
        if self.initial_tab > 0 and self.initial_tab < tabs.count():
            tabs.setCurrentIndex(self.initial_tab)
        self._refresh_firered_status()
        self._refresh_vieneu_status()
        self._auto_fill_9router(silent=True)
        self._update_font_preview()

        # Dialog buttons
        h_btns = QHBoxLayout()
        h_btns.addStretch()
        btn_cancel = QPushButton(tr("common.cancel"))
        btn_cancel.clicked.connect(self.reject)
        h_btns.addWidget(btn_cancel)

        btn_save = QPushButton(" Lưu cài đặt")
        btn_save.setObjectName("primaryButton")
        btn_save.setFixedHeight(34)
        btn_save.clicked.connect(self._save_settings)
        h_btns.addWidget(btn_save)
        layout.addLayout(h_btns)

    def _browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, tr("settings.output_dir_label"), self.edit_output_dir.text()
        )
        if folder:
            self.edit_output_dir.setText(folder)

    def _on_clear_cache_clicked(self) -> None:
        ret = QMessageBox.question(
            self,
            "Xác nhận xoá Cache",
            "Bạn có chắc chắn muốn dọn dẹp và xoá sạch toàn bộ bộ nhớ đệm?\n\n"
            "Các bản dịch tạm, âm thanh TTS đã sinh và ảnh cắt tạm thời sẽ được dọn dẹp sạch sẽ.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if ret == QMessageBox.Yes:
            if self.cache_service:
                self.cache_service.clear_all_cache([self.settings.temp_dir])
            else:
                import sqlite3
                p = os.path.expanduser("~/.vtranslator/vtranslator.db")
                if os.path.exists(p):
                    c = sqlite3.connect(p)
                    with c:
                        c.execute("DELETE FROM analysis_cache;")
                    c.close()
                for d in ["~/.vtranslator/tts_cache", "~/.vtranslator/temp", "~/.vtranslator/avatars", "analysis"]:
                    exp = os.path.expanduser(d)
                    if os.path.exists(exp):
                        shutil.rmtree(exp, ignore_errors=True)
                        os.makedirs(exp, exist_ok=True)
            QMessageBox.information(self, "Thành công", "✅ Đã xoá sạch toàn bộ bộ nhớ đệm thành công!")

    def _refresh_firered_status(self) -> None:
        self.firered_manager.model_dir = self.edit_firered_dir.text().strip()
        info = self.firered_manager.get_info()
        if info["installed"]:
            self.lbl_firered_status.setText(
                tr("settings.firered_ready", size=info["size_mb"], path=info["model_dir"])
            )
            self.lbl_firered_status.setObjectName("statusOk")
            self.btn_download_firered.setEnabled(False)
        else:
            self.lbl_firered_status.setText(tr("settings.firered_missing"))
            self.lbl_firered_status.setObjectName("statusWarn")
            self.btn_download_firered.setEnabled(True)

        self.lbl_firered_status.style().unpolish(self.lbl_firered_status)
        self.lbl_firered_status.style().polish(self.lbl_firered_status)

    def _download_firered_model(self) -> None:
        self.btn_download_firered.setEnabled(False)
        self.prog_firered.setVisible(True)
        self.prog_firered.setValue(0)
        self.lbl_firered_status.setText("Downloading FireRedASR2 model...")

        self.firered_manager.model_dir = self.edit_firered_dir.text().strip()
        self._download_worker = ModelDownloadWorker(self.firered_manager, self)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished_success.connect(self._on_download_success)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_progress(self, percent: int, msg: str) -> None:
        self.prog_firered.setValue(percent)
        self.lbl_firered_status.setText(msg)

    def _on_download_success(self) -> None:
        self.prog_firered.setVisible(False)
        self._refresh_firered_status()
        QMessageBox.information(self, tr("common.success"), "FireRedASR2 model downloaded successfully!")

    def _on_download_failed(self, err: str) -> None:
        self.prog_firered.setVisible(False)
        self.btn_download_firered.setEnabled(True)
        self._refresh_firered_status()
        QMessageBox.critical(self, tr("common.error"), f"Download failed:\n{err}")

    def _auto_fill_9router(self, silent: bool = False) -> None:
        ok, msg, models = NineRouterService.check_connection()
        key = NineRouterService.find_local_key()
        self.edit_trans_url.setText(NineRouterService.DEFAULT_URL)
        if key:
            self.edit_trans_key.setText(key)
        if ok:
            self.lbl_9router_status.setText(f"Online ({len(models)} models)")
            self.lbl_9router_status.setObjectName("badgeOk")
            if models:
                curr_data = self.combo_ai_model.currentData() or self.combo_ai_model.currentText()
                self.combo_ai_model.clear()
                # Direct Gemini options
                if NineRouterService.find_direct_gemini_key():
                    self.combo_ai_model.addItem("gemini-3.5-flash-lite (Direct Gemini API - 1.5s)", "gemini-3.5-flash-lite")
                    self.combo_ai_model.addItem("gemini-flash-lite-latest (Direct Gemini API - 2s)", "gemini-flash-lite-latest")
                for m in models:
                    self.combo_ai_model.addItem(m, m)
                idx = -1
                for i in range(self.combo_ai_model.count()):
                    if self.combo_ai_model.itemData(i) == curr_data or self.combo_ai_model.itemText(i) == curr_data:
                        idx = i
                        break
                if idx >= 0:
                    self.combo_ai_model.setCurrentIndex(idx)
                else:
                    self.combo_ai_model.setCurrentIndex(0)
        else:
            self.lbl_9router_status.setText("Offline")
            self.lbl_9router_status.setObjectName("badgeOff")
        self.lbl_9router_status.style().unpolish(self.lbl_9router_status)
        self.lbl_9router_status.style().polish(self.lbl_9router_status)
        if not silent and ok:
            QMessageBox.information(self, "9router Kết nối", f"Đã kết nối thành công tới 9router!\nTìm thấy {len(models)} mô hình AI.")

    def _test_ai_model_speed(self) -> None:
        model = self.combo_ai_model.currentData() or self.combo_ai_model.currentText().strip()
        if not model:
            return
        self.lbl_model_test_status.setText("⏳ Đang đo...")
        self.btn_test_model.setEnabled(False)

        def worker():
            ok, lat, msg = NineRouterService.test_model_speed(
                model,
                url=self.edit_trans_url.text().strip(),
                api_key=self.edit_trans_key.text().strip()
            )
            def on_done():
                self.btn_test_model.setEnabled(True)
                self.lbl_model_test_status.setText(msg)
                self.lbl_model_test_status.setObjectName("statusOk" if ok else "statusError")
                self.lbl_model_test_status.style().unpolish(self.lbl_model_test_status)
                self.lbl_model_test_status.style().polish(self.lbl_model_test_status)
            QTimer.singleShot(0, on_done)

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def _refresh_vieneu_status(self) -> None:
        info = self.model_manager.get_info()
        if info["loaded"]:
            self.lbl_vn_info.setText(f"{info['status_text']}\n{info['model_dir']}")
            self.lbl_vn_info.setObjectName("statusOk")
            self._selected_vieneu_dir = info["model_dir"]
            self.edit_vieneu_dir.setText(info["model_dir"])
            self.btn_download_vieneu.setEnabled(False)
        else:
            self.lbl_vn_info.setText(tr("settings.vieneu_missing"))
            self.lbl_vn_info.setObjectName("statusWarn")
            self.btn_download_vieneu.setEnabled(True)
        self.lbl_vn_info.style().unpolish(self.lbl_vn_info)
        self.lbl_vn_info.style().polish(self.lbl_vn_info)

    def _download_vieneu_model(self) -> None:
        self.btn_download_vieneu.setEnabled(False)
        self.prog_vieneu.setVisible(True)
        self.prog_vieneu.setValue(0)
        self.lbl_vn_info.setText("Đang tải VieNeu-TTS v3 Turbo; ứng dụng vẫn dùng được.")
        self._download_worker = ModelDownloadWorker(self.model_manager, self)
        self._download_worker.progress.connect(self._on_vieneu_download_progress)
        self._download_worker.finished_success.connect(self._on_vieneu_download_success)
        self._download_worker.failed.connect(self._on_vieneu_download_failed)
        self._download_worker.start()

    def _on_vieneu_download_progress(self, percent: int, message: str) -> None:
        self.prog_vieneu.setValue(percent)
        self.lbl_vn_info.setText(message)

    def _on_vieneu_download_success(self) -> None:
        self.prog_vieneu.setVisible(False)
        self._selected_vieneu_dir = self.model_manager.model_dir or str(self.model_manager.MODEL_DIR)
        self.settings.vieneu_model_dir = self._selected_vieneu_dir
        self.edit_vieneu_dir.setText(self._selected_vieneu_dir)
        self._refresh_vieneu_status()

    def _on_vieneu_download_failed(self, error: str) -> None:
        self.prog_vieneu.setVisible(False)
        self.btn_download_vieneu.setEnabled(True)
        self._refresh_vieneu_status()
        QMessageBox.critical(self, tr("common.error"), f"Không tải được mô hình VieNeu:\n{error}")

    def _browse_vieneu_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, tr("settings.select_vieneu_dir"), self.edit_vieneu_dir.text()
        )
        if folder:
            ok, msg = self.model_manager.set_custom_dir(folder)
            self._refresh_vieneu_status()
            if ok:
                self._selected_vieneu_dir = folder
                self.edit_vieneu_dir.setText(folder)
                QMessageBox.information(self, tr("common.success"), msg)
            else:
                QMessageBox.warning(self, tr("common.error"), msg)

    def _test_ffmpeg(self) -> None:
        ffmpeg = shutil.which(self.edit_ffmpeg.text().strip())
        ffprobe = shutil.which(self.edit_ffprobe.text().strip())
        if ffmpeg and ffprobe:
            self.label_ff_status.setText(f"FFmpeg: {ffmpeg}\nFFprobe: {ffprobe}")
            self.label_ff_status.setObjectName("statusOk")
        else:
            missing = []
            if not ffmpeg:
                missing.append("FFmpeg")
            if not ffprobe:
                missing.append("FFprobe")
            self.label_ff_status.setText("Không tìm thấy: " + ", ".join(missing))
            self.label_ff_status.setObjectName("statusError")
        self.label_ff_status.style().unpolish(self.label_ff_status)
        self.label_ff_status.style().polish(self.label_ff_status)

    def _pick_text_color(self) -> None:
        col = QColorDialog.getColor(QColor(self.current_primary_hex), self, "Chọn màu chữ phụ đề")
        if col.isValid():
            self.current_primary_hex = col.name().upper()
            btn_fg = "#000000" if col.lightness() > 128 else "#FFFFFF"
            self.btn_pick_text_color.setStyleSheet(
                f"background-color: {self.current_primary_hex}; color: {btn_fg}; font-weight: bold; padding: 4px 10px;"
            )
            self.btn_pick_text_color.setText(f"■ {self.current_primary_hex}")
            self._update_font_preview()

    def _pick_outline_color(self) -> None:
        col = QColorDialog.getColor(QColor(self.current_outline_hex), self, "Chọn màu viền chữ phụ đề")
        if col.isValid():
            self.current_outline_hex = col.name().upper()
            btn_fg = "#000000" if col.lightness() > 128 else "#FFFFFF"
            self.btn_pick_outline_color.setStyleSheet(
                f"background-color: {self.current_outline_hex}; color: {btn_fg}; font-weight: bold; padding: 4px 10px;"
            )
            self.btn_pick_outline_color.setText(f"■ {self.current_outline_hex}")
            self._update_font_preview()

    def _update_font_preview(self) -> None:
        font_name = self.combo_font.currentText()
        size = max(13, min(36, self.spin_font_size.value()))
        outline_px = self.spin_outline.value()

        shadows = []
        if outline_px > 0:
            for dx in range(-outline_px, outline_px + 1):
                for dy in range(-outline_px, outline_px + 1):
                    if dx != 0 or dy != 0:
                        shadows.append(f"{dx}px {dy}px 0px {self.current_outline_hex}")
        shadow_css = f"text-shadow: {', '.join(shadows)};" if shadows else ""

        self.lbl_font_preview.setStyleSheet(
            f"color: {self.current_primary_hex}; "
            f"font-family: '{font_name}'; "
            f"font-size: {size}px; "
            f"font-weight: bold; "
            f"{shadow_css}"
        )

    def _save_settings(self) -> None:
        ffmpeg_path = self.edit_ffmpeg.text().strip()
        ffprobe_path = self.edit_ffprobe.text().strip()
        if not ffmpeg_path or not ffprobe_path:
            QMessageBox.warning(self, "Thiếu đường dẫn", "Cần nhập cả đường dẫn FFmpeg và FFprobe.")
            return
        self.settings.language = self.combo_lang.currentData()
        I18n.set_language(self.settings.language)
        out_dir = self.edit_output_dir.text().strip()
        if not out_dir or out_dir.rstrip("/") == "/Users" or not os.access(os.path.dirname(os.path.abspath(out_dir)), os.W_OK):
            out_dir = os.path.expanduser("~/Movies/vTranslator")
        self.settings.output_dir = out_dir
        self.settings.vision_model = self.combo_ai_model.currentData() or self.combo_ai_model.currentText().strip() or "ag/gemini-3.8-flash-low"
        self.settings.ffmpeg_path = ffmpeg_path
        self.settings.ffprobe_path = ffprobe_path
        self.settings.asr_engine = self.combo_asr_engine.currentData()
        self.settings.firered_model_dir = self.edit_firered_dir.text().strip()

        # Runtime dubbing uses the installed local VieNeu model and cloned voices.
        self.settings.tts_engine = "vieneu"
        self.settings.vieneu_model_dir = self._selected_vieneu_dir

        # Audio settings
        self.settings.original_volume = self.slider_orig_vol.value() / 100.0
        self.settings.dubbing_volume = self.slider_dub_vol.value() / 100.0
        self.settings.max_tempo_stretch = self.slider_tempo.value() / 100.0

        # Subtitle style
        self.settings.font_name = self.combo_font.currentText()
        self.settings.font_size = self.spin_font_size.value()
        self.settings.margin_v = self.spin_margin_v.value()
        self.settings.outline_width = self.spin_outline.value()
        self.settings.primary_color = hex_to_ass(self.current_primary_hex)
        self.settings.outline_color = hex_to_ass(self.current_outline_hex)
        self.settings.alignment = self.combo_alignment.currentData() or 2
        self.settings.single_line = self.cb_single_line.isChecked()

        # Render settings
        self.settings.render_preset = self.combo_preset.currentText()
        self.settings.render_crf = self.spin_crf.value()
        self.settings.burn_subtitles = self.cb_burn_sub.isChecked()

        self.settings.ensure_directories()

        # Save to Keychain
        self.keychain.set_secret("translation_api_url", self.edit_trans_url.text().strip())
        self.keychain.set_secret("translation_api_key", self.edit_trans_key.text().strip())

        self.accept()

