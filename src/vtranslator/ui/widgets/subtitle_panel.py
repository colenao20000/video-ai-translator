"""Subtitle editor panel."""

from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QComboBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from ...domain.segments import SubtitleSegment
from ...domain.characters import Character
from ...domain.enums import TtsStatus, SegmentFitStatus
from ..i18n import tr


class SubtitlePanel(QWidget):
    translation_edited = Signal(str, str)
    character_reassigned = Signal(str, str)
    speech_style_edited = Signal(str, str)
    extract_ai_requested = Signal()
    extract_ocr_requested = Signal()
    extract_hybrid_requested = Signal()
    toggle_ocr_roi_requested = Signal(bool)
    ocr_preset_selected = Signal(str)
    translate_all_requested = Signal()
    generate_tts_requested = Signal()
    render_requested = Signal()
    sub_style_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.segments: List[SubtitleSegment] = []
        self.characters: List[Character] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── 1. MAIN WORKFLOW ACTION BAR (Logic 3-Step Header) ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        title = QLabel("📝 Kịch bản & Phụ đề")
        title.setObjectName("panelTitle")
        toolbar.addWidget(title)

        self.lbl_count = QLabel("(0 câu)")
        self.lbl_count.setObjectName("mutedLabel")
        toolbar.addWidget(self.lbl_count)

        self.lbl_9router = QLabel("9router Local")
        self.lbl_9router.setObjectName("badgeOff")
        toolbar.addWidget(self.lbl_9router)

        toolbar.addStretch()

        # Step 1: Quét phụ đề OCR trực tiếp từ video
        self.btn_hybrid = QPushButton("🔍 1. Quét phụ đề OCR")
        self.btn_hybrid.setObjectName("accentButton")
        self.btn_hybrid.setFixedHeight(32)
        self.btn_hybrid.setToolTip("Quét phụ đề trực tiếp từ video bằng OCR siêu tốc (không dùng ASR).")
        self.btn_hybrid.clicked.connect(self.extract_ocr_requested.emit)
        toolbar.addWidget(self.btn_hybrid)

        # Step 2: Dịch toàn bộ kịch bản
        self.btn_translate = QPushButton("🌐 2. Dịch tiếng Việt")
        self.btn_translate.setFixedHeight(32)
        self.btn_translate.setToolTip("Dịch câu thoại sang tiếng Việt; rà lại nghĩa và độ dài trước khi xuất.")
        self.btn_translate.clicked.connect(self.translate_all_requested.emit)
        toolbar.addWidget(self.btn_translate)

        # Step 3: Tạo giọng đọc AI
        self.btn_tts = QPushButton("🔊 3. Tạo giọng đọc AI")
        self.btn_tts.setFixedHeight(32)
        self.btn_tts.setToolTip("Tạo giọng cho các câu đã dịch và gán người nói.")
        self.btn_tts.clicked.connect(self.generate_tts_requested.emit)
        toolbar.addWidget(self.btn_tts)

        # Step 4: Xuất Video thành phẩm (Render)
        self.btn_render = QPushButton("🎬 4. Xuất Video (Render)")
        self.btn_render.setObjectName("accentButton")
        self.btn_render.setFixedHeight(32)
        self.btn_render.setToolTip("Bước 4: Xuất video thành phẩm từ toàn bộ phụ đề và giọng đọc đã làm xong")
        self.btn_render.clicked.connect(self.render_requested.emit)
        toolbar.addWidget(self.btn_render)

        layout.addLayout(toolbar)

        # ── 2. SUB-TOOLBAR (Công cụ phụ & Khung quét OCR) ──
        sub_bar = QHBoxLayout()
        sub_bar.setContentsMargins(0, 0, 0, 0)
        sub_bar.setSpacing(6)

        self.btn_toggle_ocr_roi = QPushButton("⛶ Khung quét chữ")
        self.btn_toggle_ocr_roi.setCheckable(True)
        self.btn_toggle_ocr_roi.setChecked(True)
        self.btn_toggle_ocr_roi.setFixedHeight(26)
        self.btn_toggle_ocr_roi.setToolTip("Bật / tắt khung chữ nhật khoanh vùng chữ trên màn hình xem trước")
        self.btn_toggle_ocr_roi.toggled.connect(lambda chk: self.toggle_ocr_roi_requested.emit(chk))
        sub_bar.addWidget(self.btn_toggle_ocr_roi)

        self.combo_ocr_presets = QComboBox()
        self.combo_ocr_presets.setFixedHeight(26)
        self.combo_ocr_presets.addItem("Vị trí: Đáy màn hình", "bottom")
        self.combo_ocr_presets.addItem("Vị trí: Giữa màn hình", "center")
        self.combo_ocr_presets.addItem("Vị trí: Toàn bộ nửa dưới", "full_bottom")
        self.combo_ocr_presets.currentIndexChanged.connect(self._on_preset_changed)
        sub_bar.addWidget(self.combo_ocr_presets)

        # Nút phụ chạy riêng lẻ nếu người dùng muốn tùy biến
        self.btn_extract = QPushButton("🎙️ Chỉ chạy ASR")
        self.btn_extract.setFixedHeight(26)
        self.btn_extract.setToolTip("Chỉ nhận diện âm thanh bằng ASR (không quét OCR)")
        self.btn_extract.clicked.connect(self.extract_ai_requested.emit)
        sub_bar.addWidget(self.btn_extract)

        self.btn_ocr = QPushButton("🔍 Chỉ quét OCR")
        self.btn_ocr.setFixedHeight(26)
        self.btn_ocr.setToolTip("Chỉ quét chữ hình ảnh qua OCR (không dùng ASR)")
        self.btn_ocr.clicked.connect(self.extract_ocr_requested.emit)
        sub_bar.addWidget(self.btn_ocr)

        self.btn_sub_style = QPushButton("🎨 Phông chữ & vị trí")
        self.btn_sub_style.setFixedHeight(26)
        self.btn_sub_style.setToolTip("Cấu hình phông chữ, cỡ chữ, màu sắc, vị trí hiển thị và chế độ gom 1 dòng")
        self.btn_sub_style.clicked.connect(self.sub_style_requested.emit)
        sub_bar.addWidget(self.btn_sub_style)

        sub_bar.addStretch()

        # Ô tìm kiếm câu thoại
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔍 Lọc câu thoại...")
        self.edit_search.setFixedWidth(150)
        self.edit_search.setFixedHeight(26)
        self.edit_search.textChanged.connect(self._on_filter_text_changed)
        sub_bar.addWidget(self.edit_search)

        layout.addLayout(sub_bar)

        # Subtitle review table.
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "#",
            "Bắt đầu",
            "Kết thúc",
            "Nhân vật nói",
            "Nguồn / độ tin cậy",
            "Lời thoại gốc",
            "Bản dịch tiếng Việt",
            "Sắc thái giọng",
            "Trạng thái TTS"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)

        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

    def set_data(self, segments: List[SubtitleSegment], characters: List[Character]) -> None:
        self.segments = segments
        self.characters = characters
        self.lbl_count.setText(f"({len(segments)} câu)")

        self.table.blockSignals(True)
        self.table.setRowCount(len(segments))

        _tts_labels = {
            TtsStatus.READY: "✅ Đã có giọng",
            TtsStatus.GENERATING: "⏳ Đang tạo...",
            TtsStatus.FAILED: "❌ Lỗi",
            TtsStatus.SKIPPED: "⏭ Bỏ qua",
            TtsStatus.PENDING: "Chờ tạo",
        }
        styles = [
            ("", "Tự động"),
            ("tự nhiên, bình tĩnh", "Tự nhiên"),
            ("vui vẻ, tươi sáng", "Vui vẻ"),
            ("buồn, trầm lắng", "Buồn"),
            ("tức giận, dứt khoát", "Tức giận"),
            ("lo lắng, bất an", "Lo lắng"),
            ("thì thầm, nhẹ nhàng", "Thì thầm"),
            ("nghiêm nghị, chắc chắn", "Nghiêm nghị"),
            ("hào hứng, giàu năng lượng", "Hào hứng"),
        ]

        for row, seg in enumerate(segments):
            # # (ID)
            item_id = QTableWidgetItem(f"{row + 1}")
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
            item_id.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item_id)

            # Start
            item_start = QTableWidgetItem(f"{seg.start_sec:.2f}s")
            item_start.setFlags(item_start.flags() & ~Qt.ItemIsEditable)
            item_start.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, item_start)

            # End
            item_end = QTableWidgetItem(f"{seg.end_sec:.2f}s")
            item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
            item_end.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, item_end)

            # Character dropdown
            combo_char = QComboBox()
            combo_char.addItem("Chưa xác định", "char_unknown")
            for c in characters:
                combo_char.addItem(f"{c.name} ({c.voice_id})", c.id)
            idx_c = combo_char.findData(seg.character_id)
            if idx_c >= 0:
                combo_char.setCurrentIndex(idx_c)
            combo_char.currentIndexChanged.connect(
                lambda idx, s=seg, cb=combo_char: self.character_reassigned.emit(
                    s.segment_id, cb.currentData()
                )
            )
            source_labels = {
                "manual": "Đã gán tay",
                "diarization": f"Âm thanh · {round(seg.confidence * 100)}%",
                "model_guess": "AI gợi ý",
                "heuristic": "Ước lượng",
                "unassigned": "Chưa xác định",
            }
            combo_char.setToolTip(
                f"Nguồn gán: {source_labels.get(seg.speaker_source, 'Chưa xác định')}\n"
                f"Speaker: {seg.speaker_id} · độ tin cậy {round(seg.confidence * 100)}%"
            )
            self.table.setCellWidget(row, 3, combo_char)

            speaker_status = QTableWidgetItem(source_labels.get(seg.speaker_source, "Chưa xác định"))
            speaker_status.setFlags(speaker_status.flags() & ~Qt.ItemIsEditable)
            speaker_status.setTextAlignment(Qt.AlignCenter)
            speaker_status.setToolTip(combo_char.toolTip())
            self.table.setItem(row, 4, speaker_status)

            # Original text
            item_orig = QTableWidgetItem(seg.original_text)
            item_orig.setFlags(item_orig.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 5, item_orig)

            # Translation text (Editable directly)
            item_trans = QTableWidgetItem(seg.translated_text)
            item_trans.setData(Qt.UserRole, seg.segment_id)
            item_trans.setToolTip("Nhấp đúp chuột để chỉnh sửa câu dịch trực tiếp")
            self.table.setItem(row, 6, item_trans)

            # Per-line acting direction is editable and participates in the TTS cache key.
            combo_style = QComboBox()
            for style_value, label in styles:
                combo_style.addItem(label, style_value)
            current_style = (seg.speech_style or "").strip()
            style_index = combo_style.findData(current_style)
            if style_index < 0 and current_style:
                combo_style.addItem(current_style, current_style)
                style_index = combo_style.count() - 1
            combo_style.setCurrentIndex(max(0, style_index))
            combo_style.setToolTip("Chỉ dẫn diễn xuất cho câu này. Đổi sắc thái sẽ tạo lại giọng đọc.")
            combo_style.currentIndexChanged.connect(
                lambda idx, s=seg, cb=combo_style: self.speech_style_edited.emit(
                    s.segment_id, cb.currentData() or ""
                )
            )
            self.table.setCellWidget(row, 7, combo_style)

            # TTS Status
            status_text = _tts_labels.get(seg.tts_status, "Chờ tạo")
            if seg.tts_status == TtsStatus.READY and seg.fit_status == SegmentFitStatus.OVERFLOW:
                status_text = "⚠ Quá khung"
            elif seg.tts_status == TtsStatus.READY and seg.fit_status == SegmentFitStatus.COLLISION:
                status_text = "⚠ Chồng thoại"
            elif seg.tts_status == TtsStatus.READY and seg.fit_status == SegmentFitStatus.ADJUSTED:
                status_text = "Đã căn tốc độ"
            item_status = QTableWidgetItem(status_text)
            item_status.setFlags(item_status.flags() & ~Qt.ItemIsEditable)
            item_status.setTextAlignment(Qt.AlignCenter)
            if seg.fit_status in (SegmentFitStatus.OVERFLOW, SegmentFitStatus.COLLISION):
                item_status.setToolTip("Âm thanh cần kiểm tra. Câu nói được giữ nguyên, không bị cắt cụt.")
            self.table.setItem(row, 8, item_status)

        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 6:
            seg_id = item.data(Qt.UserRole)
            if seg_id:
                self.translation_edited.emit(seg_id, item.text().strip())

    def _on_filter_text_changed(self, query: str) -> None:
        q = query.strip().lower()
        for r in range(self.table.rowCount()):
            it_orig = self.table.item(r, 5)
            it_trans = self.table.item(r, 6)
            match = True
            if q:
                orig_match = it_orig and q in it_orig.text().lower()
                trans_match = it_trans and q in it_trans.text().lower()
                match = orig_match or trans_match
            self.table.setRowHidden(r, not match)

    def _on_preset_changed(self, idx: int) -> None:
        val = self.combo_ocr_presets.itemData(idx)
        if val:
            self.ocr_preset_selected.emit(val)

    def set_ocr_roi_info(self, x: int, y: int, w: int, h: int, is_active: bool = True) -> None:
        self.btn_toggle_ocr_roi.blockSignals(True)
        self.btn_toggle_ocr_roi.setChecked(is_active)
        self.btn_toggle_ocr_roi.blockSignals(False)
