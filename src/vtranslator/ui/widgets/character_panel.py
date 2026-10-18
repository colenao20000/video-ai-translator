"""Character and voice assignment panel."""

import os
from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QComboBox, QHeaderView, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon
from ...domain.characters import Character, DEFAULT_PALETTE
from ...domain.voices import AVAILABLE_VOICES, VoiceLibrary
from ...domain.enums import SpeakerGender, CharacterMatchStatus


class CharacterPanel(QWidget):
    character_changed = Signal(Character)
    character_deleted = Signal(str)
    reset_requested = Signal()
    preview_voice_requested = Signal(str)
    smart_assign_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.characters: List[Character] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Action Header ──
        header = QHBoxLayout()
        header.setContentsMargins(0, 4, 0, 4)
        header.setSpacing(8)

        title = QLabel("🎭 Nhân vật & Giọng nói")
        title.setObjectName("panelTitle")
        header.addWidget(title)

        self.lbl_char_count = QLabel("(0 nhân vật)")
        self.lbl_char_count.setObjectName("mutedLabel")
        header.addWidget(self.lbl_char_count)

        header.addStretch()

        self.btn_smart_assign = QPushButton("🤖 Tự động phân vai (AI)")
        self.btn_smart_assign.setObjectName("accentButton")
        self.btn_smart_assign.setFixedHeight(32)
        self.btn_smart_assign.setToolTip("AI tự gán hồ sơ và giọng theo người nói; bạn có thể chỉnh lại trong bảng.")
        self.btn_smart_assign.clicked.connect(self.smart_assign_requested.emit)
        header.addWidget(self.btn_smart_assign)

        self.btn_add = QPushButton("➕ Thêm vai")
        self.btn_add.setFixedHeight(32)
        self.btn_add.clicked.connect(self._add_new_character)
        header.addWidget(self.btn_add)

        self.btn_reset = QPushButton("🔄 Đặt lại")
        self.btn_reset.setFixedHeight(32)
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        header.addWidget(self.btn_reset)

        layout.addLayout(header)

        # ── Character Table ──
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Ảnh",
            "Tên / Vai diễn",
            "Mã Speaker",
            "Giới tính",
            "Giọng đọc",
            "Trạng thái ghép",
            "Nghe thử",
            "Xóa"
        ])
        self.table.setIconSize(QSize(44, 44))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)

        layout.addWidget(self.table)

    def set_characters(self, characters: List[Character]) -> None:
        self.characters = characters
        self.lbl_char_count.setText(f"({len(characters)} nhân vật)")
        self.table.blockSignals(True)
        self.table.setRowCount(len(characters))

        for row, char in enumerate(characters):
            self.table.setRowHeight(row, 54)

            # Avatar
            item_avatar = QTableWidgetItem()
            item_avatar.setFlags(item_avatar.flags() & ~Qt.ItemIsEditable)
            if char.avatar_path and os.path.exists(char.avatar_path):
                pixmap = QPixmap(char.avatar_path).scaled(
                    42, 42, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                item_avatar.setIcon(QIcon(pixmap))
            else:
                item_avatar.setText("👤")
                item_avatar.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item_avatar)

            # Name
            item_name = QTableWidgetItem(char.name)
            item_name.setData(Qt.UserRole, char.id)
            self.table.setItem(row, 1, item_name)

            # Speaker ID
            item_speaker = QTableWidgetItem(char.speaker_id)
            item_speaker.setFlags(item_speaker.flags() & ~Qt.ItemIsEditable)
            item_speaker.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, item_speaker)

            # Gender
            combo_gender = QComboBox()
            combo_gender.addItems(["Nam", "Nữ", "Chưa rõ"])
            gender_index = {
                SpeakerGender.MALE: 0,
                SpeakerGender.FEMALE: 1,
                SpeakerGender.UNKNOWN: 2,
            }.get(char.gender, 2)
            combo_gender.setCurrentIndex(gender_index)
            combo_gender.currentIndexChanged.connect(
                lambda idx, c=char: self._on_gender_changed(c, idx)
            )
            self.table.setCellWidget(row, 3, combo_gender)

            # Voice
            combo_voice = QComboBox()
            for v in AVAILABLE_VOICES:
                gender_icon = "♂" if v.gender.value == "male" else "♀"
                combo_voice.addItem(f"{gender_icon} {v.name} ({v.region} · {v.description})", v.id)
            canonical_v = VoiceLibrary.get_by_id(char.voice_id)
            target_voice_id = canonical_v.id if canonical_v else char.voice_id
            idx_v = combo_voice.findData(target_voice_id)
            if idx_v >= 0:
                combo_voice.setCurrentIndex(idx_v)
            combo_voice.currentIndexChanged.connect(
                lambda idx, c=char, cb=combo_voice: self._on_voice_changed(c, cb.currentData())
            )
            self.table.setCellWidget(row, 4, combo_voice)

            confidence = max(0, min(100, round(char.confidence * 100)))
            if char.match_status == CharacterMatchStatus.MANUAL:
                status_text = f"Chỉnh tay · {confidence}%"
            elif char.match_status == CharacterMatchStatus.UNKNOWN:
                status_text = f"Ước lượng · {confidence}%"
            else:
                status_text = f"Tự động · {confidence}%"
            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 5, status_item)

            # Preview button
            btn_play = QPushButton("▶")
            btn_play.setFixedWidth(36)
            btn_play.clicked.connect(
                lambda checked=False, c=char: self.preview_voice_requested.emit(c.voice_id)
            )
            self.table.setCellWidget(row, 6, btn_play)

            # Delete button
            btn_del = QPushButton("✕")
            btn_del.setFixedWidth(32)
            btn_del.clicked.connect(lambda checked=False, c=char: self._delete_character(c))
            self.table.setCellWidget(row, 7, btn_del)

        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 1:
            char_id = item.data(Qt.UserRole)
            new_name = item.text().strip()
            for c in self.characters:
                if c.id == char_id and new_name and c.name != new_name:
                    c.name = new_name
                    c.match_status = CharacterMatchStatus.MANUAL
                    c.confidence = 1.0
                    self.character_changed.emit(c)
                    break

    def _on_gender_changed(self, char: Character, idx: int) -> None:
        char.gender = (SpeakerGender.MALE, SpeakerGender.FEMALE, SpeakerGender.UNKNOWN)[idx]
        matching_voices = VoiceLibrary.filter_by_gender(char.gender)
        if matching_voices:
            char.update_voice(matching_voices[0].id)
        char.match_status = CharacterMatchStatus.MANUAL
        self.set_characters(self.characters)
        self.character_changed.emit(char)

    def _on_voice_changed(self, char: Character, new_voice_id: str) -> None:
        if new_voice_id and char.voice_id != new_voice_id:
            char.update_voice(new_voice_id)
            char.match_status = CharacterMatchStatus.MANUAL
            self.character_changed.emit(char)

    def _add_new_character(self) -> None:
        new_idx = len(self.characters) + 1
        char_id = f"char_{new_idx:02d}"
        speaker_id = f"SPEAKER_{new_idx - 1:02d}"
        name = f"Nhân vật {new_idx}"
        assigned_v = AVAILABLE_VOICES[(new_idx - 1) % len(AVAILABLE_VOICES)]
        voice_id = assigned_v.id
        color = DEFAULT_PALETTE[(new_idx - 1) % len(DEFAULT_PALETTE)]

        new_char = Character(
            id=char_id, speaker_id=speaker_id, name=name,
            gender=assigned_v.gender,
            voice_id=voice_id, color=color,
            match_status=CharacterMatchStatus.MANUAL,
        )
        self.characters.append(new_char)
        self.set_characters(self.characters)
        self.character_changed.emit(new_char)

    def _delete_character(self, char: Character) -> None:
        if len(self.characters) <= 1:
            QMessageBox.warning(self, "Lỗi", "Cần giữ ít nhất 1 nhân vật trong dự án.")
            return
        self.characters = [c for c in self.characters if c.id != char.id]
        self.set_characters(self.characters)
        self.character_deleted.emit(char.id)
