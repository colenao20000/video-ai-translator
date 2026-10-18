"""Character Domain Model."""

from typing import Optional
from pydantic import BaseModel, Field
from .enums import SpeakerGender, SpeakerAgeGroup, VoiceCharacter, CharacterMatchStatus


DEFAULT_PALETTE = [
    "#007AFF", "#34C759", "#AF52DE", "#FF9500", "#FF2D55",
    "#5856D6", "#FFCC00", "#5AC8FA", "#4CD964", "#FF3B30",
    "#30B0C7", "#D4AC0D", "#A569BD", "#5DADE2", "#48C9B0"
]


class Character(BaseModel):
    id: str = Field(..., description="Mã định danh nhân vật (duy nhất trong project, ví dụ 'char_01')")
    speaker_id: str = Field("SPEAKER_00", description="Mã người nói ước lượng từ âm thanh")
    name: str = Field("Nhân vật", description="Tên hiển thị của nhân vật")
    voice_id: str = Field("Gemini Puck", description="Mã/Tên giọng đọc được gán - Nguồn sự thật duy nhất cho TTS")
    gender: SpeakerGender = Field(SpeakerGender.MALE, description="Giới tính nhân vật")
    color: str = Field("#007AFF", description="Màu nhận diện trên Timeline/Subtitle")
    confidence: float = Field(0.90, description="Độ tin cậy nhận diện")
    age_group: SpeakerAgeGroup = Field(SpeakerAgeGroup.ADULT, description="Nhóm tuổi")
    voice_character: VoiceCharacter = Field(VoiceCharacter.NORMAL, description="Đặc tính ngữ điệu giọng")
    avatar_path: Optional[str] = Field(None, description="Đường dẫn ảnh chân dung / avatar trích xuất từ video")
    custom_prompt: Optional[str] = Field(None, description="Ghi chú / Prompt tùy biến cho TTS")
    match_status: CharacterMatchStatus = Field(CharacterMatchStatus.CONFIRMED, description="Trạng thái xác minh nhân vật: Confirmed, Probable, Unknown, Manual")

    def update_voice(self, new_voice_id: str) -> None:
        """Cập nhật voice_id - nguồn sự thật duy nhất, luôn chuẩn hóa sang ID chuẩn."""
        from .voices import VoiceLibrary
        v = VoiceLibrary.resolve(new_voice_id)
        self.voice_id = v.id if v else new_voice_id

