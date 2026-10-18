"""Dialogue / Subtitle Segment Domain Model."""

from typing import Optional
from pydantic import BaseModel, Field, model_validator
from .enums import TtsStatus, SegmentFitStatus


class SubtitleSegment(BaseModel):
    """
    Nguồn dữ liệu timeline duy nhất (Single Source of Truth) cho toàn bộ pipeline.
    Tuân thủ quy tắc bất biến:
    - segment_id không đổi xuyên suốt pipeline.
    - start_ms, end_ms giữ nguyên mốc thời gian của video nguồn (source slot).
    - tts_start_ms, tts_end_ms, tts_duration_ms ghi nhận timeline âm thanh lồng tiếng thực tế.
    - speaker_id và character_id liên kết theo danh tính, không phụ thuộc nội dung dịch.
    """
    segment_id: str = Field(..., description="Mã định danh phân đoạn duy nhất và ổn định")
    speaker_id: str = Field("SPEAKER_UNKNOWN", description="Mã người nói ước lượng từ âm thanh")
    character_id: str = Field("char_unknown", description="Mã nhân vật được resolve (Character ID)")
    speaker_source: str = Field("unassigned", description="Nguồn xác định người nói: diarization, manual, OCR hoặc unassigned")
    start_ms: int = Field(..., description="Thời điểm bắt đầu theo video gốc (ms)")
    end_ms: int = Field(..., description="Thời điểm kết thúc theo video gốc (ms)")
    original_text: str = Field("", description="Lời thoại gốc trích xuất từ audio / OCR")
    translated_text: str = Field("", description="Lời thoại đã dịch sang ngôn ngữ đích")
    source_language: str = Field("auto", description="Ngôn ngữ nguồn")
    target_language: str = Field("vi", description="Ngôn ngữ đích")
    voice_id: str = Field("", description="Giọng TTS resolve từ hồ sơ nhân vật")
    tts_audio_path: Optional[str] = Field(None, description="Đường dẫn file WAV âm thanh thuyết minh AI")
    tts_asset_id: Optional[str] = Field(None, description="ID asset âm thanh TTS")
    tts_duration_ms: Optional[int] = Field(None, description="Thời lượng thực tế của âm thanh TTS (ms)")
    tts_start_ms: Optional[int] = Field(None, description="Thời điểm bắt đầu phát âm thanh TTS trong master audio (ms)")
    tts_end_ms: Optional[int] = Field(None, description="Thời điểm kết thúc phát âm thanh TTS trong master audio (ms)")
    applied_tempo: float = Field(1.0, description="Hệ số co giãn tốc độ an toàn đã áp dụng")
    fit_status: SegmentFitStatus = Field(SegmentFitStatus.FIT, description="Trạng thái khớp slot: fit, overflow, underflow, adjusted, collision")
    tts_status: TtsStatus = Field(TtsStatus.PENDING, description="Trạng thái sinh TTS")
    confidence: float = Field(1.0, description="Độ tin cậy nhận diện")
    speech_style: Optional[str] = Field(None, description="Ngữ điệu / cảm xúc diễn xuất theo ngữ cảnh phân cảnh")
    error_code: Optional[str] = Field(None, description="Mã lỗi nếu xử lý thất bại")
    status: str = Field("pending", description="Trạng thái xử lý chung")

    @property
    def source_text(self) -> str:
        return self.original_text

    @source_text.setter
    def source_text(self, value: str) -> None:
        self.original_text = value

    @property
    def source_slot_duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def duration_ms(self) -> int:
        return self.source_slot_duration_ms

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def effective_tts_start_ms(self) -> int:
        return self.tts_start_ms if self.tts_start_ms is not None else self.start_ms

    @property
    def effective_tts_end_ms(self) -> int:
        if self.tts_end_ms is not None:
            return self.tts_end_ms
        if self.tts_duration_ms is not None:
            return self.effective_tts_start_ms + self.tts_duration_ms
        return self.end_ms

    def update_translation(self, new_translated_text: str) -> None:
        """Cập nhật bản dịch mà KHÔNG thay đổi identity (speaker/character) hay timestamp nguồn."""
        self.translated_text = new_translated_text.strip()
        self.invalidate_tts()

    def update_speech_style(self, new_style: Optional[str]) -> None:
        normalized = (new_style or "").strip() or None
        if self.speech_style != normalized:
            self.speech_style = normalized
            self.invalidate_tts()

    def invalidate_tts(self) -> None:
        self.tts_status = TtsStatus.PENDING
        self.tts_audio_path = None
        self.tts_asset_id = None
        self.tts_duration_ms = None
        self.tts_start_ms = None
        self.tts_end_ms = None
        self.applied_tempo = 1.0

    def sync_voice_from_character(self, character_voice_id: str) -> None:
        """Đồng bộ voice_id từ Character. Vô hiệu hóa cache TTS của chính segment này nếu đổi giọng."""
        if self.voice_id != character_voice_id:
            self.voice_id = character_voice_id
            self.invalidate_tts()


# Alias chuẩn hóa theo tài liệu yêu cầu
DialogueSegment = SubtitleSegment
