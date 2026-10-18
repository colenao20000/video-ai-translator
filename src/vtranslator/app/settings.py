"""Application Settings and Configuration."""

import os
from pydantic import BaseModel, Field
from ..domain.enums import LanguageCode


class AppSettings(BaseModel):
    language: LanguageCode = Field(LanguageCode.VI, description="Ngôn ngữ giao diện (mặc định Tiếng Việt)")
    theme: str = Field("dark", description="Giao diện: dark hoặc light")
    ffmpeg_path: str = Field("ffmpeg", description="Đường dẫn file thực thi ffmpeg")
    ffprobe_path: str = Field("ffprobe", description="Đường dẫn file thực thi ffprobe")
    output_dir: str = Field(os.path.expanduser("~/Movies/vTranslator"), description="Thư mục xuất video")
    temp_dir: str = Field(os.path.expanduser("~/.vtranslator/temp"), description="Thư mục tạm thời")
    vision_model: str = Field("ag/gemini-3.8-flash-low", description="Tên model AI nhận diện kịch bản và dịch thuật")
    auto_save: bool = Field(True, description="Tự động lưu dự án")
    asr_engine: str = Field("firered", description="Công cụ nhận diện giọng nói: firered hoặc whisper")
    firered_model_dir: str = Field(
        os.path.expanduser("~/.vtranslator/models/FireRedASR2-CTC-int8"),
        description="Thư mục chứa mô hình FireRedASR2-CTC-int8",
    )
    vieneu_model_dir: str = Field(
        "",
        description="Thư mục mô hình VieNeu cục bộ; để trống để tự tìm",
    )
    tts_engine: str = Field("vieneu", description="Tổng hợp giọng cục bộ VieNeu với thư viện giọng clone đã cài")
    original_volume: float = Field(0.10, ge=0.0, le=2.0, description="Âm lượng video gốc (mặc định 10%)")
    dubbing_volume: float = Field(1.00, ge=0.0, le=2.0, description="Âm lượng lồng tiếng AI (mặc định 100%)")
    max_tempo_stretch: float = Field(1.18, ge=1.0, le=2.0, description="Ngưỡng atempo tối đa co giãn giọng đọc (1.15-1.18x là chuẩn vàng rõ chữ)")
    render_preset: str = Field("fast", description="Preset xuất video: fast, medium, slow")
    render_crf: int = Field(20, description="Chất lượng video CRF (18 - 26)")
    burn_subtitles: bool = Field(True, description="Tự động chèn phụ đề cứng khi xuất video")
    font_name: str = Field("Arial", description="Tên phông chữ phụ đề")
    font_size: int = Field(22, description="Cỡ chữ phụ đề")
    primary_color: str = Field("&H00FFFFFF", description="Màu chữ chính (ASS format)")
    outline_color: str = Field("&H00000000", description="Màu viền chữ (ASS format)")
    outline_width: int = Field(2, description="Độ dày viền chữ")
    margin_v: int = Field(32, description="Khoảng cách lề dưới")
    alignment: int = Field(2, description="Vị trí phụ đề (2=Đáy, 5=Giữa, 8=Đỉnh)")
    single_line: bool = Field(True, description="Tự động gom phụ đề thành 1 dòng duy nhất")

    def ensure_directories(self) -> None:
        safe_fallback = os.path.expanduser("~/Movies/vTranslator")
        if not self.output_dir or not os.access(os.path.dirname(os.path.abspath(self.output_dir)), os.W_OK) or self.output_dir.rstrip("/") == "/Users":
            self.output_dir = safe_fallback
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(os.path.expanduser("~/.vtranslator/models"), exist_ok=True)
