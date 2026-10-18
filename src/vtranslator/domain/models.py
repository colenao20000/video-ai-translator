"""Core Configuration and State Domain Models."""

from typing import Optional
from pydantic import BaseModel, Field
from .enums import FlipMode, CoverMode, SubtitleStyleEngine


class VideoMetadata(BaseModel):
    path: str
    width: int = 1920
    height: int = 1080
    duration_ms: int = 0
    fps: float = 30.0
    bitrate: int = 0
    video_codec: str = "h264"
    audio_codec: str = "aac"
    audio_sample_rate: int = 48000
    audio_channels: int = 2

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


class TransformSettings(BaseModel):
    zoom_percent: int = Field(100, ge=50, le=400, description="Tỉ lệ phóng to/thu nhỏ (50% - 400%)")
    flip_mode: FlipMode = Field(FlipMode.OFF, description="Chế độ lật hình: OFF, Horizontal, Vertical, Both")

    @property
    def zoom_ratio(self) -> float:
        return self.zoom_percent / 100.0


class BlurCoverSettings(BaseModel):
    enabled: bool = True
    mode: CoverMode = CoverMode.BLUR
    # Tọa độ và kích thước vùng che (tính theo pixel hoặc tỉ lệ khung hình)
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    blur_radius: int = Field(20, ge=1, le=100, description="Độ mờ khi dùng chế độ blur")
    solid_color: str = Field("#000000", description="Mã màu viền che (solid black mặc định)")
    opacity: float = Field(1.0, ge=0.0, le=1.0)


class AudioMixSettings(BaseModel):
    original_volume: float = Field(0.10, ge=0.0, le=2.0, description="Âm lượng video gốc (mặc định 10%)")
    dubbing_volume: float = Field(1.00, ge=0.0, le=2.0, description="Âm lượng lồng tiếng AI (mặc định 100%)")
    bgm_volume: float = Field(0.20, ge=0.0, le=2.0, description="Âm lượng nhạc nền phụ")
    bgm_path: Optional[str] = None
    fit_to_slot: bool = True
    max_tempo_stretch: float = Field(1.18, ge=1.0, le=2.0, description="Ngưỡng atempo tối đa co giãn giọng đọc (1.15-1.18x là chuẩn vàng rõ chữ)")


class SubtitleStyleSettings(BaseModel):
    engine: SubtitleStyleEngine = SubtitleStyleEngine.ASS
    font_name: str = "Arial"
    font_size: int = 22
    primary_color: str = "&H00FFFFFF"  # White in ASS
    outline_color: str = "&H00000000"  # Black in ASS
    outline_width: int = 2
    margin_v: int = 30
    alignment: int = 2  # Bottom center in ASS
    single_line: bool = True  # Tự động gom thành 1 dòng duy nhất (không rớt dòng)


class LogoSettings(BaseModel):
    enabled: bool = False
    path: str = ""
    position: str = "top_right"  # "top_right", "top_left", "bottom_right", "bottom_left"
    scale_percent: int = Field(12, ge=3, le=50, description="Kích thước logo (% chiều rộng video)")
    opacity: float = Field(0.85, ge=0.1, le=1.0, description="Độ mờ đục logo (0.1 - 1.0)")
    margin_x: int = 24
    margin_y: int = 24
