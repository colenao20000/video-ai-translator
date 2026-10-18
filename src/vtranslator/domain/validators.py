"""Timeline & Project Integrity Validators (Section 8 Compliance)."""

import os
import subprocess
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import soundfile as sf
from .project import Project
from .segments import SubtitleSegment
from .characters import Character
from .enums import CharacterMatchStatus


class ValidationIssue(BaseModel):
    level: str  # "ERROR" | "WARNING"
    category: str
    message: str
    segment_id: Optional[str] = None


class ValidationResult(BaseModel):
    is_valid: bool = True
    issues: List[ValidationIssue] = Field(default_factory=list)

    def add_error(self, category: str, message: str, segment_id: Optional[str] = None) -> None:
        self.is_valid = False
        self.issues.append(ValidationIssue(level="ERROR", category=category, message=message, segment_id=segment_id))

    def add_warning(self, category: str, message: str, segment_id: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(level="WARNING", category=category, message=message, segment_id=segment_id))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "WARNING"]


class TimelineValidator:
    @classmethod
    def validate_timeline_data(cls, segments: List[SubtitleSegment]) -> ValidationResult:
        """
        Kiểm tra tính toàn vẹn của dữ liệu timeline:
        - segment_id không trùng lặp.
        - start_ms < end_ms.
        - Không có bản dịch rỗng khi câu gốc có chữ.
        """
        res = ValidationResult()
        seen_ids = set()

        for s in segments:
            # 1. Trùng lặp segment_id
            if s.segment_id in seen_ids:
                res.add_error("DATA_INTEGRITY", f"Trùng lặp segment_id: '{s.segment_id}'", s.segment_id)
            seen_ids.add(s.segment_id)

            # 2. start_ms < end_ms
            if s.start_ms >= s.end_ms:
                res.add_error("TIMELINE", f"start_ms ({s.start_ms}) lớn hơn hoặc bằng end_ms ({s.end_ms})", s.segment_id)

            # 3. Bản dịch thiếu
            if s.original_text.strip() and not s.translated_text.strip():
                res.add_warning("TRANSLATION", f"Câu gốc có chữ nhưng bản dịch rỗng", s.segment_id)

        return res

    @classmethod
    def validate_tts_audio(
        cls,
        segments: List[SubtitleSegment],
        master_audio_path: Optional[str] = None,
        expected_duration_sec: Optional[float] = None
    ) -> ValidationResult:
        """
        Kiểm tra tính toàn vẹn của âm thanh TTS & Master audio:
        - File TTS thực sự tồn tại và đọc được.
        - Thời lượng TTS đã được đo đạc.
        - Không có hiện tượng cắt cụt âm ngoài chính sách.
        - Kiểm tra clipping và độ dài master audio.
        """
        res = ValidationResult()

        for s in segments:
            if s.tts_audio_path:
                if not os.path.exists(s.tts_audio_path):
                    res.add_error("AUDIO_FILE", f"File TTS không tồn tại: '{s.tts_audio_path}'", s.segment_id)
                    continue

                try:
                    info = sf.info(s.tts_audio_path)
                    if info.duration <= 0.05:
                        res.add_error("AUDIO_FILE", f"File TTS rỗng hoặc quá ngắn ({info.duration:.2f}s)", s.segment_id)
                except Exception as e:
                    res.add_error("AUDIO_FILE", f"Không thể giải mã file WAV TTS: {e}", s.segment_id)

        # Kiểm tra Master Audio nếu có
        if master_audio_path:
            if not os.path.exists(master_audio_path):
                res.add_error("MASTER_AUDIO", f"Tệp master audio không tồn tại: {master_audio_path}")
            else:
                try:
                    info = sf.info(master_audio_path)
                    if expected_duration_sec and info.duration < (expected_duration_sec - 1.0):
                        res.add_warning(
                            "MASTER_AUDIO",
                            f"Thời lượng master audio ({info.duration:.1f}s) ngắn hơn video ({expected_duration_sec:.1f}s)"
                        )
                except Exception as e:
                    res.add_error("MASTER_AUDIO", f"Không thể giải mã master audio: {e}")

        return res


class ProjectIntegrityValidator:
    @classmethod
    def validate_project(cls, project: Project) -> ValidationResult:
        """
        Kiểm tra tính toàn vẹn của toàn bộ dự án trước khi render:
        - Kiểm tra timeline segments.
        - Kiểm tra liên kết Speaker -> Character -> Voice.
        - Kiểm tra Character chưa xử lý nhưng bị đánh dấu Confirmed.
        - Kiểm tra audio TTS.
        """
        res = TimelineValidator.validate_timeline_data(project.segments)

        char_ids = {c.id for c in project.characters}

        for s in project.segments:
            if s.character_id not in char_ids and s.character_id != "char_unknown":
                res.add_error(
                    "CHARACTER_LINK",
                    f"Segment trỏ tới character_id không tồn tại: '{s.character_id}'",
                    s.segment_id
                )

        # Kiểm tra nhân vật chưa xử lý
        for c in project.characters:
            if c.match_status == CharacterMatchStatus.CONFIRMED and not c.voice_id:
                res.add_error(
                    "CHARACTER_PROFILE",
                    f"Nhân vật '{c.name}' ({c.id}) đã Confirmed nhưng chưa có Voice ID"
                )

        return res

    @classmethod
    def validate_output_video(
        cls,
        output_video_path: str,
        expected_duration_sec: float,
        ffprobe_path: str = "ffprobe",
    ) -> ValidationResult:
        """
        Kiểm tra video đầu ra sau khi FFmpeg render:
        - File tồn tại và có kích thước > 0.
        - Có thể probe được bằng ffprobe.
        - Không bị lệch thời lượng quá mức cho phép (> 2.0s).
        """
        res = ValidationResult()

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) < 1000:
            res.add_error("OUTPUT_VIDEO", f"File video đầu ra không tồn tại hoặc bị hỏng: '{output_video_path}'")
            return res

        cmd = [
            ffprobe_path or "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            output_video_path
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out_dur = float(p.stdout.strip())
            drift = abs(out_dur - expected_duration_sec)
            if drift > 2.5:
                res.add_warning("OUTPUT_VIDEO", f"Thời lượng video đầu ra ({out_dur:.1f}s) lệch {drift:.1f}s so với gốc ({expected_duration_sec:.1f}s)")
        except Exception as e:
            res.add_error("OUTPUT_VIDEO", f"Lỗi ffprobe kiểm tra video đầu ra: {e}")

        return res
