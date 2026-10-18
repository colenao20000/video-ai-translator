"""Advanced SubStation Alpha (.ass) Subtitle Generator."""

from typing import List
from ..domain.segments import SubtitleSegment
from ..domain.models import SubtitleStyleSettings


class AssGenerator:
    @staticmethod
    def format_timestamp(ms: int) -> str:
        """Định dạng timestamp chuẩn ASS: H:MM:SS.cs (centiseconds)."""
        total_centiseconds = max(0, int(round(ms / 10.0)))
        hours, remainder = divmod(total_centiseconds, 360_000)
        minutes, remainder = divmod(remainder, 6_000)
        seconds, centiseconds = divmod(remainder, 100)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    @classmethod
    def generate(
        cls,
        segments: List[SubtitleSegment],
        output_ass_path: str,
        style: SubtitleStyleSettings = SubtitleStyleSettings(),
        res_x: int = 1920,
        res_y: int = 1080
    ) -> str:
        is_single_line = getattr(style, "single_line", True)
        wrap_style = 2 if is_single_line else 0
        lines = [
            "[Script Info]",
            "Title: vTranslate Subtitle",
            "ScriptType: v4.00+",
            f"WrapStyle: {wrap_style}",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {res_x}",
            f"PlayResY: {res_y}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{style.font_name},{style.font_size},{style.primary_color},&H000000FF,{style.outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{style.outline_width},1,{style.alignment},10,10,{style.margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]

        for seg in segments:
            raw_text = seg.translated_text or seg.original_text
            if is_single_line:
                raw_text = " ".join(raw_text.split())
                text = raw_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            else:
                text = raw_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
            # Đồng bộ với timeline phát âm thực tế của giọng đọc (Section 6.2 #5)
            start = cls.format_timestamp(seg.effective_tts_start_ms)
            end = cls.format_timestamp(seg.effective_tts_end_ms)
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        content = "\n".join(lines) + "\n"
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_ass_path
