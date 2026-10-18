"""Blur & Solid Cover Subtitles Filter Graph Builder."""

from typing import Optional
from ..domain.models import BlurCoverSettings
from ..domain.enums import CoverMode


class BlurCoverBuilder:
    @staticmethod
    def build_filter(
        settings: BlurCoverSettings,
        in_label: str = "[v_in]",
        out_label: str = "[v_out]",
        max_w: Optional[int] = None,
        max_h: Optional[int] = None,
    ) -> Optional[str]:
        """Tạo chuỗi filter graph FFmpeg để làm mờ hoặc che hộp phụ đề cũ (có bảo vệ biên khung hình)."""
        if not settings.enabled or settings.width <= 0 or settings.height <= 0:
            return None

        x = max(0, int(settings.x))
        y = max(0, int(settings.y))
        w = int(settings.width)
        h = int(settings.height)

        if max_w and max_w > 10:
            if x >= max_w - 4:
                x = max(0, max_w - 10)
            w = min(w, max_w - x)

        if max_h and max_h > 10:
            if y >= max_h - 4:
                y = max(0, max_h - 10)
            h = min(h, max_h - y)

        if w <= 2 or h <= 2:
            return None

        if settings.mode == CoverMode.SOLID:
            color = settings.solid_color.replace("#", "")
            return f"{in_label}drawbox=x={x}:y={y}:w={w}:h={h}:color=0x{color}:t=fill{out_label};"
        else:
            rad = max(1, settings.blur_radius)
            return (
                f"{in_label}split[m_blur][p_blur];"
                f"[p_blur]crop=w={w}:h={h}:x={x}:y={y}[pc_blur];"
                f"[pc_blur]avgblur=sizeX={rad}:sizeY={rad}[pb_blur];"
                f"[m_blur][pb_blur]overlay=x={x}:y={y}{out_label};"
            )
