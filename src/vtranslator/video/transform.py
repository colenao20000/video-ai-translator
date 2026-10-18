"""Video Transform Filter Builder (Phóng to 50%-400% & Lật hình theo docs/VIDEO_TRANSFORM.md)."""

from typing import List, Optional
from ..domain.models import TransformSettings
from ..domain.enums import FlipMode


class VideoTransformBuilder:
    @staticmethod
    def build_filter_string(
        settings: TransformSettings,
        target_width: int = 1920,
        target_height: int = 1080
    ) -> Optional[str]:
        """
        Xây dựng chuỗi bộ lọc FFmpeg cho Flip và Zoom đúng đặc tả VIDEO_TRANSFORM.md (Mục 4.1 & 4.2):
        1. Áp dụng Flip trước: hflip, vflip, hoặc hflip,vflip.
        2. Áp dụng Zoom:
           - Zoom In (z > 1.0): scale=w='trunc(iw*z/2)*2':h='trunc(ih*z/2)*2':flags=lanczos,crop=w=W:h=H:x='(iw-W)/2':y='(ih-H)/2'
           - Zoom Out (z < 1.0): scale=w='trunc(iw*z/2)*2':h='trunc(ih*z/2)*2':flags=lanczos,pad=w=W:h=H:x='(ow-iw)/2':y='(oh-ih)/2':color=black
        """
        filters: List[str] = []

        # 1. Flip
        if settings.flip_mode == FlipMode.HORIZONTAL:
            filters.append("hflip")
        elif settings.flip_mode == FlipMode.VERTICAL:
            filters.append("vflip")
        elif settings.flip_mode == FlipMode.BOTH:
            filters.append("hflip")
            filters.append("vflip")

        # 2. Zoom
        z = settings.zoom_ratio
        if abs(z - 1.0) > 0.001:
            w = target_width
            h = target_height
            filters.append(f"scale=w='trunc(iw*{z:.4f}/2)*2':h='trunc(ih*{z:.4f}/2)*2':flags=lanczos")
            if z > 1.0:
                # Zoom In: Crop chính giữa về đúng kích thước chuẩn
                filters.append(f"crop=w={w}:h={h}:x='(iw-{w})/2':y='(ih-{h})/2'")
            else:
                # Zoom Out: Bù viền đen quanh tâm về đúng kích thước chuẩn
                filters.append(f"pad=w={w}:h={h}:x='(ow-iw)/2':y='(oh-ih)/2':color=black")

        return ",".join(filters) if filters else None
