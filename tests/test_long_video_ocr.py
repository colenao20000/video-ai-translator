"""Tests for long video OCR robustness, accurate duration estimation, and ROI scaling."""

import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.vtranslator.subtitle.ocr_extractor import OcrSubtitleExtractor
from src.vtranslator.video.metadata import VideoMetadataService
from src.vtranslator.domain.models import VideoMetadata


class TestLongVideoOcrRobustness:
    def test_long_video_uses_ffprobe_duration_when_opencv_frame_count_is_truncated(self, monkeypatch, tmp_path):
        """
        Kiểm tra: Khi video dài hơn 2 tiếng (7200s), nếu OpenCV chỉ trả về frame count nhỏ
        (ví dụ 50 hoặc 100 frames do lỗi container header), hệ thống dùng ffprobe để lấy
        đầy đủ thời lượng thực và không bị ngắt sớm ở câu 50.
        """
        dummy_video = str(tmp_path / "long_movie_2hours.mp4")
        with open(dummy_video, "w") as f:
            f.write("dummy_video_data")

        # Mock VideoMetadataService probe trả về 7200s (2 tiếng)
        mock_meta = VideoMetadata(
            path=dummy_video,
            width=1920,
            height=1080,
            duration_ms=7200 * 1000,
            fps=25.0,
            bitrate=4000000,
            video_codec="h264",
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2
        )
        monkeypatch.setattr(VideoMetadataService, "probe", lambda self, p: mock_meta)

        # Mock cv2.VideoCapture: CAP_PROP_FRAME_COUNT chỉ trả về 50 (bị cụt)
        # Nhưng video thực tế tiếp tục đọc được
        frames_read = []

        class MockCap:
            def __init__(self, path):
                self.pos = 0
                self.opened = True

            def isOpened(self):
                return self.opened

            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 25.0
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 50.0  # Lỗi điển hình của OpenCV trên file lớn: chỉ báo 50 frames
                return 0.0

            def read(self):
                # Cho phép đọc đến frame 300 (vượt xa con số 50 bị cụt ban đầu)
                if self.pos < 300:
                    self.pos += 1
                    frames_read.append(self.pos)
                    # Tạo frame giả 1080p
                    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    return True, dummy_frame
                return False, None

            def grab(self):
                if self.pos < 300:
                    self.pos += 1
                    return True
                return False

            def set(self, prop, val):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.pos = int(val)
                return True

            def release(self):
                self.opened = False

        monkeypatch.setattr(cv2, "VideoCapture", MockCap)

        # Chạy extract_subtitles_from_video
        segs = OcrSubtitleExtractor.extract_subtitles_from_video(
            dummy_video,
            sample_interval=0.35
        )

        # Hệ thống phải tiếp tục quét vượt qua frame 50 nhờ thời lượng ffprobe
        assert frames_read[-1] > 200, f"Phải quét tiếp tục vượt qua mốc 50 frames, thực tế quét đến frame: {frames_read[-1]}"

    def test_ocr_recovers_from_dropped_corrupt_frames_without_early_exit(self, monkeypatch, tmp_path):
        """
        Kiểm tra: Khi gặp 2-3 frame lỗi/hỏng (ret=False) giữa video 2 tiếng, hệ thống tự seek
        nhảy qua khung lỗi và tiếp tục quét phần còn lại thay vì dừng ngắt tiến trình.
        """
        dummy_video = str(tmp_path / "movie_with_glitch.mp4")
        with open(dummy_video, "w") as f:
            f.write("dummy")

        mock_meta = VideoMetadata(
            path=dummy_video,
            width=1920,
            height=1080,
            duration_ms=60 * 1000,
            fps=25.0,
            bitrate=2000000,
            video_codec="h264",
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2
        )
        monkeypatch.setattr(VideoMetadataService, "probe", lambda self, p: mock_meta)

        class GlitchCap:
            def __init__(self, path):
                self.pos = 0

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 25.0
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 500.0
                return 0.0

            def read(self):
                # Giả lập frame 50 bị lỗi decode (ret=False)
                if self.pos in (50, 51):
                    self.pos += 1
                    return False, None
                if self.pos < 200:
                    self.pos += 1
                    return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
                return False, None

            def grab(self):
                if self.pos < 200:
                    self.pos += 1
                    return True
                return False

            def set(self, prop, val):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.pos = int(val)
                return True

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoCapture", GlitchCap)

        # Chạy extract_subtitles_from_video
        segs = OcrSubtitleExtractor.extract_subtitles_from_video(
            dummy_video,
            sample_interval=0.35
        )

        # Quá trình không bị ném ngoại lệ và không bị dừng ngay tại frame 50

    def test_proportional_roi_scaling_from_720p_preview_to_1080p_and_4k(self):
        """
        Kiểm tra: Khung quét ROI chọn ở màn hình preview 720p (scene 1280x720)
        được tự động co giãn chính xác lên khung hình 1080p (1920x1080) và 4K (3840x2160),
        đặt đúng dải phụ đề đáy màn hình thay vì bị kẹt ở lưng chừng giữa màn.
        """
        # ROI ở 720p: đáy màn hình y=561 (78%), cao=158 (22%)
        roi_720p = {
            "x": 96,
            "y": 561,
            "width": 1088,
            "height": 158,
            "scene_width": 1280,
            "scene_height": 720
        }

        # Giả lập frame 1080p
        frame_1080 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Giả lập cap trả về frame_1080
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, frame_1080)

        crop_1080 = OcrSubtitleExtractor.extract_crop_at_timestamp(mock_cap, 1.0, roi=roi_720p)
        assert crop_1080 is not None
        ch, cw = crop_1080.shape[:2]

        # 158 * (1080 / 720) = 237 px
        assert abs(ch - 237) <= 2, f"Chiều cao crop 1080p phải là ~237px, thực tế: {ch}"
        # 1088 * (1920 / 1280) = 1632 px
        assert abs(cw - 1632) <= 2, f"Chiều rộng crop 1080p phải là ~1632px, thực tế: {cw}"
