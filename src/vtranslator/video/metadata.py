"""Video Metadata Extractor using FFprobe."""

import json
import subprocess
from typing import Optional
from ..domain.models import VideoMetadata


class VideoMetadataService:
    def __init__(self, ffprobe_path: str = "ffprobe"):
        self.ffprobe_path = ffprobe_path

    def probe(self, video_path: str) -> VideoMetadata:
        """Trích xuất thông số kỹ thuật của video bằng ffprobe."""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        video_stream = None
        audio_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        width = 1920
        height = 1080
        fps = 30.0
        v_codec = "h264"
        duration_ms = 0
        bitrate = 0

        if video_stream:
            width = int(video_stream.get("width", 1920))
            height = int(video_stream.get("height", 1080))
            v_codec = video_stream.get("codec_name", "h264")

            # Tính FPS từ r_frame_rate hoặc avg_frame_rate
            fps_str = video_stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)

        # Duration từ format hoặc video stream
        format_info = data.get("format", {})
        if "duration" in format_info:
            duration_ms = int(float(format_info["duration"]) * 1000)
        elif video_stream and "duration" in video_stream:
            duration_ms = int(float(video_stream["duration"]) * 1000)

        if "bit_rate" in format_info:
            bitrate = int(format_info["bit_rate"])

        a_codec = audio_stream.get("codec_name", "aac") if audio_stream else "none"
        sample_rate = int(audio_stream.get("sample_rate", 48000)) if audio_stream else 48000
        channels = int(audio_stream.get("channels", 2)) if audio_stream else 2

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            duration_ms=duration_ms,
            fps=fps,
            bitrate=bitrate,
            video_codec=v_codec,
            audio_codec=a_codec,
            audio_sample_rate=sample_rate,
            audio_channels=channels
        )

    def extract_metadata(self, video_path: str) -> VideoMetadata:
        """Alias for probe() to support both probe and extract_metadata calls."""
        return self.probe(video_path)
