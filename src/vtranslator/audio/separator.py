"""Separate audio with Demucs when available; otherwise create approximate DSP stems."""

import os
import hashlib
import json
import subprocess
from typing import Tuple, Optional, Callable
import soundfile as sf
import numpy as np


class AudioSeparatorService:
    def __init__(self, temp_dir: str = "~/.vtranslator/temp", ffmpeg_path: str = "ffmpeg"):
        self.temp_dir = os.path.expanduser(temp_dir)
        self.ffmpeg_path = ffmpeg_path
        os.makedirs(self.temp_dir, exist_ok=True)
        self._demucs_model = None
        self.last_method = "unknown"

    @staticmethod
    def _method_metadata_path(vocals_path: str) -> str:
        return vocals_path + ".json"

    def get_cached_paths(self, video_or_audio_path: str) -> Tuple[str, str]:
        """Return stable stem paths tied to the current source file version."""
        source_stat = os.stat(video_or_audio_path)
        cache_identity = (
            f"{os.path.abspath(video_or_audio_path)}:{source_stat.st_size}:"
            f"{source_stat.st_mtime_ns}"
        )
        source_hash = hashlib.md5(cache_identity.encode("utf-8")).hexdigest()[:14]
        vocals = os.path.join(self.temp_dir, f"vocals_{source_hash}.wav")
        background = os.path.join(self.temp_dir, f"bgm_{source_hash}.wav")
        return vocals, background

    def get_separation_method(self, vocals_path: Optional[str]) -> str:
        if not vocals_path:
            return "unknown"
        try:
            with open(self._method_metadata_path(vocals_path), "r", encoding="utf-8") as f:
                method = json.load(f).get("method")
                return method if method in ("demucs", "filter_approx") else "cached_unknown"
        except Exception:
            return "cached_unknown"

    def _get_demucs_model(self):
        if self._demucs_model is None:
            try:
                import torch
                from demucs.pretrained import get_model
                model = get_model("htdemucs")
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                self._demucs_model = (model, device)
            except Exception as e:
                print(f"Không thể khởi tạo Demucs neural model: {e}")
                self._demucs_model = None
        return self._demucs_model

    def separate(
        self,
        video_or_audio_path: str,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[str, str]:
        """
        Tách âm thanh thành 2 rãnh:
        - vocals_wav_path: Rãnh ưu tiên giọng nói, dùng cho ASR và diarization.
        - background_wav_path: Rãnh nền ước lượng để ghép lại; DSP fallback có thể còn lẫn giọng.
        """
        if not os.path.exists(video_or_audio_path):
            raise FileNotFoundError(f"Không tìm thấy tệp: {video_or_audio_path}")

        vocals_out, background_out = self.get_cached_paths(video_or_audio_path)
        v_hash = os.path.basename(vocals_out).removeprefix("vocals_").removesuffix(".wav")

        if (
            not force_refresh
            and os.path.exists(vocals_out)
            and os.path.exists(background_out)
            and os.path.getsize(vocals_out) > 1000
            and os.path.getsize(background_out) > 1000
        ):
            self.last_method = self.get_separation_method(vocals_out)
            if progress_callback:
                status = "Demucs" if self.last_method == "demucs" else "bộ lọc xấp xỉ"
                progress_callback(35, f"Đã nạp các rãnh âm thanh đã lưu ({status}).")
            return vocals_out, background_out

        # Trích xuất âm thanh stereo 44.1kHz từ video nguồn
        source_wav = os.path.join(self.temp_dir, f"raw_audio_{v_hash}.wav")
        if progress_callback:
            progress_callback(8, "Đang trích xuất rãnh âm thanh stereo từ video...")

        subprocess.run(
            [
                self.ffmpeg_path, "-y", "-i", video_or_audio_path,
                "-vn", "-ar", "44100", "-ac", "2", source_wav
            ],
            capture_output=True,
            check=True
        )

        demucs_bundle = self._get_demucs_model()
        separated_by_demucs = False

        if demucs_bundle is not None:
            try:
                import torch
                from demucs.apply import apply_model

                model, device = demucs_bundle
                audio_data, sr = sf.read(source_wav, dtype="float32")
                if audio_data.ndim == 1:
                    audio_data = np.stack([audio_data, audio_data], axis=1)

                total_samples = len(audio_data)
                total_sec = total_samples / float(sr)

                # Chia thành các đoạn chunk 60s để kiểm soát RAM và cập nhật tiến trình liên tục
                chunk_sec = 60
                chunk_samples = sr * chunk_sec
                num_chunks = max(1, int(np.ceil(total_samples / chunk_samples)))

                vocals_chunks = []
                bgm_chunks = []

                for c_idx in range(num_chunks):
                    s_idx = c_idx * chunk_samples
                    e_idx = min(total_samples, (c_idx + 1) * chunk_samples)
                    chunk_np = audio_data[s_idx:e_idx]

                    chunk_tensor = torch.from_numpy(chunk_np.T).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = apply_model(model, chunk_tensor, device=device, shifts=0, split=True, overlap=0.10)

                    out_np = out.squeeze(0).cpu().numpy()
                    vocals_chunks.append(out_np[3].T)
                    bgm_chunks.append((out_np[0] + out_np[1] + out_np[2]).T)

                    if progress_callback:
                        done_sec = min(int(total_sec), int((c_idx + 1) * chunk_sec))
                        pct = int(10 + ((c_idx + 1) / num_chunks) * 24)
                        progress_callback(
                            pct,
                            f"Tách âm thanh AI (Demucs): {done_sec}s / {int(total_sec)}s ({int(((c_idx + 1) / num_chunks) * 100)}%)..."
                        )

                vocals_all = np.concatenate(vocals_chunks, axis=0)
                bgm_all = np.concatenate(bgm_chunks, axis=0)

                sf.write(vocals_out, vocals_all, sr, subtype="PCM_16")
                sf.write(background_out, bgm_all, sr, subtype="PCM_16")
                separated_by_demucs = True
                self.last_method = "demucs"
                print("✅ Đã tách âm thanh thành công bằng Demucs v4 (Chunked)!")
            except Exception as e:
                print(f"Lỗi tách bằng Demucs ({e}). Chuyển sang thuật toán FFmpeg DSP...")

        if not separated_by_demucs:
            # Band filters cannot isolate sources; label this result as approximate.
            self.last_method = "filter_approx"
            if progress_callback:
                progress_callback(
                    20,
                    "Demucs không khả dụng; đang tạo rãnh lọc DSP xấp xỉ (có thể còn lẫn giọng và nhạc).",
                )

            subprocess.run(
                [
                    self.ffmpeg_path, "-y", "-i", source_wav,
                    "-af", "highpass=f=200,lowpass=f=3600",
                    vocals_out
                ],
                capture_output=True,
                check=True
            )
            subprocess.run(
                [
                    self.ffmpeg_path, "-y", "-i", source_wav,
                    "-af", "stereotools=mlev=0:slev=1.4",
                    background_out
                ],
                capture_output=True,
                check=True
            )

        try:
            with open(self._method_metadata_path(vocals_out), "w", encoding="utf-8") as f:
                json.dump({"method": self.last_method}, f)
        except Exception:
            pass

        # Dọn dẹp tệp tạm raw_audio
        if os.path.exists(source_wav):
            try:
                os.remove(source_wav)
            except Exception:
                pass

        if progress_callback:
            label = "Demucs" if self.last_method == "demucs" else "bộ lọc DSP xấp xỉ"
            progress_callback(35, f"Hoàn tất xử lý âm thanh bằng {label}.")

        return vocals_out, background_out
