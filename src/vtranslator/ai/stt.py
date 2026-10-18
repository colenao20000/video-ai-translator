"""Speech-to-Text (STT) service using FireRedASR2 or Faster-Whisper."""

import os
import sys
import json
import hashlib
import subprocess
from typing import List, Callable, Optional, Dict, Any
from ..domain.segments import SubtitleSegment
from ..domain.enums import TtsStatus
from ..services.cache_service import CacheService
from .firered_asr import FireRedAsrManager, FireRedAsrService

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SttService:
    def __init__(
        self,
        cache_service: CacheService,
        engine: str = "firered",
        whisper_model_size: str = "base",
        firered_model_dir: Optional[str] = None,
        ffmpeg_path: str = "ffmpeg",
        temp_dir: Optional[str] = None,
    ):
        self.cache_service = cache_service
        self.engine = engine  # "firered" or "whisper"
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir = os.path.expanduser(temp_dir or "~/.vtranslator/temp")
        self.whisper_model_size = whisper_model_size
        self.firered_manager = FireRedAsrManager(model_dir=firered_model_dir)
        self.firered_service = FireRedAsrService(self.firered_manager)

    def set_engine(self, engine: str) -> None:
        """Switch speech recognition engine ('firered' or 'whisper')."""
        self.engine = engine.lower()

    def get_engine_info(self) -> Dict[str, Any]:
        """Return information about configured STT engines."""
        firered_info = self.firered_manager.get_info()
        return {
            "current_engine": self.engine,
            "firered": firered_info,
            "whisper": {
                "model_size": self.whisper_model_size,
                "installed": True,
            },
        }

    def extract_subtitles(
        self,
        video_path: str,
        language: str = "auto",
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> List[SubtitleSegment]:
        """Extract subtitles from video in an isolated subprocess."""
        # 1. Check cache
        cache_key = f"{video_path}_{self.engine}"
        if not force_refresh:
            cached = self.cache_service.get_stt(cache_key) or self.cache_service.get_stt(video_path)
            if cached and len(cached) > 0:
                if progress_callback:
                    progress_callback(100, f"Loaded {len(cached)} segments from cache.")
                return [SubtitleSegment(**item) for item in cached]

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if progress_callback:
            progress_callback(5, "Extracting 16kHz audio stream...")

        # 2. Extract audio to 16kHz WAV
        os.makedirs(self.temp_dir, exist_ok=True)
        v_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:12]
        temp_audio = os.path.join(self.temp_dir, f"stt_{v_hash}.wav")

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_audio,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception as e:
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass
            raise RuntimeError(f"Audio extraction error: {e}")

        try:
            # 3. Route to selected engine
            if self.engine == "firered":
                try:
                    if not self.firered_manager.is_model_installed():
                        if progress_callback:
                            progress_callback(10, "Đang tải mô hình FireRedASR2-AED...")
                        self.firered_manager.download_model(progress_callback=progress_callback)
                    if progress_callback:
                        progress_callback(25, "Đang nhận diện giọng nói bằng FireRedASR2...")
                    result_segments = self.firered_service.transcribe(
                        temp_audio, progress_callback=progress_callback
                    )
                except Exception as err:
                    if progress_callback:
                        progress_callback(25, f"FireRedASR2 gặp lỗi ({err}). Đang dùng fallback...")
                    result_segments = self._extract_with_whisper(
                        temp_audio, language, v_hash, progress_callback
                    )
            else:
                result_segments = self._extract_with_whisper(
                    temp_audio, language, v_hash, progress_callback
                )

            # 4. Save cache
            self.cache_service.save_stt(cache_key, [s.model_dump() for s in result_segments])
            self.cache_service.save_stt(video_path, [s.model_dump() for s in result_segments])

            if progress_callback:
                progress_callback(100, f"Successfully extracted {len(result_segments)} subtitle segments.")

            return result_segments

        finally:
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass

    def _extract_with_whisper(
        self,
        audio_path: str,
        language: str,
        v_hash: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[SubtitleSegment]:
        """Fallback STT using faster-whisper with precise word-level timestamps."""
        temp_dir = os.path.expanduser("~/.vtranslator/temp")

        worker_script = f'''
import sys
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from faster_whisper import WhisperModel
import soundfile as sf

audio_path = sys.argv[1]
lang = sys.argv[2]

try:
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    model = WhisperModel("{self.whisper_model_size}", device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
    lang_param = None if lang in ("auto", "") else lang
    segments_gen, info = model.transcribe(
        audio_data,
        language=lang_param,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    out = []
    idx = 1
    for s in segments_gen:
        t = s.text.strip()
        if t:
            # Dùng mốc thời gian từ word_timestamps để căn chuẩn theo lời thoại thực tế
            if hasattr(s, "words") and s.words and len(s.words) > 0:
                seg_start_ms = int(s.words[0].start * 1000)
                seg_end_ms = int(s.words[-1].end * 1000)
            else:
                seg_start_ms = int(s.start * 1000)
                seg_end_ms = int(s.end * 1000)

            if seg_end_ms <= seg_start_ms:
                seg_end_ms = seg_start_ms + 400

            out.append({{
                "segment_id": f"seg_{{idx:03d}}",
                "speaker_id": "SPEAKER_00",
                "character_id": "char_01",
                "start_ms": seg_start_ms,
                "end_ms": seg_end_ms,
                "original_text": t,
                "translated_text": "",
                "voice_id": "Gemini Puck"
            }})
            idx += 1

    print(json.dumps(out, ensure_ascii=False))
except Exception as e:
    sys.stderr.write(f"WHISPER_ERROR: {{e}}\\n")
    sys.exit(1)
'''
        script_path = os.path.join(temp_dir, f"_whisper_worker_{v_hash}.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(worker_script)

        try:
            if progress_callback:
                progress_callback(40, "Running Whisper transcription...")

            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "2"

            res = subprocess.run(
                [sys.executable, script_path, audio_path, language],
                capture_output=True,
                text=True,
                env=env,
            )

            if res.returncode != 0:
                err_msg = res.stderr.strip() or res.stdout.strip()
                raise RuntimeError(f"Whisper transcription error (code {res.returncode}): {err_msg}")

            data = json.loads(res.stdout)
            return [SubtitleSegment(**item) for item in data]

        finally:
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except Exception:
                    pass
