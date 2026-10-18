"""FireRedASR2-CTC INT8 model manager and inference service."""

import os
import sys
import json
import re
import hashlib
import tarfile
import urllib.request
import subprocess
from typing import List, Optional, Callable, Dict, Any
from ..domain.segments import SubtitleSegment


class FireRedAsrManager:
    """Manages FireRedASR2-CTC INT8 model files and downloads."""

    ARCHIVE_URL = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2"
    )
    DEFAULT_MODEL_DIR = os.path.expanduser("~/.vtranslator/models/FireRedASR2-CTC-int8")
    DEFAULT_VAD_DIR = os.path.expanduser("~/.vtranslator/models/FireRedVAD")

    def __init__(self, model_dir: Optional[str] = None, vad_dir: Optional[str] = None):
        self.model_dir = os.path.abspath(model_dir or self.DEFAULT_MODEL_DIR)
        self.vad_dir = os.path.abspath(vad_dir or self.DEFAULT_VAD_DIR)

    def is_model_installed(self) -> bool:
        """Check if FireRedASR2-CTC INT8 model files exist on disk."""
        if not os.path.isdir(self.model_dir):
            return False
        model_file = os.path.join(self.model_dir, "model.int8.onnx")
        tokens_file = os.path.join(self.model_dir, "tokens.txt")
        return (
            os.path.isfile(model_file)
            and os.path.getsize(model_file) > 500 * 1024 * 1024
            and os.path.isfile(tokens_file)
        )

    def is_vad_installed(self) -> bool:
        """Check if FireRedVAD model files exist."""
        sub_vad = os.path.join(self.vad_dir, "VAD", "model.pth.tar")
        if os.path.isfile(sub_vad):
            return True
        return os.path.isfile(os.path.join(self.vad_dir, "model.pth.tar"))

    def get_vad_model_dir(self) -> Optional[str]:
        """Return the subfolder containing VAD model if present."""
        sub_vad = os.path.join(self.vad_dir, "VAD")
        if os.path.isfile(os.path.join(sub_vad, "model.pth.tar")):
            return sub_vad
        if os.path.isfile(os.path.join(self.vad_dir, "model.pth.tar")):
            return self.vad_dir
        return None

    def get_info(self) -> Dict[str, Any]:
        """Get model installation status and metadata."""
        installed = self.is_model_installed()
        size_mb = 0
        if installed:
            try:
                size_mb = sum(
                    os.path.getsize(os.path.join(self.model_dir, f))
                    for f in os.listdir(self.model_dir)
                    if os.path.isfile(os.path.join(self.model_dir, f))
                ) / (1024 * 1024)
            except Exception:
                pass

        return {
            "installed": installed,
            "model_dir": self.model_dir,
            "vad_installed": self.is_vad_installed(),
            "vad_dir": self.vad_dir,
            "size_mb": round(size_mb, 1),
            "engine_name": "FireRedASR2-CTC (INT8 Quantized)",
        }

    def download_model(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> bool:
        """Download and extract FireRedASR2-CTC INT8 model weights."""
        os.makedirs(self.model_dir, exist_ok=True)

        if self.is_model_installed():
            if progress_callback:
                progress_callback(100, "Mô hình FireRedASR2-CTC INT8 đã có sẵn.")
            return True

        temp_archive = os.path.join(os.path.dirname(self.model_dir), "fireredasr2_ctc_int8.tar.bz2")

        if progress_callback:
            progress_callback(5, "Đang tải mô hình FireRedASR2-CTC INT8 siêu nhẹ (~496MB)...")

        req = urllib.request.Request(self.ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as response, open(temp_archive, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 520516278))
            downloaded = 0
            block_size = 1024 * 512

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    pct = int(5 + (downloaded / total_size) * 80)
                    progress_callback(pct, f"Tải FireRedASR2-CTC: {downloaded // (1024*1024)}/{total_size // (1024*1024)} MB")

        if progress_callback:
            progress_callback(88, "Đang giải nén mô hình...")

        with tarfile.open(temp_archive, "r:bz2") as archive:
            for member in archive.getmembers():
                filename = os.path.basename(member.name)
                if filename in ("model.int8.onnx", "tokens.txt"):
                    fileobj = archive.extractfile(member)
                    if fileobj:
                        target_file = os.path.join(self.model_dir, filename)
                        with open(target_file, "wb") as f_out:
                            f_out.write(fileobj.read())

        if os.path.exists(temp_archive):
            try:
                os.remove(temp_archive)
            except Exception:
                pass

        if progress_callback:
            progress_callback(100, "Hoàn tất cài đặt mô hình FireRedASR2-CTC INT8!")

        return self.is_model_installed()


class FireRedAsrService:
    """Speech recognition inference engine using FireRedASR2-CTC INT8 via sherpa-onnx."""

    def __init__(self, model_manager: Optional[FireRedAsrManager] = None):
        self.manager = model_manager or FireRedAsrManager()

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[SubtitleSegment]:
        """Transcribe 16kHz WAV audio using FireRedASR2-CTC INT8."""
        if not self.manager.is_model_installed():
            raise RuntimeError(
                f"Mô hình FireRedASR2-CTC INT8 chưa được cài đặt tại: {self.manager.model_dir}. "
                "Vui lòng tải mô hình trước khi tiếp tục."
            )

        model_file = os.path.join(self.manager.model_dir, "model.int8.onnx")
        tokens_file = os.path.join(self.manager.model_dir, "tokens.txt")
        vad_dir = self.manager.get_vad_model_dir() or ""

        temp_dir = os.path.expanduser("~/.vtranslator/temp")
        os.makedirs(temp_dir, exist_ok=True)

        worker_script = f"""
import sys
import os
import json
import re
import soundfile as sf
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_path = sys.argv[1]
tokens_path = sys.argv[2]
audio_path = sys.argv[3]
vad_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "none" else ""

# 1. Load Audio
wav_np, sr = sf.read(audio_path, dtype="float32")
if wav_np.ndim > 1:
    wav_np = wav_np.mean(axis=1)
assert sr == 16000, f"Expected 16kHz audio, got {{sr}}"
total_duration_s = len(wav_np) / sr

vad_segments = []

# 2. VAD Segmentation
if vad_dir and os.path.exists(vad_dir):
    try:
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
        vad_cfg = FireRedVadConfig(use_gpu=False)
        vad = FireRedVad.from_pretrained(vad_dir, vad_cfg)
        vad_res, _ = vad.detect(audio_path)
        vad_segments = vad_res.get("timestamps", [])
    except Exception as e:
        sys.stderr.write(f"VAD warning: {{e}}\\n")

if not vad_segments:
    # 15s chunks fallback
    chunk_len = 15.0
    cur = 0.0
    while cur < total_duration_s:
        end = min(total_duration_s, cur + chunk_len)
        vad_segments.append((cur, end))
        cur = end

# 3. Load FireRedASR2-CTC INT8 via sherpa-onnx
import sherpa_onnx
recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
    model=model_path,
    tokens=tokens_path,
    num_threads=2,
    provider="cpu",
    decoding_method="greedy_search",
    debug=False
)

out_segments = []
idx = 1
total_chunks = len(vad_segments)

for c_idx, (start_s, end_s) in enumerate(vad_segments):
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    if end_sample <= start_sample + int(0.15 * sr):
        continue

    chunk_data = wav_np[start_sample:end_sample]

    stream = recognizer.create_stream()
    stream.accept_waveform(sr, chunk_data)
    recognizer.decode_stream(stream)
    res = stream.result

    raw_text = res.text.strip() if hasattr(res, "text") else ""
    # Remove CTC silence tokens
    raw_text = re.sub(r"(<blank>)|(<sil>)", "", raw_text).strip()
    if not raw_text:
        continue

    # Căn chỉnh timeline chuẩn xác theo mốc token thực tế
    if hasattr(res, "timestamps") and hasattr(res, "tokens") and len(res.timestamps) == len(res.tokens) and len(res.timestamps) > 0:
        groups = []
        cur_toks = [res.tokens[0]]
        cur_ts = [res.timestamps[0]]
        for t_idx in range(1, len(res.timestamps)):
            gap = res.timestamps[t_idx] - res.timestamps[t_idx - 1]
            if gap > 0.85 and cur_toks:
                groups.append((cur_toks, cur_ts))
                cur_toks = [res.tokens[t_idx]]
                cur_ts = [res.timestamps[t_idx]]
            else:
                cur_toks.append(res.tokens[t_idx])
                cur_ts.append(res.timestamps[t_idx])
        if cur_toks:
            groups.append((cur_toks, cur_ts))

        for grp_toks, grp_ts in groups:
            sub_text = "".join(grp_toks).strip()
            sub_text = re.sub(r"(<blank>)|(<sil>)", "", sub_text).strip()
            if not sub_text:
                continue
            seg_start_ms = int(max(start_s, start_s + grp_ts[0] - 0.15) * 1000)
            seg_end_ms = int(min(end_s, start_s + grp_ts[-1] + 0.35) * 1000)
            if seg_end_ms <= seg_start_ms:
                seg_end_ms = seg_start_ms + 400

            out_segments.append({{
                "segment_id": f"seg_{{idx:03d}}",
                "speaker_id": "SPEAKER_00",
                "character_id": "char_01",
                "start_ms": seg_start_ms,
                "end_ms": seg_end_ms,
                "original_text": sub_text,
                "translated_text": "",
                "voice_id": "Gemini Puck"
            }})
            idx += 1
    else:
        out_segments.append({{
            "segment_id": f"seg_{{idx:03d}}",
            "speaker_id": "SPEAKER_00",
            "character_id": "char_01",
            "start_ms": int(start_s * 1000),
            "end_ms": int(end_s * 1000),
            "original_text": raw_text,
            "translated_text": "",
            "voice_id": "Gemini Puck"
        }})
        idx += 1

    pct = int(10 + (c_idx + 1) / total_chunks * 80)
    sys.stderr.write(f"PROGRESS:{{pct}}:Nhận diện câu {{c_idx+1}}/{{total_chunks}}\\n")
    sys.stderr.flush()

out_json_file = sys.argv[5]
with open(out_json_file, "w", encoding="utf-8") as f:
    json.dump(out_segments, f, ensure_ascii=False)
"""

        v_hash_audio = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        script_path = os.path.join(temp_dir, f"_firered_ctc_worker_{os.getpid()}_{v_hash_audio}.py")
        out_json_path = os.path.join(temp_dir, f"_firered_out_{os.getpid()}_{v_hash_audio}.json")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(worker_script)

        try:
            if progress_callback:
                progress_callback(10, "Đang nạp mô hình FireRedASR2-CTC INT8...")

            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "2"

            process = subprocess.Popen(
                [
                    sys.executable,
                    script_path,
                    model_file,
                    tokens_file,
                    audio_path,
                    vad_dir if vad_dir else "none",
                    out_json_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            stderr_lines = []
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stderr_lines.append(line)
                    if "PROGRESS:" in line:
                        parts = line.strip().split(":", 2)
                        if len(parts) >= 3 and progress_callback:
                            try:
                                p = int(parts[1])
                                msg = parts[2]
                                progress_callback(p, f"FireRedASR2-CTC: {msg}")
                            except ValueError:
                                pass

            process.wait()
            if process.returncode != 0:
                err_msg = "".join(stderr_lines[-10:])
                raise RuntimeError(f"FireRedASR2-CTC worker error: {err_msg.strip()}")

            if not os.path.exists(out_json_path):
                raise RuntimeError("FireRedASR2-CTC worker không thể ghi tệp kết quả nhận diện.")

            with open(out_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return [SubtitleSegment(**item) for item in data]

        finally:
            for p_to_clean in (script_path, out_json_path):
                if os.path.exists(p_to_clean):
                    try:
                        os.remove(p_to_clean)
                    except Exception:
                        pass
