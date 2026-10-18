"""Download and locate the local VieNeu-TTS v3 Turbo ONNX model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import quote

import httpx


_PROJECT_HF_CACHE = Path(__file__).resolve().parents[3] / "models" / "huggingface_cache" / "hub"
os.environ.setdefault("HF_HUB_CACHE", str(_PROJECT_HF_CACHE))


class VieNeuModelManager:
    MODEL_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
    MODEL_BASE_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
    MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "vieneu_models"
    CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

    GRAPH_FILES = (
        "config.json",
        "tokenizer.json",
        "vieneu_prefill.onnx",
        "vieneu_decode_step.onnx",
        "vieneu_acoustic_cached.onnx",
        "vieneu_backbone_shared.data",
        "vieneu_v3_heads.npz",
    )
    CODEC_FILES = (
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_shared.data",
        "moss_audio_tokenizer_decode_step.onnx",
        "codec_browser_onnx_meta.json",
        "moss_audio_tokenizer_encode.onnx",
        "moss_audio_tokenizer_encode.data",
    )
    ROOT_FILES = ("config.json", "speaker_encoder.onnx", "denoiser.onnx")
    REQUIRED_FILES = tuple(
        [f"onnx_update/{name}" for name in GRAPH_FILES] + list(ROOT_FILES[1:])
    )

    CANDIDATE_PATHS = (
        str(MODEL_DIR),
        os.path.expanduser("~/.vtranslator/models/vieneu_models"),
        os.path.expanduser("~/Downloads/vieneu_models"),
    )

    def __init__(self, custom_path: Optional[str] = None):
        self.model_dir: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.voices_metadata: Dict[str, Any] = {"presets": {}}
        self.cloned_gemini_voices: Dict[str, Any] = {}

        if custom_path and self.validate_model_dir(custom_path):
            self.model_dir = os.path.abspath(os.path.expanduser(custom_path))
        else:
            self.model_dir = self.find_model_dir()

        if self.model_dir:
            self._load_metadata()

    @classmethod
    def validate_model_dir(cls, directory: str) -> bool:
        if not directory or not os.path.isdir(os.path.expanduser(directory)):
            return False
        root = Path(os.path.expanduser(directory))
        return all((root / name).is_file() for name in cls.REQUIRED_FILES)

    @classmethod
    def find_model_dir(cls) -> Optional[str]:
        for directory in cls.CANDIDATE_PATHS:
            if cls.validate_model_dir(directory):
                return os.path.abspath(os.path.expanduser(directory))
        return None

    def _load_metadata(self) -> None:
        if not self.model_dir:
            return

        root = Path(self.model_dir)
        try:
            self.config = self._read_json(root / "config.json")
        except (OSError, ValueError):
            self.config = {}

        # VieNeu v3 stores custom voice embeddings in the same schema it reads
        # from a model directory. Older Gemini embeddings are deliberately not
        # loaded because they were produced for an incompatible model version.
        try:
            self.voices_metadata = self._read_json(root / "voices_v3_turbo.json")
        except (OSError, ValueError):
            self.voices_metadata = {"presets": {}}

        presets = self.voices_metadata.get("presets", {})
        self.cloned_gemini_voices = {
            name: data
            for name, data in presets.items()
            if name.lower().startswith("gemini ") and isinstance(data, dict)
        }

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        import json

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def is_loaded(self) -> bool:
        return bool(self.model_dir and self.validate_model_dir(self.model_dir))

    def get_cloned_gemini_voices(self) -> Dict[str, Any]:
        return self.cloned_gemini_voices

    def get_info(self) -> Dict[str, Any]:
        if not self.is_loaded():
            return {
                "loaded": False,
                "status_text": "Chưa tải mô hình VieNeu-TTS v3 Turbo",
                "model_dir": str(self.MODEL_DIR),
                "voice_count": 0,
                "cloned_count": 0,
            }

        total_count = len(self.voices_metadata.get("presets", {}))
        cloned_count = len(self.cloned_gemini_voices)
        return {
            "loaded": True,
            "status_text": (
                f"VieNeu-TTS v3 Turbo ONNX đã sẵn sàng · {total_count} giọng"
                f" · {cloned_count} giọng mẫu đã đăng ký"
            ),
            "model_dir": self.model_dir or "",
            "voice_count": total_count,
            "cloned_count": cloned_count,
        }

    def set_custom_dir(self, directory: str) -> Tuple[bool, str]:
        expanded = os.path.abspath(os.path.expanduser(directory or ""))
        if self.validate_model_dir(expanded):
            self.model_dir = expanded
            self._load_metadata()
            return True, f"Mô hình VieNeu đã nạp từ: {expanded}"
        return False, "Thư mục thiếu các tệp ONNX bắt buộc của VieNeu v3 Turbo."

    def download_model(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> bool:
        """Download the fp32 ONNX checkpoint into the project models directory."""
        destination = self.MODEL_DIR
        graph_dir = destination / "onnx_update"
        graph_dir.mkdir(parents=True, exist_ok=True)
        files = [
            (f"onnx_update/{name}", graph_dir / name) for name in self.GRAPH_FILES
        ] + [
            (name, destination / name) for name in self.ROOT_FILES
        ]

        timeout = httpx.Timeout(connect=30.0, read=90.0, write=90.0, pool=30.0)
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            file_weights: Dict[str, int] = {}
            for remote_name, output_path in files:
                if output_path.is_file() and output_path.stat().st_size > 0:
                    file_weights[remote_name] = output_path.stat().st_size
                    continue
                url = f"{self.MODEL_BASE_URL}/{quote(remote_name, safe='/')}"
                try:
                    head = client.head(url)
                    size = int(
                        head.headers.get("content-length")
                        or head.headers.get("x-linked-size")
                        or 0
                    )
                except Exception:
                    size = 0
                file_weights[remote_name] = max(size, 1)

            total_weight = max(1, sum(file_weights.values()))
            completed_weight = 0
            for index, (remote_name, output_path) in enumerate(files):
                if output_path.is_file() and output_path.stat().st_size > 0:
                    completed_weight += file_weights[remote_name]
                    if progress_callback:
                        progress_callback(
                            int(completed_weight * 88 / total_weight),
                            f"Đã có {remote_name}",
                        )
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path = output_path.with_name(output_path.name + ".part")
                url = f"{self.MODEL_BASE_URL}/{quote(remote_name, safe='/')}"
                if progress_callback:
                    progress_callback(
                        int(completed_weight * 88 / total_weight),
                        f"Đang tải {remote_name} ({index + 1}/{len(files)})",
                    )

                try:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        total = int(response.headers.get("content-length", "0") or 0)
                        if total > 0 and total != file_weights[remote_name]:
                            total_weight += total - file_weights[remote_name]
                            file_weights[remote_name] = total
                        downloaded = 0
                        with partial_path.open("wb") as output:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                output.write(chunk)
                                downloaded += len(chunk)
                                if progress_callback and total > 0:
                                    fraction = min(downloaded / total, 1.0)
                                    overall = int(
                                        (completed_weight + downloaded)
                                        * 88
                                        / max(1, total_weight)
                                    )
                                    progress_callback(
                                        overall,
                                        f"Đang tải {remote_name}: {downloaded // (1024 * 1024)} MB",
                                    )
                    if partial_path.stat().st_size == 0:
                        raise RuntimeError(f"Tệp tải về bị rỗng: {remote_name}")
                    os.replace(partial_path, output_path)
                    completed_weight += file_weights[remote_name]
                except Exception:
                    try:
                        partial_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise

        if not self.validate_model_dir(str(destination)):
            missing = [
                name
                for name in self.REQUIRED_FILES
                if not (destination / name).is_file()
            ]
            raise RuntimeError("Thiếu tệp mô hình: " + ", ".join(missing))

        # Seed VieNeu's normal Hugging Face cache with the codec used by both
        # synthesis and local voice enrollment, so inference works offline.
        from huggingface_hub import hf_hub_download

        for index, filename in enumerate(self.CODEC_FILES):
            if progress_callback:
                progress_callback(
                    95 + int(index * 5 / len(self.CODEC_FILES)),
                    f"Đang chuẩn bị codec ONNX {index + 1}/{len(self.CODEC_FILES)}",
                )
            hf_hub_download(repo_id=self.CODEC_REPO, filename=filename)

        self.model_dir = str(destination.resolve())
        self._load_metadata()
        if progress_callback:
            progress_callback(100, "Đã tải xong VieNeu-TTS v3 Turbo ONNX.")
        return True
