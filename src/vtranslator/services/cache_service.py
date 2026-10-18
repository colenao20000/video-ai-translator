"""Cache Service for Video Analysis, Diarization, STT, and Translations."""

import os
import json
import hashlib
from typing import Optional, List, Dict, Any
from ..database.repositories import AnalysisCacheRepository


class CacheService:
    def __init__(self, cache_repo: AnalysisCacheRepository, local_analysis_dir: str = "analysis"):
        self.cache_repo = cache_repo
        self.local_analysis_dir = local_analysis_dir
        os.makedirs(self.local_analysis_dir, exist_ok=True)

    @staticmethod
    def compute_video_hash(video_path: str) -> str:
        """
        Tạo fingerprint nhẹ cho video dựa trên kích thước, thời gian sửa và hai vùng đầu/cuối.
        """
        if not os.path.exists(video_path):
            return ""

        file_size = os.path.getsize(video_path)
        modified_ns = os.stat(video_path).st_mtime_ns
        hasher = hashlib.md5()
        hasher.update(str(file_size).encode("utf-8"))
        hasher.update(str(modified_ns).encode("utf-8"))

        with open(video_path, "rb") as f:
            # 1MB đầu
            head = f.read(1024 * 1024)
            hasher.update(head)
            # 1MB cuối
            if file_size > 1024 * 1024:
                f.seek(max(0, file_size - 1024 * 1024))
                tail = f.read(1024 * 1024)
                hasher.update(tail)

        return hasher.hexdigest()

    def get_diarization(self, video_path: str) -> Optional[List[Dict[str, Any]]]:
        v_hash = self.compute_video_hash(video_path)
        if not v_hash:
            return None
        return self.cache_repo.get(v_hash, "diarization")

    def save_diarization(self, video_path: str, segments: List[Dict[str, Any]]) -> None:
        v_hash = self.compute_video_hash(video_path)
        if v_hash:
            self.cache_repo.set(v_hash, "diarization", segments)

    def get_speakers_analysis(self, video_path: str) -> Optional[List[Dict[str, Any]]]:
        """Đọc cache speakers từ DB hoặc file analysis/speakers.json để tương thích."""
        v_hash = self.compute_video_hash(video_path)
        if not v_hash:
            return None

        cached = self.cache_repo.get(v_hash, "speakers")
        if cached:
            return cached

        # Kiểm tra file analysis/speakers.json
        speakers_file = os.path.join(self.local_analysis_dir, "speakers.json")
        if os.path.exists(speakers_file):
            try:
                with open(speakers_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("video_hash") == v_hash:
                        return data.get("speakers", [])
            except Exception:
                pass
        return None

    def save_speakers_analysis(self, video_path: str, speakers: List[Dict[str, Any]]) -> None:
        v_hash = self.compute_video_hash(video_path)
        if not v_hash:
            return

        self.cache_repo.set(v_hash, "speakers", speakers)

        # Lưu đồng thời ra analysis/speakers.json
        speakers_file = os.path.join(self.local_analysis_dir, "speakers.json")
        try:
            with open(speakers_file, "w", encoding="utf-8") as f:
                json.dump({
                    "video_hash": v_hash,
                    "video_path": video_path,
                    "speakers": speakers
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_stt(self, video_path: str) -> Optional[List[Dict[str, Any]]]:
        v_hash = self.compute_video_hash(video_path)
        if not v_hash:
            return None
        return self.cache_repo.get(v_hash, "stt")

    def save_stt(self, video_path: str, segments: List[Dict[str, Any]]) -> None:
        v_hash = self.compute_video_hash(video_path)
        if v_hash:
            self.cache_repo.set(v_hash, "stt", segments)

    def get_translation(
        self, text: str, src_lang: str, target_lang: str, context_key: Optional[str] = None
    ) -> Optional[str]:
        context = f"{context_key}:" if context_key else ""
        key = hashlib.md5(f"{src_lang}:{target_lang}:{context}{text}".encode("utf-8")).hexdigest()
        cached = self.cache_repo.get(key, "translation")
        if cached and isinstance(cached, dict):
            return cached.get("translated")
        return None

    def save_translation(
        self,
        text: str,
        src_lang: str,
        target_lang: str,
        translated: str,
        context_key: Optional[str] = None,
    ) -> None:
        context = f"{context_key}:" if context_key else ""
        key = hashlib.md5(f"{src_lang}:{target_lang}:{context}{text}".encode("utf-8")).hexdigest()
        self.cache_repo.set(key, "translation", {"translated": translated})

    def clear_all_cache(self, extra_directories: Optional[List[str]] = None) -> None:
        """Xóa sạch toàn bộ bộ nhớ đệm (analysis_cache, tts_cache, temp)."""
        import shutil
        conn = self.cache_repo.db.get_connection()
        with conn:
            conn.execute("DELETE FROM analysis_cache;")

        directories = ["~/.vtranslator/tts_cache", "~/.vtranslator/temp", "~/.vtranslator/avatars", self.local_analysis_dir]
        directories.extend(extra_directories or [])
        for d in dict.fromkeys(os.path.expanduser(path) for path in directories):
            exp = d
            if os.path.exists(exp):
                shutil.rmtree(exp, ignore_errors=True)
                os.makedirs(exp, exist_ok=True)
