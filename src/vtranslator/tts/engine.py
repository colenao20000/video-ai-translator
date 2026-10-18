"""Local multi-voice TTS using the installed VieNeu model and cloned presets."""

import os
import re
import hashlib
import json
import time
import logging
from typing import Optional, List, Callable
from ..domain.voices import VoiceLibrary
from ..domain.segments import SubtitleSegment
from ..domain.characters import Character
from ..domain.enums import TtsStatus
from .model_manager import VieNeuModelManager

logger = logging.getLogger("vtranslator.tts.engine")


class TtsGenerationError(RuntimeError):
    """Raised when local dubbing cannot produce a valid audio asset."""


class TtsEngine:
    def __init__(
        self,
        temp_dir: str = "~/.vtranslator/tts_cache",
        preferred_engine: str = "vieneu",
        inter_segment_delay: float = 0.0,
        ffmpeg_path: str = "ffmpeg",
        vieneu_model_dir: Optional[str] = None,
    ):
        self.temp_dir = os.path.expanduser(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        # Legacy provider selection is accepted for older callers; this engine
        # always synthesizes with the local cloned-voice model.
        self.preferred_engine = "vieneu"
        if preferred_engine.lower() != "vieneu":
            logger.info("Ignoring legacy TTS provider '%s'; using local VieNeu.", preferred_engine)
        self.inter_segment_delay = max(0.0, inter_segment_delay)
        self.ffmpeg_path = ffmpeg_path
        self._last_synthesis_time = 0.0

        self._configured_model_dir = os.path.expanduser(vieneu_model_dir) if vieneu_model_dir else ""
        self.model_manager = VieNeuModelManager(custom_path=self._configured_model_dir or None)
        self._vieneu_tts = None

    def reload_config(
        self,
        ffmpeg_path: Optional[str] = None,
        vieneu_model_dir: Optional[str] = None,
    ) -> None:
        """Tải lại cấu hình khi người dùng thay đổi Settings."""
        self.preferred_engine = "vieneu"
        if ffmpeg_path:
            self.ffmpeg_path = ffmpeg_path
        if vieneu_model_dir is not None:
            configured_dir = os.path.expanduser(vieneu_model_dir) if vieneu_model_dir else ""
            if configured_dir != self._configured_model_dir:
                self._configured_model_dir = configured_dir
                self.model_manager = VieNeuModelManager(custom_path=configured_dir or None)
                self._vieneu_tts = None

    def _init_vieneu_backend(self) -> None:
        """Khởi tạo engine VieNeu on-device từ các tệp ONNX trong models/vieneu_models."""
        if self._vieneu_tts is not None:
            return
        if self.model_manager.is_loaded():
            try:
                from vieneu.v3turbo import V3TurboVieNeuTTS
                model_dir = self.model_manager.model_dir
                self._vieneu_tts = V3TurboVieNeuTTS(
                    backbone_repo=model_dir,
                    backend="onnx",
                    device="cpu",
                    precision="fp32",
                    onnx_subfolder="onnx_update",
                    onnx_dir=os.path.join(model_dir, "onnx_update"),
                )
                self._register_reference_voices()

                logger.info(
                    "Đã nạp VieNeu v3 Turbo ONNX với %s preset giọng.",
                    len(self._vieneu_tts._preset_voices),
                )
            except Exception as e:
                logger.warning(f"Không thể nạp VieNeu on-device backend ({e}).")
                self._vieneu_tts = None

    def _register_reference_voices(self) -> None:
        """Enroll bundled voice-reference clips once, offline, in the v3 schema."""
        if not self._vieneu_tts or not self.model_manager.model_dir:
            return

        from pathlib import Path

        reference_dir = Path(__file__).resolve().parents[3] / "resources" / "voices" / "gemini_references"
        added = False
        for voice in VoiceLibrary.get_all():
            if voice.id in self._vieneu_tts._preset_voices:
                continue
            reference_path = reference_dir / f"{voice.gemini_voice}.wav"
            if not reference_path.is_file():
                logger.warning("Missing local reference audio for voice %s: %s", voice.id, reference_path)
                continue
            try:
                self._vieneu_tts.add_voice(
                    voice.id,
                    str(reference_path),
                    denoise=False,
                    use_ref_codes=True,
                    description=voice.description,
                    gender=voice.gender.value,
                    save=False,
                )
                added = True
            except Exception:
                logger.exception("Could not enroll local voice reference %s", voice.id)

        if added:
            voices_path = os.path.join(self.model_manager.model_dir, "voices_v3_turbo.json")
            self._vieneu_tts.save_voices(path=voices_path)
            self.model_manager._load_metadata()

    def _resolve_voice_for_vieneu(self, voice_name: str) -> str:
        """Resolve the selected canonical voice to a preset present in the VieNeu model."""
        if not self._vieneu_tts:
            raise TtsGenerationError("Chưa cài hoặc chưa nạp mô hình VieNeu on-device.")
        vo = VoiceLibrary.resolve(voice_name)
        if vo and vo.id in self._vieneu_tts._preset_voices:
            return vo.id

        raise TtsGenerationError(f"Không tìm thấy giọng VieNeu đã chọn: {voice_name or '(trống)'}.")

    @staticmethod
    def validate_and_sync_segment_characters(
        segments: List[SubtitleSegment],
        characters: Optional[List[Character]] = None
    ) -> None:
        """
        Soát lại chuẩn từng câu thoại cho đúng nhân vật:
        - Đối chiếu character_id của segment với danh sách nhân vật.
        - Đồng bộ segment.voice_id theo lựa chọn hiện có trong hồ sơ nhân vật.
        - Nếu segment có voice_id không khớp với nhân vật, cập nhật lại và làm mới cache TTS.
        """
        if not characters or not segments:
            return

        char_by_id = {c.id: c for c in characters}
        char_by_speaker = {c.speaker_id: c for c in characters}

        for seg in segments:
            matched_char = None
            if seg.character_id in char_by_id:
                matched_char = char_by_id[seg.character_id]
            elif seg.speaker_id in char_by_speaker:
                matched_char = char_by_speaker[seg.speaker_id]
                seg.character_id = matched_char.id
            else:
                # An unresolved speaker must not inherit the first character's voice.
                continue

            target_voice = matched_char.voice_id or ""
            if seg.voice_id != target_voice:
                logger.info(f"Đồng bộ giọng đọc segment {seg.segment_id} theo nhân vật '{matched_char.name}': {seg.voice_id} -> {target_voice}")
                seg.sync_voice_from_character(target_voice)

    def _style_for_segment(
        self, seg: SubtitleSegment, characters: Optional[List[Character]] = None
    ) -> Optional[str]:
        return (seg.speech_style or "").strip() or None

    @staticmethod
    def _fallback_voice_for_segment(
        seg: SubtitleSegment,
        characters: Optional[List[Character]] = None,
    ) -> str:
        """Choose a stable local preset when speaker or voice assignment is missing."""
        char_by_id = {c.id: c for c in (characters or [])}
        char_by_speaker = {c.speaker_id: c for c in (characters or [])}
        character = char_by_id.get(seg.character_id) or char_by_speaker.get(seg.speaker_id)

        gender = getattr(getattr(character, "gender", None), "value", "unknown")
        candidates = [v for v in VoiceLibrary.get_all() if v.gender.value == gender]
        if not candidates:
            candidates = VoiceLibrary.get_all()
        if not candidates:
            raise TtsGenerationError("Danh mục giọng TTS cục bộ đang trống.")

        speaker_key = (seg.speaker_id or "").strip().lower()
        if not speaker_key or speaker_key in {"speaker_unknown", "unknown", "unassigned"}:
            preferred = "Gemini Kore" if gender == "female" else "Gemini Puck"
            if VoiceLibrary.resolve(preferred):
                return preferred
            return candidates[0].id

        digest = hashlib.sha256(speaker_key.encode("utf-8")).digest()
        return candidates[int.from_bytes(digest[:4], "big") % len(candidates)].id

    def _cache_identity(self, seg: SubtitleSegment, characters: Optional[List[Character]] = None) -> str:
        text = seg.translated_text.strip()
        if not text:
            raise TtsGenerationError(
                f"Câu {seg.segment_id} chưa có bản dịch; không thể tạo giọng lồng tiếng từ câu gốc."
            )
        voice = (seg.voice_id or "").strip()
        if VoiceLibrary.resolve(voice) is None:
            raise TtsGenerationError(
                f"Câu {seg.segment_id} chưa có giọng hợp lệ. Hãy chọn giọng trong hồ sơ nhân vật."
            )
        descriptor = {
            "text": text,
            "voice": voice,
            "style": self._style_for_segment(seg, characters),
            "engine": self.preferred_engine,
            "model": "vieneu-v3-local",
        }
        return hashlib.sha256(
            json.dumps(descriptor, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]

    def _has_cached_audio(self, seg: SubtitleSegment, characters: Optional[List[Character]] = None) -> bool:
        identity = self._cache_identity(seg, characters)
        safe_id = re.sub(r"[^\w.-]", "_", seg.segment_id)
        path = os.path.join(self.temp_dir, f"{safe_id}_{identity}.wav")
        return os.path.exists(path) and os.path.getsize(path) > 1000

    def generate_segment_audio(
        self,
        seg: SubtitleSegment,
        force_refresh: bool = False,
        characters: Optional[List[Character]] = None,
    ) -> str:
        """Sinh âm thanh thuyết minh AI cho 1 phân đoạn phụ đề, có soát chuẩn nhân vật và sắc thái kịch bản."""
        if characters:
            self.validate_and_sync_segment_characters([seg], characters)

        text = seg.translated_text.strip()
        if not text:
            seg.tts_status = TtsStatus.SKIPPED
            return ""

        selected_voice = VoiceLibrary.resolve(seg.voice_id or "")
        if selected_voice is None:
            selected_voice = VoiceLibrary.resolve(
                self._fallback_voice_for_segment(seg, characters)
            )
        if selected_voice is None:
            raise TtsGenerationError("Danh mục giọng TTS cục bộ đang trống.")
        voice_name = selected_voice.id
        if seg.voice_id != voice_name:
            seg.voice_id = voice_name
            seg.invalidate_tts()

        identity = self._cache_identity(seg, characters)
        safe_id = re.sub(r"[^\w.-]", "_", seg.segment_id)
        output_wav = os.path.join(self.temp_dir, f"{safe_id}_{identity}.wav")

        if not force_refresh and os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
            seg.tts_audio_path = output_wav
            seg.tts_status = TtsStatus.READY
            seg.tts_asset_id = identity
            seg.error_code = None
            try:
                import soundfile as sf
                seg.tts_duration_ms = int(sf.info(output_wav).duration * 1000)
            except Exception:
                pass
            return output_wav

        seg.tts_status = TtsStatus.GENERATING

        # Xác định sắc thái / chỉ dẫn diễn xuất phù hợp
        style_desc = self._style_for_segment(seg, characters)

        try:
            self._synthesize_sync(text, voice_name, output_wav, style_desc=style_desc)
            if not os.path.exists(output_wav) or os.path.getsize(output_wav) <= 100:
                raise TtsGenerationError("Engine không tạo được tệp âm thanh hợp lệ.")
            seg.tts_audio_path = output_wav
            seg.tts_status = TtsStatus.READY
            seg.tts_asset_id = identity
            seg.error_code = None
            return output_wav
        except Exception as e:
            logger.error(f"Lỗi sinh TTS ({seg.segment_id} - {voice_name}): {e}")
            seg.tts_status = TtsStatus.FAILED
            seg.error_code = str(e)
            return ""

    def generate_segments_batch(
        self,
        segments: List[SubtitleSegment],
        characters: Optional[List[Character]] = None,
        force_refresh: bool = False,
        inter_segment_delay: Optional[float] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[SubtitleSegment]:
        """Generate all translated lines with local VieNeu and the selected cloned voices."""
        if not segments:
            return segments

        untranslated = [
            s.segment_id for s in segments
            if s.original_text.strip() and not s.translated_text.strip()
        ]
        if untranslated:
            preview = ", ".join(untranslated[:8])
            raise TtsGenerationError(
                f"Còn {len(untranslated)} câu chưa dịch ({preview}). Hãy hoàn tất bản dịch trước khi tạo giọng."
            )

        # 1. Soát lại chuẩn từng câu thoại cho đúng nhân vật
        if characters:
            self.validate_and_sync_segment_characters(segments, characters)

        valid_lines = [s for s in segments if s.translated_text.strip()]
        for seg in valid_lines:
            if VoiceLibrary.resolve(seg.voice_id) is None:
                seg.voice_id = self._fallback_voice_for_segment(seg, characters)
                seg.invalidate_tts()

        valid_segments = [s for s in segments if s.translated_text.strip()]
        total_count = len(valid_segments)
        if total_count == 0:
            return segments

        delay_sec = max(0.0, inter_segment_delay if inter_segment_delay is not None else self.inter_segment_delay)
        for idx, seg in enumerate(valid_segments):
            needs_synthesis = force_refresh or not self._has_cached_audio(seg, characters)

            if needs_synthesis and delay_sec > 0 and self._last_synthesis_time > 0:
                wait_dur = delay_sec - (time.time() - self._last_synthesis_time)
                if wait_dur > 0:
                    time.sleep(wait_dur)

            self.generate_segment_audio(seg, force_refresh=force_refresh, characters=characters)
            if needs_synthesis and seg.tts_status == TtsStatus.READY:
                self._last_synthesis_time = time.time()

            if progress_callback:
                pct = int(((idx + 1) / total_count) * 100)
                char_label = ""
                if characters:
                    matched = next((c for c in characters if c.id == seg.character_id), None)
                    if matched:
                        char_label = f" [{matched.name}]"
                progress_callback(pct, f"Đang tổng hợp giọng cục bộ ({idx + 1}/{total_count}){char_label}: {seg.voice_id}...")

        failed = [s.segment_id for s in valid_segments if s.tts_status == TtsStatus.FAILED]
        if failed:
            details = [
                f"{s.segment_id}: {s.error_code}"
                for s in valid_segments
                if s.tts_status == TtsStatus.FAILED and s.error_code
            ]
            preview = "; ".join(details[:4]) or ", ".join(failed[:8])
            raise TtsGenerationError(
                f"Không tạo được giọng cho {len(failed)} câu ({preview}). Kiểm tra cấu hình TTS và thử lại."
            )
        return segments

    def preview_voice(self, voice_name: str, sample_text: str = "Xin chào, đây là giọng đọc thử nghiệm của tôi.") -> str:
        """Generate and return a short audio preview for the selected voice."""
        clean_voice = re.sub(r"[^\w]", "_", voice_name)
        out_wav = os.path.join(self.temp_dir, f"preview_{clean_voice}.wav")

        # Luôn tạo mới khi nghe thử để phản ánh chính xác chất giọng
        self._synthesize_sync(sample_text, voice_name, out_wav)
        return out_wav

    def _clean_tts_text(self, text: str, style_desc: Optional[str] = None) -> str:
        """Clean dialogue and use punctuation as a small local prosody cue."""
        t = (text or "").strip()
        t = re.sub(r'^["\'“”‘’]+|["\'“”‘’]+$', '', t).strip()
        # Preserve parenthetical and bracketed dialogue; they may contain real words.
        # Acting directions belong in speech_style, not destructive text cleanup.
        t = re.sub(r'\s+', ' ', t).strip()
        if not t:
            return ""

        # Bổ sung dấu câu cảm xúc phù hợp ngữ cảnh nếu câu chưa có dấu kết thúc
        if not t.endswith(('.', '!', '?', '...', '…', ':', ';', ',', '!!', '?!')):
            if style_desc:
                s_lower = style_desc.lower()
                if any(w in s_lower for w in [
                    "hét", "quát", "hoảng", "sợ", "cứu", "khẩn", "cảnh báo", "giận",
                    "tức giận", "vui vẻ", "hào hứng", "phấn khích", "excited", "angry", "happy", "urgent"
                ]):
                    t += "!"
                elif any(w in s_lower for w in ["hỏi", "sao", "nghi", "question", "curious"]):
                    t += "?"
                elif any(w in s_lower for w in [
                    "thì thầm", "đau", "yếu", "ngập ngừng", "buồn", "whisper", "sad", "hesitant"
                ]):
                    t += "..."
                else:
                    t += "."
            else:
                t += "."

        # VieNeu v3 supports these inline non-verbal cues. Its ``style``
        # argument is deprecated, so pass only cues recognized by the model.
        if style_desc:
            style = style_desc.lower()
            emotion_cue = None
            if any(word in style for word in ("cười", "vui vẻ", "hào hứng", "phấn khích", "laugh", "happy", "excited")):
                emotion_cue = "[cười]"
            elif any(word in style for word in ("thở dài", "buồn", "mệt mỏi", "thất vọng", "sigh", "sad", "tired")):
                emotion_cue = "[thở dài]"
            elif any(word in style for word in ("hắng giọng", "clear throat")):
                emotion_cue = "[hắng giọng]"
            if emotion_cue and emotion_cue not in t.lower():
                t = f"{t} {emotion_cue}"
        return t

    def _synthesize_sync(
        self,
        text: str,
        voice_name: str,
        output_wav: str,
        style_desc: Optional[str] = None,
    ) -> None:
        """Synthesize locally with VieNeu and the requested cloned voice preset."""
        clean_text = self._clean_tts_text(text, style_desc=style_desc)
        if not clean_text:
            raise TtsGenerationError("Câu thoại không còn nội dung sau khi làm sạch văn bản.")
        self._init_vieneu_backend()
        v_name = self._resolve_voice_for_vieneu(voice_name)
        temp = 0.82
        if style_desc:
            s_lower = style_desc.lower()
            if any(w in s_lower for w in [
                "hét", "quát", "hoảng", "sợ", "cứu", "khẩn", "giận", "hung", "tức giận",
                "vui vẻ", "hào hứng", "phấn khích", "excited", "angry", "happy", "urgent"
            ]):
                temp = 0.88
            elif any(w in s_lower for w in [
                "thì thầm", "trầm", "buồn", "nghẹn", "yếu", "tự sự", "bình tĩnh",
                "nghiêm nghị", "whisper", "sad", "calm", "serious"
            ]):
                temp = 0.76
            if any(w in s_lower for w in ["lo lắng", "bất an", "run rẩy", "worried", "nervous"]):
                temp = 0.86

        logger.info("Đang sinh TTS VieNeu cho giọng '%s' (preset='%s').", voice_name, v_name)
        try:
            audio = self._vieneu_tts.infer(clean_text, voice=v_name, temperature=temp)
            self._vieneu_tts.save(audio, output_wav)
            from ..audio.fit_to_slot import FitToSlotProcessor
            FitToSlotProcessor.trim_silence(output_wav, output_wav, pad_lead_ms=10, pad_tail_ms=30)
        except Exception as exc:
            raise TtsGenerationError(f"VieNeu không tạo được giọng '{voice_name}': {exc}") from exc
