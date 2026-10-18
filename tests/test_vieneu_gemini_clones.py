"""Unit tests for VieNeu-TTS On-Device with 10 Cloned Gemini Voices."""

import os
import io
import soundfile as sf
import pytest

from src.vtranslator.tts.model_manager import VieNeuModelManager
from src.vtranslator.tts.engine import TtsEngine
from src.vtranslator.domain.voices import VoiceLibrary, AVAILABLE_VOICES
from src.vtranslator.domain.segments import SubtitleSegment
from src.vtranslator.domain.enums import SpeakerGender, TtsStatus


class TestVieNeuGeminiClones:
    """Kiểm thử tích hợp mô hình VieNeu On-Device với 10 giọng clone từ Gemini."""

    def test_model_manager_loads_cloned_voices(self):
        mgr = VieNeuModelManager()
        clones = mgr.get_cloned_gemini_voices()
        assert len(clones) >= 10
        assert "Gemini Puck" in clones
        assert "Gemini Charon" in clones
        assert "Gemini Kore" in clones
        assert "Gemini Zephyr" in clones
        assert "Gemini Fenrir" in clones
        assert "Gemini Aoede" in clones
        assert "Gemini Sulafat" in clones
        assert "Gemini Orus" in clones
        assert "Gemini Gacrux" in clones
        assert "Gemini Leda" in clones

        info = mgr.get_info()
        assert info["cloned_count"] >= 10
        assert "Gemini Clone" in info["status_text"]

    def test_voice_library_recognizes_cloned_and_aliases(self):
        # Kiểm tra truy vấn trực tiếp bằng tên clone
        v_puck = VoiceLibrary.get_by_id("Gemini Puck")
        assert v_puck.id == "Gemini Puck"
        assert v_puck.gender == SpeakerGender.MALE

        v_kore = VoiceLibrary.get_by_id("Gemini Kore")
        assert v_kore.id == "Gemini Kore"
        assert v_kore.gender == SpeakerGender.FEMALE

        # Kiểm tra truy vấn qua alias ngắn gọn
        assert VoiceLibrary.get_by_id("puck").gemini_voice == "Puck"
        assert VoiceLibrary.get_by_id("charon").gemini_voice == "Charon"
        assert VoiceLibrary.get_by_id("kore").gemini_voice == "Kore"
        assert VoiceLibrary.get_by_id("zephyr").gemini_voice == "Zephyr"
        assert VoiceLibrary.get_by_id("fenrir").gemini_voice == "Fenrir"
        assert VoiceLibrary.get_by_id("orus").gemini_voice == "Orus"

    def test_engine_init_registers_clones_and_aliases(self):
        engine = TtsEngine(preferred_engine="vieneu")
        engine._init_vieneu_backend()
        assert engine._vieneu_tts is not None
        assert "Gemini Puck" in engine._vieneu_tts._preset_voices
        assert "Gemini Charon" in engine._vieneu_tts._preset_voices

        # Kiểm tra alias
        assert engine._vieneu_tts.resolve_voice_name("puck") == "Gemini Puck"
        assert engine._vieneu_tts.resolve_voice_name("charon") == "Gemini Charon"
        assert engine._vieneu_tts.resolve_voice_name("kore") == "Gemini Kore"

    def test_offline_vieneu_synthesis_runs_without_api(self):
        engine = TtsEngine(preferred_engine="vieneu")
        seg = SubtitleSegment(
            segment_id="test_offline_1",
            start_ms=0,
            end_ms=2000,
            original_text="Xin chào, kiểm tra lồng tiếng offline VieNeu.",
            translated_text="Xin chào, kiểm tra lồng tiếng offline VieNeu.",
            voice_id="Gemini Puck"
        )
        wav_path = engine.generate_segment_audio(seg, force_refresh=True)
        assert wav_path != ""
        assert os.path.exists(wav_path)
        assert os.path.getsize(wav_path) > 5000
        assert seg.tts_status == TtsStatus.READY

        # Kiểm tra audio hợp lệ
        data, sr = sf.read(wav_path)
        assert sr in (24000, 48000)
        assert len(data) > 0

    def test_engine_fallback_from_gemini_to_vieneu_clone(self, monkeypatch):
        engine = TtsEngine(preferred_engine="gemini")

        # Mock Gemini TTS để ném lỗi Rate Limit 429
        def mock_failing_synth(*args, **kwargs):
            raise RuntimeError("Gemini 429 Rate Limit Exceeded: Daily quota reached")

        if engine._gemini_tts:
            monkeypatch.setattr(engine._gemini_tts, "synthesize", mock_failing_synth)

        seg = SubtitleSegment(
            segment_id="test_fallback_1",
            start_ms=0,
            end_ms=2000,
            translated_text="Thử nghiệm dự phòng tự động sang VieNeu.",
            voice_id="Charon"
        )
        wav_path = engine.generate_segment_audio(seg, force_refresh=True)
        # Phải fallback sang VieNeu thành công mà không bị đơ hoặc lỗi
        assert wav_path != ""
        assert os.path.exists(wav_path)
        assert os.path.getsize(wav_path) > 5000
        assert seg.tts_status == TtsStatus.READY

    def test_stock_voices_completely_purged(self):
        """Xác nhận đã xoá sạch 100% giọng gốc của VieNeu, chỉ còn đúng 10 giọng Gemini Clone."""
        all_voices = VoiceLibrary.get_all()
        assert len(all_voices) == 10
        for v in all_voices:
            assert v.id.startswith("Gemini ")

        # Không còn bất kỳ giọng gốc nào trong danh sách
        stock_names = ["Minh Đức", "Trúc Ly", "Phạm Tuyên", "Thái Sơn", "Minh Quân", "Adam", "Mai Anh"]
        for name in stock_names:
            assert not any(v.id == name for v in all_voices)
            # Khi tra cứu tên cũ phải tự động trỏ về Gemini Clone tương ứng
            assert VoiceLibrary.get_by_id(name).id.startswith("Gemini ")

        # Kiểm tra backend VieNeu chỉ chứa đúng 10 preset clone
        engine = TtsEngine(preferred_engine="vieneu")
        engine._init_vieneu_backend()
        assert len(engine._vieneu_tts._preset_voices) == 10
        for name in stock_names:
            assert name not in engine._vieneu_tts._preset_voices
            # Hàm resolve phải luôn trả về giọng Clone
            assert engine._resolve_voice_for_vieneu(name).startswith("Gemini ")
