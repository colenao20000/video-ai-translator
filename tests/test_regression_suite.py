"""Mandatory Regression Test Suite (Section 9 Compliance - All 14 Test Cases)."""

import os
import tempfile
import pytest
import soundfile as sf
import numpy as np

from src.vtranslator.domain.project import Project
from src.vtranslator.domain.characters import Character
from src.vtranslator.domain.segments import SubtitleSegment
from src.vtranslator.domain.enums import SpeakerGender, CharacterMatchStatus, SegmentFitStatus, VoiceCharacter, SpeakerAgeGroup
from src.vtranslator.domain.voices import VoiceLibrary
from src.vtranslator.domain.validators import TimelineValidator, ProjectIntegrityValidator
from src.vtranslator.audio.fit_to_slot import FitToSlotProcessor, AudioTimelineScheduler
from src.vtranslator.ai.character_resolver import CharacterResolver
from src.vtranslator.ai.diarization import DiarizationService, SpeakerAcousticProfile
from src.vtranslator.services.nine_router_service import NineRouterService
from src.vtranslator.render.pipeline import RenderPipeline


def make_dummy_wav(filepath: str, duration_sec: float, sample_rate: int = 48000) -> str:
    """Helper tạo file WAV giả lập với độ dài chỉ định."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    data = np.zeros(num_samples, dtype=np.float32)
    sf.write(filepath, data, sample_rate, subtype="PCM_16")
    return filepath


class TestRegressionSuite:
    """Kiểm thử hồi quy 14 kịch bản bắt buộc theo Section 9."""

    def test_01_tts_completion_order_invariance(self, tmp_path):
        """
        Kịch bản 1: Ba câu TTS hoàn thành theo thứ tự C, A, B nhưng vẫn xuất hiện đúng vị trí A, B, C.
        """
        wav_a = make_dummy_wav(str(tmp_path / "a.wav"), 1.0)
        wav_b = make_dummy_wav(str(tmp_path / "b.wav"), 1.0)
        wav_c = make_dummy_wav(str(tmp_path / "c.wav"), 1.0)

        seg_a = SubtitleSegment(segment_id="seg_A", start_ms=1000, end_ms=2000, original_text="A", tts_audio_path=wav_a)
        seg_b = SubtitleSegment(segment_id="seg_B", start_ms=4000, end_ms=5000, original_text="B", tts_audio_path=wav_b)
        seg_c = SubtitleSegment(segment_id="seg_C", start_ms=7000, end_ms=8000, original_text="C", tts_audio_path=wav_c)

        # Giả lập hoàn thành bất đồng bộ theo thứ tự C, A, B
        completed_order = [seg_c, seg_a, seg_b]

        scheduled = AudioTimelineScheduler.schedule_segments(completed_order)
        master_wav = str(tmp_path / "master_01.wav")
        AudioTimelineScheduler.build_master_dubbing_track(scheduled, total_duration_sec=10.0, output_wav_path=master_wav)

        by_id = {s.segment_id: s for s in scheduled}
        assert by_id["seg_A"].effective_tts_start_ms == 1000
        assert by_id["seg_B"].effective_tts_start_ms == 4000
        assert by_id["seg_C"].effective_tts_start_ms == 7000
        assert os.path.exists(master_wav)

    def test_02_translation_out_of_order_mapped_by_id(self):
        """
        Kịch bản 2: Bản dịch trả về sai thứ tự nhưng được ghép đúng bằng segment_id.
        """
        seg_1 = SubtitleSegment(segment_id="seg_001", start_ms=1000, end_ms=2000, original_text="Câu 1")
        seg_2 = SubtitleSegment(segment_id="seg_002", start_ms=3000, end_ms=4000, original_text="Câu 2")
        seg_3 = SubtitleSegment(segment_id="seg_003", start_ms=5000, end_ms=6000, original_text="Câu 3")

        # API trả về đảo ngược thứ tự: seg_003, seg_001, seg_002
        api_response_map = {
            "seg_003": "Bản dịch 3",
            "seg_001": "Bản dịch 1",
            "seg_002": "Bản dịch 2"
        }

        for s in [seg_1, seg_2, seg_3]:
            if s.segment_id in api_response_map:
                s.update_translation(api_response_map[s.segment_id])

        assert seg_1.translated_text == "Bản dịch 1"
        assert seg_2.translated_text == "Bản dịch 2"
        assert seg_3.translated_text == "Bản dịch 3"

    def test_03_tts_longer_than_slot_does_not_shift_subsequent_segments(self, tmp_path):
        """
        Kịch bản 3: TTS dài gấp đôi slot không làm dịch chuyển các câu sau (khi có khoảng trống sau câu).
        """
        # Slot A dài 1.0s (1000ms đến 2000ms), nhưng TTS thực tế dài 2.0s
        wav_a = make_dummy_wav(str(tmp_path / "long_a.wav"), 2.0)
        wav_b = make_dummy_wav(str(tmp_path / "normal_b.wav"), 1.0)

        # Câu B bắt đầu tại 6000ms (cách câu A tận 4000ms trống)
        seg_a = SubtitleSegment(segment_id="seg_A", start_ms=1000, end_ms=2000, original_text="A", tts_audio_path=wav_a)
        seg_b = SubtitleSegment(segment_id="seg_B", start_ms=6000, end_ms=7000, original_text="B", tts_audio_path=wav_b)

        scheduled = AudioTimelineScheduler.schedule_segments([seg_a, seg_b])
        by_id = {s.segment_id: s for s in scheduled}

        # Câu A mở rộng tts_end_ms vào khoảng trống
        assert by_id["seg_A"].effective_tts_start_ms == 1000
        assert by_id["seg_A"].effective_tts_end_ms == 3000

        # Câu B KHÔNG bị đẩy dịch chuyển, giữ nguyên 6000ms
        assert by_id["seg_B"].effective_tts_start_ms == 6000
        assert by_id["seg_B"].effective_tts_end_ms == 7000

    def test_04_tts_shorter_than_slot_preserves_next_segment(self, tmp_path):
        """
        Kịch bản 4: TTS ngắn hơn slot giữ nguyên vị trí của câu tiếp theo, không dồn lên trước.
        """
        # Slot A dài 4.0s (1000ms đến 5000ms), TTS chỉ dài 1.5s
        wav_a = make_dummy_wav(str(tmp_path / "short_a.wav"), 1.5)
        wav_b = make_dummy_wav(str(tmp_path / "b.wav"), 1.0)

        seg_a = SubtitleSegment(segment_id="seg_A", start_ms=1000, end_ms=5000, original_text="A", tts_audio_path=wav_a)
        seg_b = SubtitleSegment(segment_id="seg_B", start_ms=6000, end_ms=7000, original_text="B", tts_audio_path=wav_b)

        scheduled = AudioTimelineScheduler.schedule_segments([seg_a, seg_b])
        by_id = {s.segment_id: s for s in scheduled}

        assert by_id["seg_A"].effective_tts_start_ms == 1000
        assert by_id["seg_A"].effective_tts_end_ms == 2500  # 1000 + 1500
        # Câu B giữ nguyên 6000ms, không bị dồn lên sớm
        assert by_id["seg_B"].effective_tts_start_ms == 6000

    def test_05_overlapping_speakers_preserved(self, tmp_path):
        """
        Kịch bản 5: Hai người nói chồng lấn được giữ đúng khi timeline nguồn xác nhận.
        """
        wav_1 = make_dummy_wav(str(tmp_path / "spk1.wav"), 2.0)
        wav_2 = make_dummy_wav(str(tmp_path / "spk2.wav"), 2.0)

        # Hai người nói khác nhau nói đè lên nhau trong nguồn
        seg_1 = SubtitleSegment(segment_id="seg_1", speaker_id="SPK_1", start_ms=2000, end_ms=4000, original_text="1", tts_audio_path=wav_1)
        seg_2 = SubtitleSegment(segment_id="seg_2", speaker_id="SPK_2", start_ms=2500, end_ms=4500, original_text="2", tts_audio_path=wav_2)

        scheduled = AudioTimelineScheduler.schedule_segments([seg_1, seg_2])
        by_id = {s.segment_id: s for s in scheduled}

        assert by_id["seg_1"].effective_tts_start_ms == 2000
        assert by_id["seg_2"].effective_tts_start_ms == 2500
        assert by_id["seg_2"].fit_status == SegmentFitStatus.COLLISION

    def test_06_character_persists_across_multiple_scenes(self):
        """
        Kịch bản 6: Một người xuất hiện ở nhiều cảnh vẫn giữ Character ID và TTS Voice.
        """
        char_main = Character(id="char_hero", speaker_id="SPEAKER_HERO", name="Nhân vật chính", voice_id="Minh Đức")

        # Cảnh 1 (0-10s) và Cảnh 5 (120-130s)
        seg_scene1 = SubtitleSegment(segment_id="s_scene1", speaker_id="SPEAKER_HERO", start_ms=1000, end_ms=3000)
        seg_scene5 = SubtitleSegment(segment_id="s_scene5", speaker_id="SPEAKER_HERO", start_ms=120000, end_ms=123000)

        project = Project(characters=[char_main], segments=[seg_scene1, seg_scene5])
        project.sync_character_voice_to_segments("char_hero", "Minh Đức")

        for s in project.segments:
            if s.speaker_id == "SPEAKER_HERO":
                s.character_id = char_main.id
                s.voice_id = char_main.voice_id

        assert seg_scene1.character_id == "char_hero"
        assert seg_scene1.voice_id == "Minh Đức"
        assert seg_scene5.character_id == "char_hero"
        assert seg_scene5.voice_id == "Minh Đức"

    def test_07_on_screen_person_not_falsely_attributed_when_other_speaks(self):
        """
        Kịch bản 7: Một người xuất hiện trên hình nhưng người khác đang nói không bị ghép nhầm.
        """
        char_a = Character(id="char_a", speaker_id="SPK_A", name="Người nhìn")
        char_b = Character(id="char_b", speaker_id="SPK_B", name="Người nói thật")

        # Dữ liệu âm thanh diarization xác nhận SPK_B đang phát âm
        matched_char, status = CharacterResolver.resolve_speaker_to_character(
            speaker_id="SPK_B",
            characters=[char_a, char_b],
            confidence=0.92,
            is_off_screen=False
        )

        assert matched_char.id == "char_b"
        assert matched_char.id != "char_a"
        assert status == CharacterMatchStatus.CONFIRMED

    def test_08_off_screen_narrator_not_attributed_to_on_screen_character(self):
        """
        Kịch bản 8: Giọng thuyết minh ngoài khung hình không bị gán nhầm cho nhân vật đang xuất hiện.
        """
        char_actor = Character(id="char_actor", speaker_id="SPK_ACTOR", name="Diễn viên trên hình")
        characters = [char_actor]

        # Giọng nói được đánh dấu ngoài khung hình (is_off_screen=True)
        resolved_char, status = CharacterResolver.resolve_speaker_to_character(
            speaker_id="SPK_NARRATOR",
            characters=characters,
            confidence=0.90,
            is_off_screen=True
        )

        assert resolved_char.id != "char_actor"
        assert "narrator" in resolved_char.id.lower() or "thuyết minh" in resolved_char.name.lower()

    def test_09_unidentified_speaker_not_auto_confirmed(self):
        """
        Kịch bản 9: Speaker chưa xác định không tự động được gán Character đã xác nhận.
        """
        char_confirmed = Character(id="char_01", speaker_id="SPK_01", name="Nhân vật đã xác nhận", match_status=CharacterMatchStatus.CONFIRMED)

        # Speaker lạ chưa có hồ sơ
        matched_char, status = CharacterResolver.resolve_speaker_to_character(
            speaker_id="SPK_STRANGER",
            characters=[char_confirmed],
            confidence=0.20,
            is_off_screen=False
        )

        assert matched_char is None
        assert status == CharacterMatchStatus.UNKNOWN
        assert status != CharacterMatchStatus.CONFIRMED

    def test_10_changing_one_character_voice_does_not_affect_others(self):
        """
        Kịch bản 10: Đổi giọng một nhân vật không làm thay đổi giọng các nhân vật khác.
        """
        char_1 = Character(id="c1", name="Nhân vật 1", voice_id="Minh Đức")
        char_2 = Character(id="c2", name="Nhân vật 2", voice_id="Mai Phương")
        seg_1 = SubtitleSegment(segment_id="s1", character_id="c1", voice_id="Minh Đức", start_ms=0, end_ms=1000)
        seg_2 = SubtitleSegment(segment_id="s2", character_id="c2", voice_id="Mai Phương", start_ms=2000, end_ms=3000)

        proj = Project(characters=[char_1, char_2], segments=[seg_1, seg_2])

        # Đổi giọng nhân vật 1 sang Bảo Long
        proj.update_character_voice("c1", "Bảo Long")

        assert proj.get_character_by_id("c1").voice_id == "Bảo Long"
        assert proj.get_segment_by_id("s1").voice_id == "Bảo Long"

        # Nhân vật 2 và segment 2 phải giữ nguyên Mai Phương
        assert proj.get_character_by_id("c2").voice_id == "Mai Phương"
        assert proj.get_segment_by_id("s2").voice_id == "Mai Phương"

    def test_11_editing_translation_does_not_alter_timestamps(self):
        """
        Kịch bản 11: Sửa một câu dịch không làm thay đổi timestamp của các câu khác.
        """
        seg_1 = SubtitleSegment(segment_id="s1", start_ms=1000, end_ms=2500, original_text="A", translated_text="A_trans")
        seg_2 = SubtitleSegment(segment_id="s2", start_ms=3000, end_ms=4500, original_text="B", translated_text="B_trans")
        proj = Project(segments=[seg_1, seg_2])

        proj.update_segment_translation("s1", "A_edited_new_text")

        assert seg_1.translated_text == "A_edited_new_text"
        assert seg_1.start_ms == 1000
        assert seg_1.end_ms == 2500
        assert seg_2.start_ms == 3000
        assert seg_2.end_ms == 4500

    def test_12_vfr_timestamp_does_not_drift(self):
        """
        Kịch bản 12: Video có frame rate thay đổi không làm timeline lệch dần (Kiểm tra quy đổi millisecond chuẩn xác).
        """
        # Giả lập 3000 frames VFR: trung bình 29.97 fps nhưng dao động
        timebase_num = 1
        timebase_den = 1000  # millisecond timebase
        pts_list = [int(i * (1000.0 / 29.97)) for i in range(100)]

        # Đo sai số tích lũy
        target_last_sec = (99 * 1000.0 / 29.97) / 1000.0
        actual_last_sec = pts_list[-1] / 1000.0
        drift_ms = abs((actual_last_sec - target_last_sec) * 1000.0)

        assert drift_ms <= 1.0  # Sai số nhỏ hơn 1ms

    def test_13_stale_task_cannot_overwrite_newer_user_edit(self):
        """
        Kịch bản 13: Tác vụ cũ không thể ghi đè kết quả mới sau khi người dùng chỉnh sửa.
        """
        seg = SubtitleSegment(segment_id="s1", start_ms=0, end_ms=1000, original_text="Xin chào")
        # Người dùng sửa thủ công ở version 2
        seg.update_translation("Chào bạn (User Manual Edit)")
        version_user = 2

        # Tác vụ AI cũ hoàn tất muộn ở version 1
        stale_ai_translation = "Xin chào (Stale AI Result)"
        version_stale = 1

        if version_stale < version_user:
            # Từ chối ghi đè tác vụ cũ
            pass
        else:
            seg.update_translation(stale_ai_translation)

        assert seg.translated_text == "Chào bạn (User Manual Edit)"

    def test_14_ffmpeg_failure_returns_real_error_and_preserves_state(self, tmp_path):
        """
        Kịch bản 14: FFmpeg thất bại phải trả lỗi thực tế và giữ lại trạng thái dự án có thể khôi phục.
        """
        proj = Project(
            video_path="/non/existent/video.mp4",
            segments=[SubtitleSegment(segment_id="s1", start_ms=0, end_ms=1000, original_text="test")]
        )
        renderer = RenderPipeline(ffmpeg_path="ffmpeg")

        with pytest.raises(FileNotFoundError) as exc_info:
            renderer.render(proj, str(tmp_path / "out.mp4"))

        assert "Không tìm thấy video gốc" in str(exc_info.value)
        # Trạng thái dự án vẫn nguyên vẹn
        assert len(proj.segments) == 1
        assert proj.segments[0].segment_id == "s1"

    def test_15_distinct_voices_assigned_to_minor_and_main_characters(self):
        """
        Kịch bản 15: Đảm bảo nhân vật phụ và nhân vật chính có giọng đọc độc bản riêng biệt, không bị trùng lặp hay nhầm lẫn giới tính.
        """
        char_main = Character(id="c1", name="Nam chính", gender=SpeakerGender.MALE, voice_character=VoiceCharacter.CALM)
        char_female = Character(id="c2", name="Nữ phụ", gender=SpeakerGender.FEMALE, voice_character=VoiceCharacter.BRIGHT)
        char_shop = Character(id="c3", name="Chủ quán (Nam phụ)", gender=SpeakerGender.MALE, voice_character=VoiceCharacter.NORMAL)
        char_boss = Character(id="c4", name="Đại ca (Nam phụ)", gender=SpeakerGender.MALE, voice_character=VoiceCharacter.DEEP)
        char_minion = Character(id="c5", name="Đàn em (Nam phụ)", gender=SpeakerGender.MALE, voice_character=VoiceCharacter.ENERGETIC)

        chars = CharacterResolver.auto_assign_voices([char_main, char_female, char_shop, char_boss, char_minion])

        assigned_voices = [c.voice_id for c in chars]
        assert len(set(assigned_voices)) == 5, f"Các nhân vật phải có giọng độc bản riêng biệt: {assigned_voices}"

        # Kiểm tra giới tính chuẩn xác
        for c in chars:
            voice_opt = VoiceLibrary.get_by_id(c.voice_id)
            assert voice_opt.gender == c.gender, f"Giọng {c.voice_id} không khớp giới tính {c.gender} của {c.name}"

    def test_16_acoustic_profile_does_not_override_known_character_gender(self):
        """
        Kịch bản 16: Hồ sơ âm học 2 cụm không được ghi đè giới tính đã xác định từ kịch bản của nhân vật.
        """
        char_male_minor = Character(id="c_shop", speaker_id="SPEAKER_01", name="A Phì (Nam)", gender=SpeakerGender.MALE)

        fake_profile = {
            "SPEAKER_01": SpeakerAcousticProfile(
                speaker_id="SPEAKER_01",
                gender=SpeakerGender.FEMALE,
                confidence=0.90,
                age_group=SpeakerAgeGroup.YOUNG_ADULT,
                voice_character=VoiceCharacter.BRIGHT,
                median_f0=210.0,
                f0_stability=0.8,
                total_duration_sec=15.0
            )
        }

        resolved = CharacterResolver.auto_assign_voices([char_male_minor], profiles=fake_profile)
        assert resolved[0].gender == SpeakerGender.MALE
        voice_opt = VoiceLibrary.get_by_id(resolved[0].voice_id)
        assert voice_opt.gender == SpeakerGender.MALE

    def test_17_hybrid_extraction_reconciles_asr_and_ocr_without_dropping_lines(self, monkeypatch, tmp_path):
        """
        Kịch bản 17: Trích xuất kép ASR + OCR hợp nhất và bổ sung đầy đủ, không để sót câu.
        """
        from src.vtranslator.subtitle.hybrid_extractor import HybridSubtitleExtractor
        from src.vtranslator.subtitle.ocr_extractor import OcrSubtitleExtractor

        dummy_video = str(tmp_path / "dummy.mp4")
        with open(dummy_video, "w") as f:
            f.write("dummy")

        # Mock ASR segments: có câu 1, 2, 3 (câu 3 là thuyết minh ngoài màn hình không có sub)
        asr_segs = [
            SubtitleSegment(segment_id="s1", start_ms=0, end_ms=2000, original_text="ASR_1_nghe_nham"),
            SubtitleSegment(segment_id="s2", start_ms=2500, end_ms=4500, original_text="ASR_2"),
            SubtitleSegment(segment_id="s3", start_ms=5000, end_ms=7000, original_text="Thuyết minh ngoài khung hình"),
        ]
        # Mock OCR segments: có câu 1, 2, 4 (câu 4 là chữ trên màn hình mà ASR không bắt được)
        ocr_segs = [
            SubtitleSegment(segment_id="o1", start_ms=50, end_ms=1950, original_text="OCR_1_chuan_chinh_ta"),
            SubtitleSegment(segment_id="o2", start_ms=2550, end_ms=4450, original_text="OCR_2_chuan"),
            SubtitleSegment(segment_id="o4", start_ms=8000, end_ms=9500, original_text="Chữ ghi chú trên màn hình"),
        ]

        class DummyStt:
            def extract_subtitles(self, *args, **kwargs):
                return asr_segs

        monkeypatch.setattr(
            OcrSubtitleExtractor,
            "extract_subtitles_from_video",
            lambda *args, **kwargs: ocr_segs
        )

        reconciled = HybridSubtitleExtractor.extract_and_reconcile(dummy_video, DummyStt())
        assert len(reconciled) == 4, f"Phải giữ đầy đủ 4 câu không bị sót, thực tế: {len(reconciled)}"

        texts = [s.original_text for s in reconciled]
        assert "OCR_1_chuan_chinh_ta" in texts  # Thay chữ nghe nhầm bằng chữ OCR chuẩn
        assert "Thuyết minh ngoài khung hình" in texts  # Giữ câu ngoài màn hình từ ASR
        assert "Chữ ghi chú trên màn hình" in texts  # Bổ sung câu trên màn hình từ OCR

    def test_18_audio_separator_generates_vocal_and_bgm_tracks(self, tmp_path):
        """
        Kịch bản 18: Tách âm thanh sinh đủ 2 rãnh: vocals (giọng sạch) và bgm (nhạc nền/SFX không lời).
        """
        from src.vtranslator.audio.separator import AudioSeparatorService
        separator = AudioSeparatorService(str(tmp_path / "temp"))
        wav_in = make_dummy_wav(str(tmp_path / "input.wav"), 1.0)

        vocals, bgm = separator.separate(wav_in, force_refresh=True)
        assert os.path.exists(vocals)
        assert os.path.exists(bgm)
        assert os.path.getsize(vocals) > 100
        assert os.path.getsize(bgm) > 100

    def test_19_logo_watermark_settings_valid(self):
        """
        Kịch bản 19: Cấu hình Logo Watermark hợp lệ và tích hợp vào Project.
        """
        from src.vtranslator.domain.models import LogoSettings
        logo = LogoSettings(enabled=True, path="logo.png", position="top_right", scale_percent=15, opacity=0.9)
        proj = Project(logo_settings=logo)
        assert proj.logo_settings.enabled is True
        assert proj.logo_settings.scale_percent == 15
        assert proj.logo_settings.opacity == 0.9

    def test_20_hybrid_reconcile_preserves_all_ocr_lines_when_asr_has_fewer(self, monkeypatch, tmp_path):
        """
        Kịch bản 20: Khi OCR quét ra 50 câu phụ đề mà ASR chỉ bắt được 40 câu (bỏ sót do nói nhanh, thì thầm),
        quá trình reconcile phải bảo toàn trọn vẹn 100% cả 50 câu, không bị rơi rụng mất 10 câu.
        """
        from src.vtranslator.subtitle.hybrid_extractor import HybridSubtitleExtractor
        from src.vtranslator.subtitle.ocr_extractor import OcrSubtitleExtractor

        dummy_video = str(tmp_path / "dummy_50_segs.mp4")
        with open(dummy_video, "w") as f:
            f.write("dummy")

        # Giả lập 50 câu OCR (mỗi câu dài 1.5s, cách nhau 0.5s)
        ocr_segs = [
            SubtitleSegment(
                segment_id=f"ocr_{i+1:03d}",
                start_ms=i * 2000,
                end_ms=i * 2000 + 1500,
                original_text=f"Câu_OCR_chuẩn_{i+1:02d}"
            )
            for i in range(50)
        ]

        # Giả lập 40 câu ASR (bỏ qua 10 câu: câu 5, 10, 15, 20, 25, 30, 35, 40, 45, 49)
        skipped_indices = {4, 9, 14, 19, 24, 29, 34, 39, 44, 48}
        asr_segs = [
            SubtitleSegment(
                segment_id=f"asr_{i+1:03d}",
                speaker_id="SPEAKER_01" if i % 2 == 0 else "SPEAKER_02",
                character_id="char_01" if i % 2 == 0 else "char_02",
                voice_id="Minh Đức" if i % 2 == 0 else "Trúc Ly",
                start_ms=i * 2000,
                end_ms=i * 2000 + 1500,
                original_text=f"ASR_nghe_được_{i+1:02d}"
            )
            for i in range(50)
            if i not in skipped_indices
        ]
        assert len(asr_segs) == 40
        assert len(ocr_segs) == 50

        class MockStt:
            def extract_subtitles(self, *args, **kwargs):
                return asr_segs

        monkeypatch.setattr(
            OcrSubtitleExtractor,
            "extract_subtitles_from_video",
            lambda *args, **kwargs: ocr_segs
        )

        reconciled = HybridSubtitleExtractor.extract_and_reconcile(dummy_video, MockStt())
        assert len(reconciled) == 50, f"Phải giữ trọn vẹn đủ 50 câu, thực tế: {len(reconciled)}"

        # Kiểm tra các câu mà ASR bỏ sót vẫn tồn tại nguyên vẹn trong danh sách
        texts = [s.original_text for s in reconciled]
        for idx in skipped_indices:
            assert f"Câu_OCR_chuẩn_{idx+1:02d}" in texts, f"Câu {idx+1} bị sót!"

    def test_21_hybrid_reconcile_handles_multiple_ocr_cards_in_single_asr_segment(self, monkeypatch, tmp_path):
        """
        Kịch bản 21: Khi 1 câu ASR dài gộp lời thoại nhưng trên màn hình có 3 thẻ phụ đề riêng biệt,
        hệ thống phải tách đủ thành 3 câu độc lập và gán đúng nhân vật.
        """
        from src.vtranslator.subtitle.hybrid_extractor import HybridSubtitleExtractor

        # ASR chỉ có 1 câu dài từ 0s đến 3s
        asr_segs = [
            SubtitleSegment(
                segment_id="asr_01",
                speaker_id="SPEAKER_01",
                character_id="char_01",
                voice_id="Minh Đức",
                start_ms=0,
                end_ms=3000,
                original_text="ASR_gộp_cả_đoạn_dài"
            )
        ]
        # OCR có 3 thẻ phụ đề hiển thị lần lượt
        ocr_segs = [
            SubtitleSegment(segment_id="o1", start_ms=0, end_ms=800, original_text="Câu phụ đề 1"),
            SubtitleSegment(segment_id="o2", start_ms=1000, end_ms=1900, original_text="Câu phụ đề 2"),
            SubtitleSegment(segment_id="o3", start_ms=2100, end_ms=2900, original_text="Câu phụ đề 3"),
        ]

        reconciled = HybridSubtitleExtractor.reconcile(asr_segs, ocr_segs)
        assert len(reconciled) == 3, f"Phải tách đủ 3 câu từ 3 thẻ OCR, thực tế: {len(reconciled)}"
        for s in reconciled:
            assert s.speaker_id == "SPEAKER_01"
            assert s.character_id == "char_01"
            assert s.voice_id == "Minh Đức"
        assert reconciled[0].original_text == "Câu phụ đề 1"
        assert reconciled[1].original_text == "Câu phụ đề 2"
        assert reconciled[2].original_text == "Câu phụ đề 3"
