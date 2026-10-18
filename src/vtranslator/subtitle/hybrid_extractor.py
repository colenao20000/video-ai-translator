"""Combine ASR speech recognition and targeted OCR, then reconcile their cues."""

import os
import re
from typing import List, Optional, Dict, Callable
import cv2
from ..domain.segments import SubtitleSegment
from .ocr_extractor import OcrSubtitleExtractor


class HybridSubtitleExtractor:
    @staticmethod
    def _clean_text(text: str) -> str:
        t = re.sub(r"[\r\n]+", " ", text).strip()
        t = re.sub(r"\s+", " ", t)
        return t

    @classmethod
    def reconcile(
        cls,
        asr_segments: List[SubtitleSegment],
        ocr_segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """Hợp nhất và đối chiếu chéo danh sách câu từ ASR và OCR."""
        if not asr_segments:
            return ocr_segments
        if not ocr_segments:
            return asr_segments

        final_segments: List[SubtitleSegment] = []
        matched_asr_indices = set()

        for ocr_idx, ocr_seg in enumerate(ocr_segments):
            ocr_text = cls._clean_text(ocr_seg.original_text)
            if len(ocr_text) < 1:
                continue

            best_asr = None
            best_asr_idx = -1
            best_overlap = 0

            for asr_idx, asr_seg in enumerate(asr_segments):
                ov_st = max(ocr_seg.start_ms, asr_seg.start_ms)
                ov_en = min(ocr_seg.end_ms, asr_seg.end_ms)
                ov = max(0, ov_en - ov_st)
                if ov > best_overlap:
                    best_overlap = ov
                    best_asr = asr_seg
                    best_asr_idx = asr_idx

            if best_asr and best_overlap >= 150:
                matched_asr_indices.add(best_asr_idx)
                spk = best_asr.speaker_id
                char = best_asr.character_id
                voice = best_asr.voice_id
                src_lang = best_asr.source_language
                tgt_lang = best_asr.target_language
                if (best_asr.end_ms - best_asr.start_ms) > (ocr_seg.end_ms - ocr_seg.start_ms) * 1.5:
                    seg_st = ocr_seg.start_ms
                    seg_en = ocr_seg.end_ms
                else:
                    seg_st = best_asr.start_ms
                    seg_en = best_asr.end_ms
            else:
                # OCR alone cannot establish who is speaking. Keep identity
                # unresolved until audio diarization or an editor confirms it.
                spk = "SPEAKER_UNKNOWN"
                char = "char_unknown"
                voice = ""
                src_lang = asr_segments[0].source_language if asr_segments else "zh"
                tgt_lang = asr_segments[0].target_language if asr_segments else "vi"
                seg_st = ocr_seg.start_ms
                seg_en = ocr_seg.end_ms

            final_segments.append(
                SubtitleSegment(
                    segment_id=f"ocr_seg_{ocr_idx+1:03d}",
                    speaker_id=spk,
                    character_id=char,
                    voice_id=voice,
                    start_ms=seg_st,
                    end_ms=seg_en,
                    original_text=ocr_text,
                    source_language=src_lang,
                    target_language=tgt_lang,
                    speaker_source=best_asr.speaker_source if best_asr else "unassigned",
                )
            )

        # Giữ lại các câu thoại ASR không có phụ đề cứng trên màn hình (thuyết minh, ngoài hình)
        for asr_idx, asr_seg in enumerate(asr_segments):
            asr_txt = cls._clean_text(asr_seg.original_text)
            if len(asr_txt) < 2:
                continue
            # Bỏ qua tạp âm / tiếng lầm bầm vô nghĩa
            if asr_txt in ["哦哦嗯嗯嗯", "呜呜", "喂", "嗯", "啊", "哦"]:
                continue

            has_overlap = any(
                max(0, min(asr_seg.end_ms, f.end_ms) - max(asr_seg.start_ms, f.start_ms)) >= 250
                for f in final_segments
            )
            if not has_overlap:
                final_segments.append(asr_seg)

        final_segments.sort(key=lambda s: s.start_ms)

        for idx, s in enumerate(final_segments):
            s.segment_id = f"seg_{idx+1:03d}"

        return final_segments

    @classmethod
    def extract_and_reconcile(
        cls,
        video_path: str,
        stt_service,
        vocals_wav_path: Optional[str] = None,
        roi: Optional[Dict[str, int]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[SubtitleSegment]:
        """
        Trích xuất phụ đề kép kết hợp ASR và OCR:
        - Bước 1: Nhận diện lời thoại ASR từ giọng sạch.
        - Bước 2: Quét OCR vùng phụ đề tại các mốc hình phù hợp.
        - Bước 3: Đối chiếu hai nguồn; hãy rà soát câu có độ tin cậy thấp.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        # 1. Bước 1: Trích xuất lời thoại bằng ASR (từ giọng sạch)
        if progress_callback:
            progress_callback(5, "Bước 1A: Đang nhận diện lời thoại từ giọng nói sạch (ASR)...")

        audio_target = vocals_wav_path if (vocals_wav_path and os.path.exists(vocals_wav_path)) else video_path
        asr_segments: List[SubtitleSegment] = []
        try:
            asr_segments = stt_service.extract_subtitles(
                audio_target,
                force_refresh=False,
                progress_callback=lambda p, msg: progress_callback(int(5 + p * 0.15), f"ASR: {msg}") if progress_callback else None
            )
            print(f"✅ ASR nhận diện được {len(asr_segments)} phân đoạn lời thoại.")
        except Exception as e:
            print(f"⚠️ ASR gặp lỗi ({e}). Tiếp tục với OCR...")

        # 2. Bước 2: Quét phụ đề OCR từ video
        if progress_callback:
            progress_callback(20, "Bước 1B: Đang quét bóc tách phụ đề cứng trên màn hình (OCR)...")

        ocr_segments: List[SubtitleSegment] = []
        try:
            ocr_segments = OcrSubtitleExtractor.extract_subtitles_from_video(
                video_path=video_path,
                roi=roi,
                sample_interval=0.40,
                progress_callback=lambda p, msg: progress_callback(int(20 + p * 0.15), f"OCR: {msg}") if progress_callback else None
            )
            print(f"✅ OCR nhận diện được {len(ocr_segments)} câu phụ đề.")
        except Exception as e:
            print(f"⚠️ OCR quét gặp lỗi ({e}).")

        reconciled = cls.reconcile(asr_segments, ocr_segments)
        if progress_callback:
            progress_callback(35, f"Đã đối chiếu & hợp nhất thành công {len(reconciled)} câu phụ đề!")
        return reconciled
