"""OCR Subtitle Extractor - Ultra-Fast In-Process RapidOCR with Zero-Clip Padding."""

import os
import re
from typing import List, Optional, Tuple, Callable, Dict, Any
from difflib import SequenceMatcher
import cv2
import numpy as np
from ..domain.segments import SubtitleSegment
from ..domain.enums import TtsStatus

# ─── High-Performance In-Process RapidOCR Singleton ───────────────────────────
_rapid_ocr_instance = None


def _get_rapid_ocr():
    """Khởi tạo một instance RapidOCR duy nhất trong tiến trình hiện tại.
    ONNX Runtime C++ tự động đa luồng (intra_op_num_threads=4) không bị vướng Python GIL,
    không tốn chi phí khởi tạo process hay serialize IPC.
    """
    global _rapid_ocr_instance
    if _rapid_ocr_instance is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr_instance = RapidOCR(
                det_limit_side_len=640,
                det_limit_type="max",
                use_cls=False,            # Tắt xoay chữ 180° -> Tăng 20% tốc độ, không xoay nhầm câu ngắn
                det_db_thresh=0.2,        # Giữ nét chữ mờ hoặc font mảnh
                det_db_box_thresh=0.3,    # Giữ các câu ngắn 1-2 từ ('Hả?', 'Ừ!', '好')
                det_db_unclip_ratio=2.2,  # Nở rộng khung phát hiện để không nuốt nét đầu/cuối
                det_db_score_mode="fast", # Chế độ fast siêu tốc (nhanh hơn 5x so với slow)
                intra_op_num_threads=4,   # 4 luồng C++ xử lý tensor trên Intel CPU
                rec_batch_num=6,
            )
        except Exception as e:
            print(f"RapidOCR không khả dụng: {e}")
    return _rapid_ocr_instance


def _run_crop_ocr(rapid_ocr, crop: np.ndarray) -> str:
    """Xử lý OCR một khung ảnh: đệm viền đen chống rớt chữ và sắp xếp từ trái qua phải."""
    if rapid_ocr is None or crop is None or crop.size == 0:
        return ""
    try:
        # BÍ QUYẾT CHỐNG SÓT CHỮ: Đệm thêm viền đen 16px xung quanh ảnh đã cắt
        # Giúp chữ không chạm sát mép ảnh, DBNet nhận diện trọn vẹn cả câu
        padded_crop = cv2.copyMakeBorder(
            crop, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        res, _ = rapid_ocr(padded_crop, use_det=True, use_cls=False, use_rec=True)
        if res:
            # Sắp xếp các cụm chữ từ trái sang phải theo tọa độ X
            res.sort(key=lambda x: x[0][0][0])
            texts = []
            for item in res:
                t = str(item[1]).strip() if item[1] else ""
                conf = float(item[2]) if len(item) > 2 else 1.0
                if t and conf >= 0.50:
                    t_clean = re.sub(r"[\r\n]+", " ", t).strip()
                    t_clean = re.sub(r"\s+", " ", t_clean)
                    if t_clean:
                        texts.append(t_clean)
            return " ".join(texts).strip()
    except Exception:
        pass
    return ""


class OcrSubtitleExtractor:
    @staticmethod
    def extract_crop_at_timestamp(
        cap: cv2.VideoCapture,
        timestamp_sec: float,
        roi: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """Trích xuất dải ảnh chứa phụ đề tại mốc thời gian theo đúng tọa độ khung người dùng chọn."""
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ret, frame = cap.read()
        if not ret:
            return None
        h, w, _ = frame.shape

        if roi and roi.get("width", 0) > 20 and roi.get("height", 0) > 10:
            rx = float(roi.get("x", 0))
            ry = float(roi.get("y", 0))
            rw = float(roi.get("width", 0))
            rh = float(roi.get("height", 0))
            scene_w = float(roi.get("scene_width", 0))
            scene_h = float(roi.get("scene_height", 0))
            if scene_w > 0 and scene_h > 0 and (scene_w != w or scene_h != h):
                scale_x = w / scene_w
                scale_y = h / scene_h
                rx *= scale_x
                ry *= scale_y
                rw *= scale_x
                rh *= scale_y
            x1 = max(0, min(w - 10, int(rx)))
            y1 = max(0, min(h - 10, int(ry)))
            x2 = max(x1 + 10, min(w, int(rx + rw)))
            y2 = max(y1 + 10, min(h, int(ry + rh)))
            crop = frame[y1:y2, x1:x2]
        else:
            # Mặc định dải phụ đề đáy màn hình: 72% đến 100% chiều cao (không bị cắt cụt đáy chữ)
            crop = frame[int(h * 0.72) : h, int(w * 0.05) : int(w * 0.95)]

        return crop

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Làm sạch ký tự rác và khoảng trắng dư thừa trong kết quả OCR."""
        t = re.sub(r"[\r\n]+", " ", text).strip()
        t = re.sub(r"\s+", " ", t)
        if len(t) == 0:
            return ""
        # Giữ lại chữ cái đơn lẻ nếu là chữ hoặc ký tự CJK (như 'Ừ', 'Hả', '好')
        if len(t) == 1 and not (t.isalnum() or '\u4e00' <= t <= '\u9fff'):
            return ""
        return t

    @classmethod
    def extract_hybrid_subtitles(
        cls,
        video_path: str,
        roi: Optional[Dict[str, Any]] = None,
        stt_service=None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[SubtitleSegment]:
        """
        Quét phụ đề trực tiếp từ video qua OCR có chọn vùng (bỏ qua ASR nếu không cần thiết).
        """
        return cls.extract_subtitles_from_video(video_path, roi=roi, progress_callback=progress_callback)

    @classmethod
    def extract_subtitles_from_video(
        cls,
        video_path: str,
        roi: Optional[Dict[str, Any]] = None,
        sample_interval: float = 0.35,
        sample_fps: Optional[float] = None,
        min_duration_sec: float = 0.5,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[SubtitleSegment]:
        """
        Quét trực tiếp video từ khung ROI bằng RapidOCR siêu tốc:
        - Sử dụng ONNX Runtime C++ multi-thread trong tiến trình (không overhead spawn process).
        - Đệm viền đen 16px (cv2.copyMakeBorder) chống đứt nét chữ ở rìa ảnh crop.
        - Tắt use_cls=False + det_db_score_mode='fast' cho tốc độ tối đa.
        - Lọc bỏ khung hình trùng lặp cực nhanh (cv2.absdiff.max < 18) chỉ gọi OCR khi chữ thay đổi.
        - Tự động chuẩn hóa tọa độ ROI theo độ phân giải gốc của video.
        - Phục hồi và tự seek tiếp khi gặp frame lỗi trên video dài > 2 tiếng, không bị ngắt sớm.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video để quét OCR: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if video_fps <= 1.0 or video_fps > 120.0:
            video_fps = 25.0

        cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sử dụng ffprobe để lấy thời lượng chính xác tuyệt đối, tránh lỗi CAP_PROP_FRAME_COUNT trên video lớn
        ffprobe_duration_sec = 0.0
        try:
            from ..video.metadata import VideoMetadataService
            meta = VideoMetadataService().probe(video_path)
            if meta.duration_ms > 0:
                ffprobe_duration_sec = meta.duration_ms / 1000.0
            if meta.fps > 1.0:
                video_fps = meta.fps
        except Exception:
            pass

        if ffprobe_duration_sec > 0:
            total_duration_sec = ffprobe_duration_sec
            total_frames = max(cap_frames, int(total_duration_sec * video_fps))
        elif cap_frames > 0:
            total_frames = cap_frames
            total_duration_sec = total_frames / float(video_fps)
        else:
            total_frames = 200000
            total_duration_sec = total_frames / float(video_fps)

        if sample_fps is not None and sample_fps > 0:
            sample_interval = 1.0 / sample_fps

        step_frames = max(1, int(video_fps * sample_interval))

        rapid_ocr = _get_rapid_ocr()

        cur_frame = 0
        prev_crop_gray: Optional[np.ndarray] = None
        active_text = ""
        active_start_sec = 0.0
        active_end_sec = 0.0
        miss_count = 0
        raw_segments: List[Tuple[float, float, str]] = []

        roi_bounds: Optional[Tuple[int, int, int, int]] = None
        consecutive_read_failures = 0
        max_allowed_consecutive_failures = 20

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_read_failures += 1
                    # Khi gặp frame lỗi hoặc packet drop giữa video dài (đặc biệt phim > 2 tiếng)
                    # không ngắt toàn bộ tiến trình quét, mà seek nhảy qua điểm lỗi để tiếp tục
                    if consecutive_read_failures <= max_allowed_consecutive_failures:
                        cur_frame += step_frames
                        if total_frames > 0 and cur_frame >= total_frames + step_frames * 3:
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                        continue
                    else:
                        # Đã thử seek nhiều lần liên tiếp không được -> thực sự chạm đáy file video
                        break

                consecutive_read_failures = 0
                t_sec = cur_frame / video_fps
                h, w = frame.shape[:2]

                if roi_bounds is None:
                    if roi and roi.get("width", 0) > 20 and roi.get("height", 0) > 10:
                        rx = float(roi.get("x", 0))
                        ry = float(roi.get("y", 0))
                        rw = float(roi.get("width", 0))
                        rh = float(roi.get("height", 0))
                        scene_w = float(roi.get("scene_width", 0))
                        scene_h = float(roi.get("scene_height", 0))
                        if scene_w > 0 and scene_h > 0 and (scene_w != w or scene_h != h):
                            scale_x = w / scene_w
                            scale_y = h / scene_h
                            rx *= scale_x
                            ry *= scale_y
                            rw *= scale_x
                            rh *= scale_y
                        x1 = max(0, min(w - 10, int(rx)))
                        y1 = max(0, min(h - 10, int(ry)))
                        x2 = max(x1 + 10, min(w, int(rx + rw)))
                        y2 = max(y1 + 10, min(h, int(ry + rh)))
                        roi_bounds = (x1, y1, x2, y2)
                    else:
                        roi_bounds = (int(w * 0.05), int(h * 0.72), int(w * 0.95), h)

                x1, y1, x2, y2 = roi_bounds
                crop = frame[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    for _ in range(step_frames - 1):
                        if not cap.grab():
                            break
                    cur_frame += step_frames
                    if total_frames > 0 and cur_frame >= total_frames:
                        break
                    continue

                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                max_pixel = int(crop_gray.max())
                has_bright = max_pixel >= 95

                cur_text = ""
                if has_bright:
                    if prev_crop_gray is not None and prev_crop_gray.shape == crop_gray.shape:
                        # Max diff < 18: Vùng phụ đề gần như không thay đổi -> giữ nguyên chữ cũ, KHÔNG gọi OCR
                        diff_max = int(cv2.absdiff(prev_crop_gray, crop_gray).max())
                        if diff_max < 18 and active_text:
                            cur_text = active_text
                        else:
                            cur_text = _run_crop_ocr(rapid_ocr, crop)
                            prev_crop_gray = crop_gray
                    else:
                        cur_text = _run_crop_ocr(rapid_ocr, crop)
                        prev_crop_gray = crop_gray
                else:
                    prev_crop_gray = None

                # ── State machine ghép nối timeline ──────────────────────────
                if cur_text:
                    sim = SequenceMatcher(None, active_text, cur_text).ratio() if active_text else 0.0
                    if active_text and (sim >= 0.50 or active_text in cur_text or cur_text in active_text):
                        active_end_sec = t_sec + sample_interval
                        if len(cur_text) > len(active_text):
                            active_text = cur_text
                        miss_count = 0
                    else:
                        if active_text and (active_end_sec - active_start_sec) >= min_duration_sec:
                            raw_segments.append((active_start_sec, active_end_sec, active_text))
                        active_text = cur_text
                        active_start_sec = t_sec
                        active_end_sec = t_sec + sample_interval
                        miss_count = 0
                else:
                    if active_text:
                        miss_count += 1
                        if miss_count >= 2:
                            if (active_end_sec - active_start_sec) >= min_duration_sec:
                                raw_segments.append((active_start_sec, active_end_sec, active_text))
                            active_text = ""
                            miss_count = 0

                # Dùng cap.grab() để skip các frame trung gian nhanh hơn gấp nhiều lần cap.read()
                for _ in range(step_frames - 1):
                    if not cap.grab():
                        break
                cur_frame += step_frames

                # Nếu vượt frame ước lượng nhưng video vẫn còn frame thì tiếp tục quét
                if total_frames > 0 and cur_frame >= total_frames:
                    if not cap.grab():
                        break

                if progress_callback and cur_frame % (step_frames * 10) == 0:
                    p = min(99, int((cur_frame / max(1, total_frames)) * 100))
                    progress_callback(p, f"Đang quét OCR: {int(t_sec)}s / {int(total_duration_sec)}s ({len(raw_segments)} câu)...")

            if active_text and (active_end_sec - active_start_sec) >= min_duration_sec:
                raw_segments.append((active_start_sec, active_end_sec, active_text))

        finally:
            cap.release()

        # ── Hợp nhất các câu sát nhau < 0.4s có nội dung tương đồng ──────────
        merged_raw: List[Tuple[float, float, str]] = []
        for st, en, txt in raw_segments:
            if merged_raw and (st - merged_raw[-1][1]) < 0.4:
                prev_st, prev_en, prev_txt = merged_raw[-1]
                sim = SequenceMatcher(None, prev_txt, txt).ratio()
                if sim > 0.65 or prev_txt in txt or txt in prev_txt:
                    best_txt = txt if len(txt) > len(prev_txt) else prev_txt
                    merged_raw[-1] = (prev_st, en, best_txt)
                    continue
            merged_raw.append((st, en, txt))

        # ── Xuất SubtitleSegment chuẩn ─────────────────────────────────────────
        result_segments: List[SubtitleSegment] = []
        for idx, (st, en, txt) in enumerate(merged_raw):
            if en - st < 0.8:
                en = st + 0.8
            result_segments.append(
                SubtitleSegment(
                    segment_id=f"seg_{idx+1:03d}",
                    speaker_id="SPEAKER_UNKNOWN",
                    character_id="char_unknown",
                    start_ms=int(st * 1000),
                    end_ms=int(en * 1000),
                    original_text=txt,
                    translated_text="",
                    speaker_source="unassigned",
                    voice_id="",
                    tts_status=TtsStatus.PENDING
                )
            )

        if progress_callback:
            progress_callback(100, f"✅ Quét OCR xong: {len(result_segments)} câu phụ đề.")

        return result_segments
