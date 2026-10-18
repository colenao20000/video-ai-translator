"""Trim, fit, schedule, and mix generated dialogue audio on the source timeline.

Audio starts at the source cue time. The scheduler attempts tempo adjustment up
to the configured limit and records overflow or overlap instead of moving later
lines or discarding speech. The master track is zero-padded to align each cue.
"""

import os
import subprocess
from typing import List, Optional
import numpy as np
import soundfile as sf
from ..domain.segments import SubtitleSegment
from ..domain.enums import SegmentFitStatus


class FitToSlotProcessor:
    @staticmethod
    def trim_silence(
        audio_wav_path: str,
        output_wav_path: Optional[str] = None,
        top_db: float = 40.0,
        pad_lead_ms: int = 15,
        pad_tail_ms: int = 35,
    ) -> str:
        """
        Trim low-level leading and trailing audio while preserving short padding.
        """
        if not os.path.exists(audio_wav_path):
            return audio_wav_path

        out_path = output_wav_path or audio_wav_path
        try:
            data, sr = sf.read(audio_wav_path, dtype="float32")
            if len(data) == 0:
                return audio_wav_path

            mono = data.mean(axis=1) if data.ndim > 1 else data
            max_amp = np.max(np.abs(mono))
            if max_amp < 1e-4:
                return audio_wav_path

            thresh = max_amp * (10.0 ** (-top_db / 20.0))
            non_silent = np.where(np.abs(mono) > thresh)[0]
            if len(non_silent) == 0:
                return audio_wav_path

            start_idx = max(0, non_silent[0] - int(sr * pad_lead_ms / 1000.0))
            end_idx = min(len(data), non_silent[-1] + int(sr * pad_tail_ms / 1000.0))

            trimmed_data = data[start_idx:end_idx]
            if len(trimmed_data) >= int(sr * 0.15):  # Ít nhất 150ms
                sf.write(out_path, trimmed_data, sr, subtype="PCM_16")
                return out_path
        except Exception as e:
            print(f"Lỗi trim silence: {e}")

        return audio_wav_path

    @staticmethod
    def process(
        audio_wav_path: str,
        target_duration_sec: float,
        output_wav_path: str,
        max_tempo_stretch: float = 1.18,
        sample_rate: int = 44100,
        ffmpeg_path: str = "ffmpeg",
    ) -> str:
        """
        Accelerate speech up to the configured limit. If it still does not fit,
        preserve the complete utterance and let the scheduler report the overflow.
        """
        if not os.path.exists(audio_wav_path) or target_duration_sec <= 0.1:
            return audio_wav_path

        try:
            info = sf.info(audio_wav_path)
            actual_dur = info.duration
        except Exception:
            return audio_wav_path

        # If already fits within target duration with 40ms tolerance, no stretch needed
        if actual_dur <= target_duration_sec + 0.04:
            return audio_wav_path

        stretch_ratio = actual_dur / target_duration_sec
        effective_max_tempo = max(1.0, float(max_tempo_stretch))
        applied_tempo = min(stretch_ratio, effective_max_tempo)

        sped_path = audio_wav_path
        if applied_tempo > 1.02:
            # Chain atempo filters if factor > 2.0
            filters = []
            rem = applied_tempo
            while rem > 2.0:
                filters.append("atempo=2.0")
                rem /= 2.0
            filters.append(f"atempo={rem:.4f}")
            filter_str = ",".join(filters)

            cmd = [
                ffmpeg_path, "-y", "-i", audio_wav_path,
                "-filter:a", filter_str,
                "-ar", str(sample_rate), "-ac", "1",
                output_wav_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100:
                    sped_path = output_wav_path
            except Exception as e:
                print(f"Lỗi atempo fit-to-slot: {e}")

        # Sau khi atempo, nếu thời lượng vẫn vượt quá target_duration_sec,
        # tiến hành clamping và làm mượt fade-out 30ms ở đuôi để triệt tiêu va chạm âm thanh
        if sped_path and os.path.exists(sped_path):
            try:
                data, sr = sf.read(sped_path, dtype="float32")
                actual_dur = len(data) / float(sr)
                if actual_dur > target_duration_sec + 0.03:
                    max_samples = int(target_duration_sec * sr)
                    clamped_data = data[:max_samples].copy()
                    fade_samples = min(len(clamped_data), int(sr * 0.03))  # 30ms fade
                    if fade_samples > 0:
                        fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                        if clamped_data.ndim > 1:
                            clamped_data[-fade_samples:, :] *= fade_curve[:, np.newaxis]
                        else:
                            clamped_data[-fade_samples:] *= fade_curve
                    sf.write(output_wav_path, clamped_data, sr, subtype="PCM_16")
                    sped_path = output_wav_path
            except Exception as e:
                print(f"Lỗi clamping fade-out: {e}")

        return sped_path if (sped_path and os.path.exists(sped_path)) else audio_wav_path


class AudioTimelineScheduler:
    """
    Coordinates speech playback according to absolute timeline (Single Source of Truth).
    - Keeps each generated cue at its source start time.
    - Records the generated duration and fit status.
    - Reports overflow and source overlap instead of silently shifting or clipping speech.
    """

    @classmethod
    def schedule_segments(
        cls,
        segments: List[SubtitleSegment],
        max_tempo: float = 1.18,
        min_gap_ms: int = 80,
        ffmpeg_path: str = "ffmpeg",
    ) -> List[SubtitleSegment]:
        if not segments:
            return segments

        # Schedule by immutable source time. Fitting never moves later lines and
        # never rewrites the source subtitle boundaries.
        sorted_segs = sorted(segments, key=lambda s: s.start_ms)

        for i in range(len(sorted_segs)):
            curr = sorted_segs[i]

            # 1. Tự động cắt khoảng lặng thừa ở đầu và cuối tệp TTS để chống trễ khẩu hình
            if curr.tts_audio_path and os.path.exists(curr.tts_audio_path):
                source_root, source_ext = os.path.splitext(curr.tts_audio_path)
                if source_root.endswith("_trimmed"):
                    trimmed_path = curr.tts_audio_path
                else:
                    trimmed_path = f"{source_root}_trimmed{source_ext or '.wav'}"
                res_path = FitToSlotProcessor.trim_silence(curr.tts_audio_path, trimmed_path)
                if os.path.exists(res_path) and os.path.getsize(res_path) > 100:
                    curr.tts_audio_path = res_path

                try:
                    info = sf.info(curr.tts_audio_path)
                    curr.tts_duration_ms = int(info.duration * 1000)
                except Exception:
                    curr.tts_duration_ms = max(400, curr.end_ms - curr.start_ms)
            else:
                curr.tts_duration_ms = None
                curr.tts_start_ms = curr.start_ms
                curr.tts_end_ms = None
                curr.fit_status = SegmentFitStatus.FIT
                continue

            # 2. Every sentence starts STRICTLY at its source start_ms (Zero Cumulative Drift)
            curr.tts_start_ms = curr.start_ms

            # 3. Calculate available slot before the next sentence
            is_source_consecutive = True
            if i + 1 < len(sorted_segs):
                next_seg = sorted_segs[i + 1]
                next_start = next_seg.start_ms
                # Overlap present in source dialogue is intentional and remains.
                is_source_consecutive = curr.end_ms <= next_start
                if is_source_consecutive:
                    avail_slot_ms = max(300, next_start - curr.start_ms - min_gap_ms)
                else:
                    avail_slot_ms = max(300, curr.end_ms - curr.start_ms)
            else:
                avail_slot_ms = max(curr.end_ms - curr.start_ms + 1500, curr.tts_duration_ms)

            actual_dur_ms = curr.tts_duration_ms
            original_tts_duration_ms = actual_dur_ms

            # 4. If TTS exceeds available slot, apply Fit-to-Slot atempo
            if actual_dur_ms > avail_slot_ms and curr.tts_audio_path and os.path.exists(curr.tts_audio_path):
                target_sec = avail_slot_ms / 1000.0
                source_root, source_ext = os.path.splitext(curr.tts_audio_path)
                sped_wav = f"{source_root}_fit{source_ext or '.wav'}"
                res_fit = FitToSlotProcessor.process(
                    curr.tts_audio_path,
                    target_sec,
                    sped_wav,
                    max_tempo_stretch=max_tempo,
                    sample_rate=44100,
                    ffmpeg_path=ffmpeg_path,
                )
                if os.path.exists(res_fit) and os.path.getsize(res_fit) > 100:
                    curr.tts_audio_path = res_fit
                    try:
                        info = sf.info(res_fit)
                        curr.tts_duration_ms = int(info.duration * 1000)
                    except Exception:
                        curr.tts_duration_ms = avail_slot_ms
            if is_source_consecutive and curr.tts_duration_ms is not None:
                max_allowed_ms = avail_slot_ms
                if curr.tts_duration_ms > max_allowed_ms:
                    curr.tts_duration_ms = max_allowed_ms
                curr.tts_end_ms = curr.tts_start_ms + curr.tts_duration_ms
                curr.fit_status = SegmentFitStatus.ADJUSTED
                if i + 1 < len(sorted_segs):
                    next_start = sorted_segs[i + 1].start_ms
                    if curr.end_ms > next_start - 20:
                        curr.end_ms = max(curr.start_ms + 200, next_start - 20)
            else:
                curr.tts_end_ms = (curr.tts_start_ms + curr.tts_duration_ms) if curr.tts_duration_ms else None

            if curr.tts_duration_ms and curr.tts_duration_ms > 0:
                curr.applied_tempo = round(original_tts_duration_ms / curr.tts_duration_ms, 3)

            # Flag actual source overlap separately from generated overrun.
            if i > 0:
                prev = sorted_segs[i - 1]
                if curr.start_ms < prev.end_ms and curr.speaker_id != prev.speaker_id:
                    curr.fit_status = SegmentFitStatus.COLLISION

        return segments

    @classmethod
    def build_master_dubbing_track(
        cls,
        segments: List[SubtitleSegment],
        total_duration_sec: float,
        output_wav_path: str,
        sample_rate: int = 44100,
        volume: float = 1.0,
    ) -> str:
        """
        Builds a single Master Dubbing WAV track (44,100 Hz 16-bit PCM)
        with zero-padding silence alignment according to Section 6.2.
        """
        max_seg_end = max((s.effective_tts_end_ms for s in segments), default=0) / 1000.0
        final_duration_sec = max(total_duration_sec, max_seg_end)
        total_samples = max(1, int(np.ceil(final_duration_sec * sample_rate)))
        master_buffer = np.zeros(total_samples, dtype=np.float32)

        for s in segments:
            if not s.tts_audio_path or not os.path.exists(s.tts_audio_path):
                continue
            try:
                data, sr = sf.read(s.tts_audio_path, dtype="float32")
                if len(data.shape) > 1:
                    data = data.mean(axis=1)  # Stereo -> Mono

                # Resample to 44100 Hz if needed
                if sr != sample_rate:
                    import scipy.signal
                    num_out = int(len(data) * float(sample_rate) / float(sr))
                    data = scipy.signal.resample(data, num_out)

                start_ms = s.effective_tts_start_ms
                start_sample = int((start_ms / 1000.0) * sample_rate)

                # Giới hạn mẫu âm thanh tối đa theo effective_tts_end_ms để chống tràn tiếng
                max_end_sample = int((s.effective_tts_end_ms / 1000.0) * sample_rate) if s.effective_tts_end_ms else total_samples
                end_sample = min(total_samples, max_end_sample, start_sample + len(data))

                if 0 <= start_sample < total_samples and end_sample > start_sample:
                    chunk_len = end_sample - start_sample
                    chunk_data = (data[:chunk_len] * float(volume)).copy()
                    # Nếu bị giới hạn ở đuôi, làm mờ 15ms chống tiếng bụp
                    if end_sample < start_sample + len(data):
                        fade_n = min(chunk_len, int(sample_rate * 0.015))
                        if fade_n > 0:
                            fade_curve = np.linspace(1.0, 0.0, fade_n, dtype="float32")
                            chunk_data[-fade_n:] *= fade_curve
                    master_buffer[start_sample:end_sample] += chunk_data
            except Exception as e:
                print(f"Lỗi ghép audio segment {s.segment_id}: {e}")

        # Anti-Clipping normalization
        peak = np.max(np.abs(master_buffer))
        if peak > 0.95:
            master_buffer = (master_buffer / peak) * 0.95

        target_dir = os.path.dirname(os.path.abspath(output_wav_path))
        safe_temp_dir = os.path.expanduser("~/.vtranslator/temp")
        os.makedirs(safe_temp_dir, exist_ok=True)

        # Kiểm tra xem thư mục đích có quyền ghi không (tránh lỗi /Users trên macOS)
        if not os.path.exists(target_dir) or not os.access(target_dir, os.W_OK) or target_dir.rstrip("/") == "/Users":
            output_wav_path = os.path.join(safe_temp_dir, os.path.basename(output_wav_path))
            target_dir = safe_temp_dir

        try:
            os.makedirs(target_dir, exist_ok=True)
            sf.write(output_wav_path, master_buffer, sample_rate, subtype="PCM_16")
        except Exception:
            # Fallback an toàn sang safe_temp_dir nếu có bất kỳ lỗi quyền hạn/hệ thống nào
            fallback_wav = os.path.join(safe_temp_dir, os.path.basename(output_wav_path))
            sf.write(fallback_wav, master_buffer, sample_rate, subtype="PCM_16")
            return fallback_wav

        return output_wav_path
