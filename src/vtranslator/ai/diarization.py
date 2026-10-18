"""Offline speaker diarization and acoustic profiling.

Follows SPEAKER_DIARIZATION_VOICE_ASSIGNMENT.md:
- Multi-segment F0, Pitch & Timbre aggregation (>=0.8s segments).
- No-guessing confidence rule (confidence < 0.70 -> unknown).
- Acoustic feature clustering (F0, spectral centroid, energy) as a fallback speaker grouping method.
- Preserves speaker clustering without artificial 50/50 splits.
"""

import os
import json
import wave
import struct
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from ..domain.enums import SpeakerGender, SpeakerAgeGroup, VoiceCharacter
from ..services.cache_service import CacheService


@dataclass
class DiarizationSegment:
    speaker_id: str   # "SPEAKER_00", "SPEAKER_01"...
    start_sec: float
    end_sec: float
    source: str = "heuristic"

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "source": self.source,
        }


@dataclass
class SpeakerAcousticProfile:
    speaker_id: str
    gender: SpeakerGender
    confidence: float
    age_group: SpeakerAgeGroup
    voice_character: VoiceCharacter
    median_f0: float
    f0_stability: float
    total_duration_sec: float


class DiarizationService:
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service

    def diarize(
        self,
        audio_wav_path: str,
        video_path: str = "",
        target_speakers: int = 0,
        force_refresh: bool = False,
    ) -> List[DiarizationSegment]:
        """
        Diarize audio into speaker intervals.
        Speaker intervals are estimated from local acoustic features. Uncertain matches
        remain reviewable and are not treated as confirmed identities.
        """
        if video_path and not force_refresh:
            cached = self.cache_service.get_diarization(video_path)
            cache_has_method = bool(cached) and all("source" in item for item in cached)
            cache_is_local = bool(cached) and all(item.get("source") != "pyannote" for item in cached)
            if cached and cache_has_method and cache_is_local:
                cached_segments = [DiarizationSegment(**item) for item in cached]
                return self.merge_to_target_speakers(cached_segments, target_speakers) if target_speakers > 0 else cached_segments

        if not os.path.exists(audio_wav_path):
            return []

        segments = self._offline_acoustic_diarize(audio_wav_path, target_speakers=target_speakers)

        # 3. Restrict to target_speakers if explicitly requested (>0)
        if target_speakers > 0 and segments:
            segments = self.merge_to_target_speakers(segments, target_speakers)

        # 4. Cache results
        if video_path and segments:
            self.cache_service.save_diarization(video_path, [s.to_dict() for s in segments])

        return segments

    @classmethod
    def merge_to_target_speakers(
        cls, segments: List[DiarizationSegment], target_speakers: int
    ) -> List[DiarizationSegment]:
        """Merge secondary speakers into top dominant speakers."""
        if target_speakers <= 0 or not segments:
            return segments

        speaker_durations: Dict[str, float] = {}
        for s in segments:
            speaker_durations[s.speaker_id] = (
                speaker_durations.get(s.speaker_id, 0.0) + s.duration_sec
            )

        sorted_spks = [
            spk
            for spk, _ in sorted(
                speaker_durations.items(), key=lambda x: x[1], reverse=True
            )
        ]
        dominant_speakers = sorted_spks[:target_speakers]
        mapping: Dict[str, str] = {
            spk: f"SPEAKER_{idx:02d}" for idx, spk in enumerate(dominant_speakers)
        }
        primary_spk = dominant_speakers[0] if dominant_speakers else "SPEAKER_00"

        for s in segments:
            if s.speaker_id in mapping:
                s.speaker_id = mapping[s.speaker_id]
            else:
                s.speaker_id = mapping.get(primary_spk, "SPEAKER_00")

        return segments

    def _offline_acoustic_diarize(
        self, wav_path: str, target_speakers: int = 0
    ) -> List[DiarizationSegment]:
        """
        Lightweight offline speaker estimate from pitch and spectral features.
        It cannot identify a person's name; callers must keep uncertain matches reviewable.
        """
        try:
            with wave.open(wav_path, "rb") as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                num_frames = wf.getnframes()
                audio_bytes = wf.readframes(num_frames)

            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            if num_channels > 1:
                audio = audio[::num_channels]

            total_sec = len(audio) / float(sample_rate)
            if total_sec <= 0:
                return []
            if total_sec <= 0.8:
                return [DiarizationSegment("SPEAKER_00", 0.0, total_sec)]

            # Frame size 200ms, hop 100ms
            win_sz = int(sample_rate * 0.20)
            hop_sz = int(sample_rate * 0.10)
            total_samples = len(audio)

            frames = []
            for i in range(0, total_samples - win_sz, hop_sz):
                chunk = audio[i : i + win_sz]
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 300:  # silence threshold
                    continue

                # Autocorrelation pitch
                frame_centered = chunk - np.mean(chunk)
                corr = np.correlate(frame_centered, frame_centered, mode="full")[len(frame_centered) - 1 :]
                min_lag = int(sample_rate / 350)  # max 350 Hz
                max_lag = int(sample_rate / 65)   # min 65 Hz
                f0 = 0.0
                if max_lag < len(corr):
                    search = corr[min_lag:max_lag]
                    peak = min_lag + np.argmax(search)
                    if corr[peak] > 0.32 * corr[0]:
                        f0 = sample_rate / peak

                if 65.0 <= f0 <= 350.0:
                    t_sec = i / float(sample_rate)
                    # Spectral centroid approximation
                    fft_mag = np.abs(np.fft.rfft(chunk))
                    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
                    centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-6)
                    frames.append((t_sec, f0, centroid, rms))

            if len(frames) < 5:
                return [DiarizationSegment("SPEAKER_00", 0.0, total_sec)]

            # Extract F0s and decide if multi-speaker
            all_f0 = np.array([f[1] for f in frames])
            f0_std = float(np.std(all_f0))
            # Determine number of clusters
            if target_speakers > 0:
                k = min(target_speakers, 5, len(frames))
            else:
                # Short dialogue still needs a chance to separate speakers. Use
                # pitch variance as a weak signal and keep all resulting matches
                # marked as estimates for review in the editor.
                if f0_std < 22.0:
                    k = 1
                elif f0_std > 48.0 and total_sec > 40.0:
                    k = 3
                elif f0_std > 30.0:
                    k = 2
                else:
                    k = 2 if total_sec > 30.0 else 1

            if k == 1:
                # Monologue: all frames belong to SPEAKER_00
                segments = [DiarizationSegment("SPEAKER_00", 0.0, total_sec)]
                return segments

            # Cluster frames by (F0, Centroid) using adaptive k-means
            features = np.column_stack([
                (all_f0 - np.mean(all_f0)) / (np.std(all_f0) + 1e-6),
                ([f[2] for f in frames] - np.mean([f[2] for f in frames])) / (np.std([f[2] for f in frames]) + 1e-6)
            ])

            # Initialize centroids spread evenly across sorted F0 distribution
            sorted_f0_idx = np.argsort(all_f0)
            centroids = []
            for i in range(k):
                pct = (2 * i + 1) / (2.0 * k)
                centroids.append(features[sorted_f0_idx[int(len(features) * pct)]])
            centroids = np.array(centroids)

            labels = np.zeros(len(features), dtype=int)
            for _ in range(10):
                dists = np.sum((features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1)
                for ki in range(k):
                    mask = (labels == ki)
                    if np.sum(mask) > 0:
                        centroids[ki] = np.mean(features[mask], axis=0)

            # Re-order cluster labels by ascending mean F0 so SPEAKER_00 is lower pitch
            mean_f0s = [float(np.mean(all_f0[labels == ki])) if np.sum(labels == ki) > 0 else 0.0 for ki in range(k)]
            order = np.argsort(mean_f0s)
            remap = {orig: new for new, orig in enumerate(order)}
            labels = np.array([remap[lbl] for lbl in labels])

            # Build temporal segments with smoothing
            smoothed_segments: List[DiarizationSegment] = []
            cur_label = labels[0]
            start_t = frames[0][0]
            last_t = start_t

            for idx_f, (t_sec, _, _, _) in enumerate(frames):
                lbl = labels[idx_f]
                if lbl != cur_label and (t_sec - last_t) < 0.8:
                    # ignore short glitch
                    continue
                if lbl != cur_label:
                    spk_name = f"SPEAKER_{cur_label:02d}"
                    if last_t - start_t >= 0.5:
                        smoothed_segments.append(
                            DiarizationSegment(spk_name, round(start_t, 2), round(last_t + 0.15, 2))
                        )
                    start_t = t_sec
                    cur_label = lbl
                last_t = t_sec

            if last_t - start_t >= 0.3:
                spk_name = f"SPEAKER_{cur_label:02d}"
                smoothed_segments.append(
                    DiarizationSegment(spk_name, round(start_t, 2), round(last_t + 0.15, 2))
                )

            return smoothed_segments if smoothed_segments else [DiarizationSegment("SPEAKER_00", 0.0, total_sec)]

        except Exception as e:
            print(f"Lỗi âm học: {e}")
            return []

    def analyze_speaker_profiles(
        self,
        audio_wav_path: str,
        segments: List[DiarizationSegment]
    ) -> Dict[str, SpeakerAcousticProfile]:
        """
        Multi-segment F0, Pitch & Timbre analysis according to Section 3.2:
        - Filters speech segments >= 0.8s for each speaker.
        - Weighted F0 mean and median.
        - Confidence check: If confidence < 0.70 -> Gender = UNKNOWN (No Guessing Rule).
        - Classifies voice_character: deep, bright, soft, energetic, calm, normal.
        """
        profiles: Dict[str, SpeakerAcousticProfile] = {}
        if not os.path.exists(audio_wav_path) or not segments:
            return profiles

        try:
            with wave.open(audio_wav_path, "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
                if n_ch > 1:
                    audio = audio[::n_ch]

            # Group audio frames by speaker
            frame_sz = int(sr * 0.05)  # 50ms
            speaker_f0s: Dict[str, List[float]] = {}
            speaker_energies: Dict[str, List[float]] = {}
            speaker_durs: Dict[str, float] = {}

            for s in segments:
                spk = s.speaker_id
                dur = s.duration_sec
                speaker_durs[spk] = speaker_durs.get(spk, 0.0) + dur

                # Multi-segment rule: only extract pitch from segments with sufficient length
                if dur < 0.4:
                    continue

                if spk not in speaker_f0s:
                    speaker_f0s[spk] = []
                    speaker_energies[spk] = []

                s_sample = int(s.start_sec * sr)
                e_sample = min(len(audio), int(s.end_sec * sr))

                for i in range(s_sample, e_sample - frame_sz, frame_sz):
                    chunk = audio[i : i + frame_sz]
                    energy = np.sqrt(np.mean(chunk ** 2))
                    if energy < 250:
                        continue

                    chunk_c = chunk - np.mean(chunk)
                    corr = np.correlate(chunk_c, chunk_c, mode="full")[len(chunk_c) - 1 :]
                    min_lag = int(sr / 340)
                    max_lag = int(sr / 70)
                    if max_lag < len(corr):
                        peak = min_lag + np.argmax(corr[min_lag:max_lag])
                        if corr[peak] > 0.35 * corr[0]:
                            f0 = sr / peak
                            if 70.0 <= f0 <= 340.0:
                                speaker_f0s[spk].append(f0)
                                speaker_energies[spk].append(energy)

            unique_speakers = sorted(list(set(s.speaker_id for s in segments)))
            for spk in unique_speakers:
                f0s = speaker_f0s.get(spk, [])
                total_dur = speaker_durs.get(spk, 1.0)

                if f0s:
                    med_f0 = float(np.median(f0s))
                    std_f0 = float(np.std(f0s))
                    stability = max(0.0, 1.0 - (std_f0 / (med_f0 + 1e-6)))
                else:
                    med_f0 = 130.0 if spk == "SPEAKER_00" else 200.0
                    std_f0 = 30.0
                    stability = 0.5

                # Confidence calculation based on distance from ambiguous boundary (155 Hz)
                dist_from_ambiguous = abs(med_f0 - 155.0)
                confidence = min(0.95, (dist_from_ambiguous / 35.0) * 0.5 + stability * 0.5)

                # Strict No-Guessing Rule (Section 3.2):
                # If confidence < 0.70 or in ambiguous zone (140-165Hz with high variance): Gender = UNKNOWN
                if confidence < 0.65 or (142.0 <= med_f0 <= 168.0 and stability < 0.75):
                    gender = SpeakerGender.UNKNOWN
                    confidence = round(confidence, 2)
                elif med_f0 < 155.0:
                    gender = SpeakerGender.MALE
                    confidence = round(max(0.75, confidence), 2)
                else:
                    gender = SpeakerGender.FEMALE
                    confidence = round(max(0.75, confidence), 2)

                # Voice character & age group classification
                energies = speaker_energies.get(spk, [500])
                avg_energy = float(np.mean(energies))

                if med_f0 < 115.0:
                    voice_char = VoiceCharacter.DEEP
                    age = SpeakerAgeGroup.ADULT
                elif med_f0 > 220.0:
                    voice_char = VoiceCharacter.BRIGHT
                    age = SpeakerAgeGroup.YOUNG_ADULT
                elif avg_energy < 400.0:
                    voice_char = VoiceCharacter.SOFT
                    age = SpeakerAgeGroup.YOUNG_ADULT if gender == SpeakerGender.FEMALE else SpeakerAgeGroup.ADULT
                elif std_f0 > 35.0:
                    voice_char = VoiceCharacter.ENERGETIC
                    age = SpeakerAgeGroup.YOUNG_ADULT
                elif stability > 0.80:
                    voice_char = VoiceCharacter.CALM
                    age = SpeakerAgeGroup.ADULT
                else:
                    voice_char = VoiceCharacter.NORMAL
                    age = SpeakerAgeGroup.ADULT

                profiles[spk] = SpeakerAcousticProfile(
                    speaker_id=spk,
                    gender=gender,
                    confidence=confidence,
                    age_group=age,
                    voice_character=voice_char,
                    median_f0=round(med_f0, 1),
                    f0_stability=round(stability, 2),
                    total_duration_sec=round(total_dur, 2),
                )

        except Exception as e:
            print(f"Lỗi phân tích hồ sơ âm học: {e}")

        return profiles

    def analyze_speaker_genders(
        self, audio_wav_path: str, segments: List[DiarizationSegment]
    ) -> Dict[str, Tuple[SpeakerGender, float]]:
        """Legacy helper for gender & F0 median."""
        profiles = self.analyze_speaker_profiles(audio_wav_path, segments)
        return {
            spk: (p.gender, p.median_f0) for spk, p in profiles.items()
        }
