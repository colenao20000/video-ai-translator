"""Character Resolver & Automatic Voice Assignment Engine.

Strictly adheres to SPEAKER_DIARIZATION_VOICE_ASSIGNMENT.md:
- Stable link: Speaker ID -> Character ID -> TTS Voice ID.
- Exact Bipartite Compatibility Score:
    S(i, j) = S_unique + S_gender + S_age + S_character
- S_unique: 0 if unused; -1000 if already assigned to prefer distinct voices when available.
- S_gender: +100 * confidence (match), -500 (mismatch), +10 (unknown).
- S_age: +30 (match).
- S_character: +40 (match).
"""

from typing import List, Dict, Optional, Tuple, Set
from ..domain.characters import Character, DEFAULT_PALETTE
from ..domain.voices import AVAILABLE_VOICES, VoiceOption, VoiceLibrary
from ..domain.segments import SubtitleSegment
from ..domain.enums import SpeakerGender, CharacterMatchStatus, SpeakerAgeGroup, VoiceCharacter
from .diarization import DiarizationSegment, SpeakerAcousticProfile


class CharacterResolver:
    @classmethod
    def build_speaker_characters(
        cls,
        diar_segments: List[DiarizationSegment],
        profiles: Optional[Dict[str, SpeakerAcousticProfile]] = None,
        existing_characters: Optional[List[Character]] = None,
    ) -> List[Character]:
        """Build stable character records from measured speaker IDs.

        The order follows first appearance in the source. Existing names and
        manual voice choices remain attached to the same speaker ID.
        """
        first_seen: Dict[str, float] = {}
        for segment in diar_segments:
            first_seen[segment.speaker_id] = min(
                first_seen.get(segment.speaker_id, segment.start_sec), segment.start_sec
            )

        old_by_speaker = {c.speaker_id: c for c in (existing_characters or [])}
        speaker_ids = sorted(first_seen, key=lambda speaker: (first_seen[speaker], speaker))
        characters: List[Character] = []
        for index, speaker_id in enumerate(speaker_ids):
            old = old_by_speaker.get(speaker_id)
            profile = (profiles or {}).get(speaker_id)
            if old:
                char = old.model_copy(deep=True)
            else:
                gender = profile.gender if profile else SpeakerGender.UNKNOWN
                safe_speaker = "".join(ch.lower() if ch.isalnum() else "_" for ch in speaker_id).strip("_")
                char = Character(
                    id=f"char_{safe_speaker or index + 1}",
                    speaker_id=speaker_id,
                    name=f"Người nói {index + 1}",
                    gender=gender,
                    confidence=profile.confidence if profile else 0.0,
                    age_group=profile.age_group if profile else SpeakerAgeGroup.ADULT,
                    voice_character=profile.voice_character if profile else VoiceCharacter.NORMAL,
                    color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
                    match_status=CharacterMatchStatus.PROBABLE,
                )
            if profile and char.match_status != CharacterMatchStatus.MANUAL:
                char.gender = profile.gender
                char.confidence = min(profile.confidence, 0.55)
                char.age_group = profile.age_group
                char.voice_character = profile.voice_character
                char.match_status = CharacterMatchStatus.PROBABLE
            characters.append(char)

        return cls.auto_assign_voices(characters, profiles=profiles)

    @staticmethod
    def calculate_compatibility_score(
        char: Character,
        voice: VoiceOption,
        assigned_voice_ids: Set[str],
        confidence: float = 0.90,
    ) -> float:
        """
        Calculates compatibility score S(i, j) = S_unique + S_gender + S_age + S_character
        defined in Section 4.2.
        """
        # 1. S_unique: -1000 penalty if voice already used by another character
        s_unique = -1000.0 if voice.id in assigned_voice_ids else 0.0

        # 2. S_gender
        if char.gender == SpeakerGender.UNKNOWN or voice.gender == SpeakerGender.UNKNOWN:
            s_gender = 10.0
        elif char.gender == voice.gender:
            s_gender = 100.0 * max(0.5, confidence)
        else:
            s_gender = -500.0  # Heavy penalty for wrong gender

        # 3. S_age
        s_age = 30.0 if char.age_group == voice.age_group else 0.0

        # 4. S_character
        s_character = 40.0 if char.voice_character == voice.voice_character else 0.0

        # Prefer the configured Gemini presets in this local voice catalog.
        s_clone = 200.0 if voice.id.startswith("Gemini ") else 0.0

        return s_unique + s_gender + s_age + s_character + s_clone

    @classmethod
    def auto_assign_voices(
        cls,
        characters: List[Character],
        profiles: Optional[Dict[str, SpeakerAcousticProfile]] = None,
    ) -> List[Character]:
        """
        Assign compatible voices greedily, preferring unused presets when available.
        This is a suggestion step; users can review and change each assignment.
        """
        assigned_voice_ids: Set[str] = set()

        # 1. Giữ lại các giọng do người dùng tự chọn thủ công (MANUAL)
        for c in characters:
            if c.match_status == CharacterMatchStatus.MANUAL and c.voice_id:
                canonical_v = VoiceLibrary.get_by_id(c.voice_id)
                if canonical_v:
                    c.voice_id = canonical_v.id
                    assigned_voice_ids.add(c.voice_id)

        # 2. Phân bổ giọng chuẩn giới tính & không trùng lặp cho các nhân vật còn lại
        for char in characters:
            canonical_v = VoiceLibrary.get_by_id(char.voice_id)
            if canonical_v:
                char.voice_id = canonical_v.id
            else:
                canonical_v = VoiceLibrary.get_by_id("Gemini Puck")
                char.voice_id = canonical_v.id if canonical_v else "Gemini Puck"

            # Cập nhật thông tin profile âm học nếu có
            conf = 0.85
            if profiles and char.speaker_id in profiles:
                p = profiles[char.speaker_id]
                if char.gender == SpeakerGender.UNKNOWN:
                    char.gender = p.gender
                if char.age_group == SpeakerAgeGroup.ADULT and p.age_group != SpeakerAgeGroup.ADULT:
                    char.age_group = p.age_group
                if char.voice_character == VoiceCharacter.NORMAL and p.voice_character != VoiceCharacter.NORMAL:
                    char.voice_character = p.voice_character
                conf = p.confidence

            # Nếu giọng hiện tại đã đúng giới tính và chưa bị trùng lặp, giữ lại
            voice_matches_gender = (char.gender == SpeakerGender.UNKNOWN) or (canonical_v.gender == char.gender)
            if (
                char.match_status == CharacterMatchStatus.MANUAL
                or (voice_matches_gender and char.voice_id not in assigned_voice_ids and char.voice_id != "Gemini Puck")
            ):
                assigned_voice_ids.add(char.voice_id)
                continue

            # Tìm giọng tốt nhất phù hợp giới tính và đặc tính nhân vật từ AVAILABLE_VOICES
            best_voice = None
            best_score = -999999.0

            # Ưu tiên các giọng cùng giới tính
            gender_candidates = [v for v in AVAILABLE_VOICES if char.gender == SpeakerGender.UNKNOWN or v.gender == char.gender]
            search_pool = gender_candidates if gender_candidates else AVAILABLE_VOICES

            for voice in search_pool:
                score = cls.calculate_compatibility_score(
                    char, voice, assigned_voice_ids, confidence=conf
                )
                if score > best_score:
                    best_score = score
                    best_voice = voice

            if best_voice:
                char.update_voice(best_voice.id)
                assigned_voice_ids.add(best_voice.id)
            else:
                # Fallback an toàn theo giới tính
                fallback = "Gemini Zephyr" if char.gender == SpeakerGender.FEMALE else "Gemini Puck"
                char.update_voice(fallback)
                assigned_voice_ids.add(char.voice_id)

        return characters


    @classmethod
    def resolve_speaker_to_character(
        cls,
        speaker_id: str,
        characters: List[Character],
        confidence: float = 0.85,
        is_off_screen: bool = False,
    ) -> Tuple[Optional[Character], CharacterMatchStatus]:
        """
        Resolves a speaker ID to a Character based on matching and confidence.
        - If is_off_screen: return a distinct narrator character, avoiding false attribution to on-screen actor.
        - If confidence < 0.50 or speaker unknown: return None, CharacterMatchStatus.UNKNOWN.
        - If matched character found: return character, CharacterMatchStatus.CONFIRMED (if confidence >= 0.70) else UNCONFIRMED.
        """
        if is_off_screen:
            narrator_char = next(
                (c for c in characters if "narrator" in c.id.lower() or "thuyết minh" in c.name.lower()),
                None
            )
            if not narrator_char:
                narrator_char = Character(
                    id="char_narrator",
                    speaker_id=speaker_id,
                    name="Thuyết minh",
                    gender=SpeakerGender.MALE,
                    voice_id="Minh Đức",
                    match_status=CharacterMatchStatus.CONFIRMED
                )
            return narrator_char, CharacterMatchStatus.CONFIRMED

        if confidence < 0.50:
            return None, CharacterMatchStatus.UNKNOWN

        matched = next((c for c in characters if c.speaker_id == speaker_id), None)
        if not matched:
            return None, CharacterMatchStatus.UNKNOWN

        status = CharacterMatchStatus.CONFIRMED if confidence >= 0.70 else CharacterMatchStatus.UNCONFIRMED
        return matched, status

    @staticmethod
    def match_segments_with_diarization(
        segments: List[SubtitleSegment],
        diar_segments: List[DiarizationSegment],
        characters: List[Character],
    ) -> List[SubtitleSegment]:
        """
        Match each subtitle segment with the dominant speaker based on time overlap.
        Preserves original segment start_ms/end_ms unconditionally.
        """
        if not segments:
            return []

        speaker_map: Dict[str, Character] = {c.speaker_id: c for c in characters}
        for i, seg in enumerate(segments):
            start_sec, end_sec = seg.start_sec, seg.end_sec
            seg_duration = max(0.05, end_sec - start_sec)
            overlaps: Dict[str, float] = {}
            for diar in diar_segments:
                overlap = max(0.0, min(end_sec, diar.end_sec) - max(start_sec, diar.start_sec))
                if overlap:
                    overlaps[diar.speaker_id] = overlaps.get(diar.speaker_id, 0.0) + overlap

            ranked = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
            best_speaker, best_overlap = ranked[0] if ranked else (None, 0.0)
            overlap_ratio = min(1.0, best_overlap / seg_duration)
            ambiguous = (
                len(ranked) > 1
                and ranked[1][1] >= best_overlap * 0.8
            )

            if best_speaker and overlap_ratio >= 0.20 and not ambiguous:
                seg.speaker_id = best_speaker
                seg.speaker_source = "heuristic"
                seg.confidence = min(overlap_ratio, 0.55)
                char = speaker_map.get(best_speaker)
                if char:
                    seg.character_id = char.id
                    seg.sync_voice_from_character(char.voice_id)
                else:
                    seg.character_id = "char_unknown"
                    seg.voice_id = ""
            else:
                # Do not silently attach dialogue to the first/previous person.
                seg.speaker_id = "SPEAKER_UNKNOWN"
                seg.character_id = "char_unknown"
                seg.speaker_source = "unassigned"
                seg.confidence = overlap_ratio if not ambiguous else 0.0
                seg.voice_id = ""

        return segments
