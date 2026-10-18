"""Project Domain Model."""

import time
import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from .models import VideoMetadata, TransformSettings, BlurCoverSettings, AudioMixSettings, SubtitleStyleSettings, LogoSettings
from .characters import Character
from .segments import SubtitleSegment


class Project(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Dự án mới"
    video_path: str = ""
    video_metadata: Optional[VideoMetadata] = None
    segments: List[SubtitleSegment] = Field(default_factory=list)
    characters: List[Character] = Field(default_factory=list)
    transform_settings: TransformSettings = Field(default_factory=TransformSettings)
    blur_settings: BlurCoverSettings = Field(default_factory=BlurCoverSettings)
    audio_settings: AudioMixSettings = Field(default_factory=AudioMixSettings)
    subtitle_style: SubtitleStyleSettings = Field(default_factory=SubtitleStyleSettings)
    logo_settings: LogoSettings = Field(default_factory=LogoSettings)
    isolated_vocals_path: Optional[str] = None
    isolated_background_path: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def mark_updated(self) -> None:
        self.updated_at = time.time()

    def get_character_by_id(self, char_id: str) -> Optional[Character]:
        for c in self.characters:
            if c.id == char_id:
                return c
        return None

    def get_character_by_speaker(self, speaker_id: str) -> Optional[Character]:
        for c in self.characters:
            if c.speaker_id == speaker_id:
                return c
        return None

    def add_or_update_character(self, character: Character) -> None:
        for idx, c in enumerate(self.characters):
            if c.id == character.id:
                self.characters[idx] = character
                self.sync_character_voice_to_segments(character.id, character.voice_id)
                self.mark_updated()
                return
        self.characters.append(character)
        self.sync_character_voice_to_segments(character.id, character.voice_id)
        self.mark_updated()

    def update_character_voice(self, char_id: str, new_voice_id: str) -> bool:
        """
        Cập nhật giọng đọc cho nhân vật.
        Theo nguyên tắc Section 2: character.voice_id là nguồn sự thật duy nhất,
        phải lập tức đồng bộ voice_id sang mọi segment thuộc character đó.
        """
        char = self.get_character_by_id(char_id)
        if not char:
            return False
        char.update_voice(new_voice_id)
        self.sync_character_voice_to_segments(char_id, new_voice_id)
        self.mark_updated()
        return True

    def sync_character_voice_to_segments(self, char_id: str, voice_id: str) -> None:
        for seg in self.segments:
            if seg.character_id == char_id:
                seg.sync_voice_from_character(voice_id)

    def get_segment_by_id(self, seg_id: str) -> Optional[SubtitleSegment]:
        for s in self.segments:
            if s.segment_id == seg_id:
                return s
        return None

    def update_segment_translation(self, seg_id: str, new_text: str) -> bool:
        seg = self.get_segment_by_id(seg_id)
        if not seg:
            return False
        seg.update_translation(new_text)
        self.mark_updated()
        return True

    def assign_segment_character(self, seg_id: str, char_id: str) -> bool:
        seg = self.get_segment_by_id(seg_id)
        if seg and char_id == "char_unknown":
            seg.character_id = "char_unknown"
            seg.speaker_id = "SPEAKER_UNKNOWN"
            seg.speaker_source = "unassigned"
            seg.confidence = 0.0
            seg.voice_id = ""
            seg.invalidate_tts()
            self.mark_updated()
            return True
        char = self.get_character_by_id(char_id)
        if not seg or not char:
            return False
        seg.character_id = char.id
        seg.speaker_id = char.speaker_id
        seg.speaker_source = "manual"
        seg.confidence = 1.0
        seg.sync_voice_from_character(char.voice_id)
        self.mark_updated()
        return True
