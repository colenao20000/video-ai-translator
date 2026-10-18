"""Domain Enums for Video AI Translator."""

from enum import Enum


class SpeakerGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class SpeakerAgeGroup(str, Enum):
    CHILD = "child"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    SENIOR = "senior"


class VoiceStyle(str, Enum):
    TIN_TUC = "tin_tuc"
    DOC_TRUYEN = "doc_truyen"
    TU_NHIEN = "tu_nhien"


class VoiceCharacter(str, Enum):
    DEEP = "deep"
    BRIGHT = "bright"
    SOFT = "soft"
    ENERGETIC = "energetic"
    CALM = "calm"
    NORMAL = "normal"


class TtsStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"


class RenderStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FlipMode(str, Enum):
    OFF = "OFF"
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"
    BOTH = "Both"


class CoverMode(str, Enum):
    BLUR = "blur"
    SOLID = "solid"


class SubtitleStyleEngine(str, Enum):
    ASS = "ass"
    DRAWTEXT = "drawtext"


class LanguageCode(str, Enum):
    VI = "vi"
    EN = "en"
    JA = "ja"
    AUTO = "auto"


class TaskState(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class CharacterMatchStatus(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    UNKNOWN = "unknown"
    MANUAL = "manual"


class SegmentFitStatus(str, Enum):
    FIT = "fit"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    ADJUSTED = "adjusted"
    COLLISION = "collision"
