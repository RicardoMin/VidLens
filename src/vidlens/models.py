"""Shared boundary objects; timestamps are seconds on the original video timeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class VidLensError(Exception):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    kind: str = "subtitle"
    source: str = "unknown"
    language: str = ""
    id: str = ""


@dataclass
class Video:
    id: str
    url: str
    platform: str
    title: str
    duration: float
    chapters: list[dict[str, Any]] = field(default_factory=list)
    language: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcquiredVideo:
    video: Video
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class MediaAsset:
    path: Path
    offset: float = 0.0
    duration: float = 0.0
    audio_only: bool = False


@dataclass
class Frame:
    timestamp: float
    path: Path
    width: int
    height: int
    ocr: str = ""
    selection: str = "uniform"
    duplicate_of: float | None = None
    requested_timestamp: float | None = None
    visible_until: float | None = None
    ocr_regions: list[dict[str, Any]] = field(default_factory=list)
    crop_box: list[float] | None = None

    def to_dict(self, *, include_layout: bool = False) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        if not include_layout:
            result.pop("ocr_regions")
        return result
