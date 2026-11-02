from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackMeta:
    title: str = ""
    author: str = ""
    bpm: float | None = None
    nps: float | None = None
    difficulties: list[str] = field(default_factory=lambda: ["ExpertPlus"])
    notes: str = ""


@dataclass
class GeneratedPackage:
    info: dict
    difficulties: dict[str, dict]
    audio_path: Path
    cover_path: Path | None


@dataclass
class Subscore:
    name: str
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class RunResult:
    model: str
    song_stem: str
    gate_passed: bool
    subscores: list[Subscore] = field(default_factory=list)
    final_score: float | None = None
    status: str = "ok"
    errors: list[str] = field(default_factory=list)
    diagnostics: dict | None = None
    assessment: dict | None = None
