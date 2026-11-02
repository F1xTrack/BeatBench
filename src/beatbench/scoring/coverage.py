"""Coverage: how much of the (capped) track duration the generated notes span."""
from __future__ import annotations

from ..config import MAX_AUDIO_SECONDS
from .elements import map_end_beat
from ..types import GeneratedPackage, Subscore


def validate_coverage(pkg: GeneratedPackage, audio_duration_sec: float | None) -> Subscore:
    bpm = float(pkg.info.get("beatsPerMinute", 0) or 0)
    details = {"coverage_per_file": {}, "audio_duration_sec": audio_duration_sec}
    if bpm <= 0 or not audio_duration_sec or audio_duration_sec <= 0:
        return Subscore("coverage", 0.0, details)

    effective = min(float(audio_duration_sec), MAX_AUDIO_SECONDS)
    sec_per_beat = 60.0 / bpm
    per_file: dict[str, float] = {}
    for fname, dat in pkg.difficulties.items():
        last_time_sec = map_end_beat(dat) * sec_per_beat
        per_file[fname] = round(min(last_time_sec / effective, 1.0), 6)

    score = min(per_file.values()) if per_file else 0.0
    return Subscore("coverage", score, {"coverage_per_file": per_file, "audio_duration_sec": audio_duration_sec})
