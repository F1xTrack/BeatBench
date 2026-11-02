from __future__ import annotations

import numpy as np
import librosa

from beatbench.config import SYNC_TOLERANCE_MS
from beatbench.types import GeneratedPackage, Subscore
from .elements import discrete_hits

ACCENT_TOLERANCE_S = 0.100
MEDIAN_PENALTY_MS = 200.0


def _nearest_distance(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(targets, times)
    left = np.clip(idx - 1, 0, len(targets) - 1)
    right = np.clip(idx, 0, len(targets) - 1)
    return np.minimum(np.abs(times - targets[left]), np.abs(times - targets[right]))


def validate_sync(pkg: GeneratedPackage) -> Subscore:
    details = {
        "within_tolerance": 0.0,
        "median_deviation_ms": 0.0,
        "accent_coverage": 0.0,
    }
    bpm = pkg.info.get("beatsPerMinute")
    if not bpm or bpm <= 0:
        return Subscore(name="sync", score=0.0, details=details)

    notes = []
    for diff in pkg.difficulties.values():
        notes.extend(discrete_hits(diff))
    if not notes:
        return Subscore(name="sync", score=0.0, details=details)

    note_times = np.sort(np.array([n["b"] * 60.0 / bpm for n in notes]))

    y, sr = librosa.load(pkg.audio_path, sr=None)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    if len(onsets) == 0:
        return Subscore(name="sync", score=0.0, details=details)

    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.time_to_frames(onsets, sr=sr)
    strengths = envelope[frames]

    dists_s = _nearest_distance(note_times, np.asarray(onsets))
    within_tolerance = float(np.mean(dists_s <= SYNC_TOLERANCE_MS / 1000.0))
    median_deviation_ms = float(np.median(dists_s) * 1000.0)

    accent_count = max(1, int(round(len(strengths) * 0.3)))
    accent_idx = np.argsort(strengths)[-accent_count:]
    onset_times = np.asarray(onsets)[accent_idx]
    note_dists = _nearest_distance(onset_times, note_times)
    accent_coverage = float(np.mean(note_dists <= ACCENT_TOLERANCE_S))

    score = (
        0.6 * within_tolerance
        + 0.25 * accent_coverage
        + 0.15 * (1 - min(median_deviation_ms / MEDIAN_PENALTY_MS, 1.0))
    )

    details = {
        "within_tolerance": within_tolerance,
        "median_deviation_ms": median_deviation_ms,
        "accent_coverage": accent_coverage,
    }
    return Subscore(name="sync", score=round(score, 6), details=details)
