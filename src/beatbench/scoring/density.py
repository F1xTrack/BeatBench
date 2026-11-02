"""Note density vs declared difficulty and local track energy."""
from __future__ import annotations
import librosa
import numpy as np
from ..config import DEAD_ZONE_SECONDS, TARGET_NPS
from .elements import discrete_hits
from ..types import GeneratedPackage, Subscore


def _select_difficulty_files(pkg: GeneratedPackage, difficulty: str) -> list[dict]:
    for beatmap_set in pkg.info.get("difficultyBeatmapSets", []):
        for beatmap in beatmap_set.get("_difficultyBeatmaps", []):
            if beatmap.get("_difficulty") == difficulty:
                filename = beatmap.get("_beatmapFilename")
                if filename in pkg.difficulties:
                    return [pkg.difficulties[filename]]
    return list(pkg.difficulties.values())


def _quiet_energy_ratio(pkg: GeneratedPackage, note_times_sec: np.ndarray) -> float | None:
    if pkg.audio_path is None:
        return None
    try:
        y, sr = librosa.load(str(pkg.audio_path), sr=None, mono=True)
        if len(y) == 0:
            return None
        rms = librosa.feature.rms(y=y, frame_length=int(sr), hop_length=int(sr))[0]
        if len(rms) == 0:
            return None
        threshold = float(np.quantile(rms, 0.25))
        idx = np.clip((note_times_sec // 1.0).astype(int), 0, len(rms) - 1)
        return float(np.mean(rms[idx] <= threshold))
    except Exception:
        return None


def validate_density(
    pkg: GeneratedPackage,
    difficulty: str,
    bpm: float,
    target_nps: float | None = None,
) -> Subscore:
    target = (
        float(target_nps) if target_nps is not None and target_nps > 0
        else TARGET_NPS.get(difficulty, TARGET_NPS["ExpertPlus"])
    )
    sec_per_beat = 60.0 / bpm if bpm and bpm > 0 else None
    all_beats = sorted(
        n["b"] for dat in _select_difficulty_files(pkg, difficulty)
        for n in discrete_hits(dat))
    if not all_beats or sec_per_beat is None:
        return Subscore("density", 0.0, {"reason": "no notes or no bpm"})

    window, step = 4.0, 2.0
    edges = np.arange(all_beats[0], max(all_beats[-1], all_beats[0] + step), step)
    win_nps = np.array([
        sum(1 for b in all_beats if e <= b < e + window)
        for e in edges]) / (window * sec_per_beat)
    deviation = float(np.mean(np.abs(win_nps - target)) / target)

    dead_zones = sum(
        1 for prev, cur in zip(all_beats, all_beats[1:])
        if (cur - prev) * sec_per_beat > DEAD_ZONE_SECONDS)

    monotony = 0.0
    if len(win_nps) > 1 and float(np.mean(win_nps)) > 0 \
            and float(np.std(win_nps)) / float(np.mean(win_nps)) < 0.1:
        monotony = 0.1

    score = float(np.clip(1.0 - deviation - min(0.05 * dead_zones, 0.3) - monotony, 0.0, 1.0))

    details = {
        "mean_nps": round(float(np.mean(win_nps)), 3),
        "target_nps": target,
        "dead_zones_count": dead_zones,
        "monotony_penalty": monotony,
    }
    note_times_sec = np.array([b * sec_per_beat for b in all_beats])
    quiet_ratio = _quiet_energy_ratio(pkg, note_times_sec)
    if quiet_ratio is None:
        details["energy_note"] = "quiet-energy penalty skipped: audio missing or unreadable"
    else:
        score *= 1.0 - min(quiet_ratio, 0.4)
        details["quiet_ratio"] = round(quiet_ratio, 4)
    return Subscore("density", round(score, 4), details)
