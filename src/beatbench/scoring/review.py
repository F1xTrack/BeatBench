"""V3 diagnostic review. Observations are not a calibrated quality rating."""
from __future__ import annotations

from collections import Counter
from bisect import bisect_right
from functools import lru_cache
import math
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks, resample_poly
import soundfile as sf

from ..config import MAX_AUDIO_SECONDS
from ..types import GeneratedPackage
from .elements import discrete_hits

REVIEW_VERSION = "diagnostic-v3.0"
WINDOW_SECONDS = 4.0
SHORT_GAP_SECONDS = 1.0
_ANGLES = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}
_EXAMPLE_LIMIT = 12


class _BeatClock:
    """Integrate official V3 BPM events; never reinterpret editor-only BPM data."""
    def __init__(self, bpm: float, events: list[dict]):
        changes = {0.0: bpm}
        for event in events:
            beat, rate = float(event["b"]), float(event["m"])
            if not math.isfinite(beat) or beat < 0 or not math.isfinite(rate) or rate <= 0:
                raise ValueError("invalid official BPM event")
            changes[beat] = rate
        self.beats = sorted(changes)
        self.rates = [changes[b] for b in self.beats]
        self.times = [0.0]
        for index in range(1, len(self.beats)):
            self.times.append(self.times[-1] + (self.beats[index] - self.beats[index - 1]) * 60 / self.rates[index - 1])

    def seconds(self, beat: float) -> float:
        index = max(0, bisect_right(self.beats, beat) - 1)
        return self.times[index] + (beat - self.beats[index]) * 60 / self.rates[index]


def _cv(values) -> float | None:
    values = np.asarray(values, dtype=float)
    return float(np.std(values) / np.mean(values)) if len(values) and np.mean(values) > 0 else None


def _corr(left, right) -> float | None:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 3 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _distances(times, targets):
    times, targets = np.asarray(times), np.asarray(targets)
    if not len(targets):
        return None
    indexes = np.searchsorted(targets, times)
    left = targets[np.clip(indexes - 1, 0, len(targets) - 1)]
    right = targets[np.clip(indexes, 0, len(targets) - 1)]
    return np.minimum(np.abs(times - left), np.abs(times - right))


@lru_cache(maxsize=4)
def _audio_features(path: str, size: int, mtime_ns: int):
    # size/mtime belong to the cache key. Audio is never exposed to a model here.
    with sf.SoundFile(path) as stream:
        sample_rate = stream.samplerate
        full_duration = stream.frames / sample_rate
        samples = stream.read(int(MAX_AUDIO_SECONDS * sample_rate), dtype="float32", always_2d=True)
    duration = len(samples) / sample_rate
    if duration <= 0:
        raise ValueError("empty audio")
    edges = np.arange(0, duration + WINDOW_SECONDS, WINDOW_SECONDS)
    rms = np.asarray([math.sqrt(float(np.mean(samples[int(a * sample_rate):min(len(samples), int(b * sample_rate))] ** 2)))
                      for a, b in zip(edges, edges[1:])])
    levels = 20 * np.log10(np.maximum(rms, 1e-12))
    # A low-volume musical section is not declared silence by its percentile.
    silence_threshold = max(-70.0, float(np.max(levels)) - 40.0)
    audible = levels >= silence_threshold
    mono = samples.mean(axis=1)
    target_rate = 22050
    divisor = math.gcd(sample_rate, target_rate)
    mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor)
    hop = 220  # About 10 ms. Fixed independently of the map's guessed BPM.
    magnitude = np.abs(librosa.stft(mono, n_fft=1024, hop_length=hop))
    flux = np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0)
    frequencies = librosa.fft_frequencies(sr=target_rate, n_fft=1024)
    attacks = {}
    for name, lo, hi in (("full", 30, 11025), ("low", 30, 250), ("mid", 250, 2000), ("high", 2000, 11025)):
        curve = flux[(frequencies >= lo) & (frequencies < hi)].sum(axis=0)
        if not len(curve) or float(curve.max()) <= 1e-10:
            attacks[name] = np.array([], dtype=float)
            continue
        peaks, _ = find_peaks(curve, distance=5, prominence=max(float(curve.max()) * 0.03, float(np.std(curve)) * 0.5))
        times = peaks * hop / target_rate
        attacks[name] = times[(times >= 0) & (times < duration)]
    return {"duration": duration, "full_duration": full_duration, "edges": edges, "rms": rms, "levels": levels,
            "silence_threshold": silence_threshold, "audible": audible, "attacks": attacks}


def _geometry_is_unknown(note: dict) -> bool:
    custom = note.get("customData") or {}
    return bool(custom.get("animation") or custom.get("_animation") or
                any(key in custom for key in ("position", "_position", "coordinates", "_positionOverride")))


def _element_geometry(data: dict) -> dict:
    """Validate V3 arcs/chains and expose conservative, scorer-friendly evidence."""
    arcs = data.get("sliders", []) or []
    chains = data.get("burstSliders", []) or []
    invalid = []

    def finite(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def valid_endpoint(item: dict, prefix: str) -> bool:
        return (finite(item.get(prefix + "b")) and item.get(prefix + "x") in range(4)
                and item.get(prefix + "y") in range(3))

    arc_duration = 0.0
    for index, arc in enumerate(arcs):
        ok = (isinstance(arc, dict) and finite(arc.get("b")) and finite(arc.get("tb"))
              and arc.get("tb") > arc.get("b") and arc.get("c") in (0, 1)
              and arc.get("x") in range(4) and arc.get("y") in range(3)
              and arc.get("tx") in range(4) and arc.get("ty") in range(3)
              and arc.get("d") in range(9) and arc.get("tc") in (0, 1)
              and arc.get("m") in (0, 1))
        if not ok and len(invalid) < _EXAMPLE_LIMIT:
            invalid.append({"kind": "arc", "index": index})
        if ok:
            arc_duration += float(arc["tb"] - arc["b"])

    chain_slices = 0
    chain_duration = 0.0
    for index, chain in enumerate(chains):
        ok = (isinstance(chain, dict) and finite(chain.get("b")) and finite(chain.get("tb"))
              and chain.get("tb") >= chain.get("b") and chain.get("c") in (0, 1)
              and chain.get("x") in range(4) and chain.get("y") in range(3)
              and chain.get("tx") in range(4) and chain.get("ty") in range(3)
              and chain.get("d") in range(9) and isinstance(chain.get("sc"), int)
              and chain.get("sc") >= 1 and finite(chain.get("s")) and chain.get("s") > 0)
        if not ok and len(invalid) < _EXAMPLE_LIMIT:
            invalid.append({"kind": "burstSlider", "index": index})
        if ok:
            chain_slices += int(chain["sc"])
            chain_duration += float(chain["tb"] - chain["b"])

    return {"valid": not invalid, "arcs": len(arcs), "chains": len(chains),
            "chain_slices": chain_slices, "arc_duration_beats": arc_duration,
            "chain_duration_beats": chain_duration, "invalid": invalid,
            "limitation": "Arc paths and chain slice positions are represented, but final wrist trajectory and renderer behavior still require playtest."}


def _playable_hits(data: dict) -> list[dict]:
    """Return a conservative discrete projection of V3 playable elements."""
    return discrete_hits(data)


def _same_swing(previous: dict, current: dict, clock: _BeatClock) -> bool:
    """Conservatively group straight stacks/sliders; repeated cells never group."""
    if previous.get("d") not in _ANGLES or previous.get("d") != current.get("d"):
        return False
    if previous.get("a", 0) or current.get("a", 0):
        return False
    delta = float(current["b"]) - float(previous["b"])
    if delta < 0 or delta > 0.125 or clock.seconds(current["b"]) - clock.seconds(previous["b"]) > 0.09:
        return False
    dx, dy = current["x"] - previous["x"], current["y"] - previous["y"]
    if dx == dy == 0:
        return False
    angle = math.radians(_ANGLES[previous["d"]])
    along = dx * math.cos(angle) + dy * math.sin(angle)
    across = abs(dx * math.sin(angle) - dy * math.cos(angle))
    # Simultaneous stacks may be listed in either order in JSON.
    return across < 0.25 and (abs(along) > 0.1 if delta == 0 else along > 0.1)


def _flow(notes: list[dict], clock: _BeatClock, timebase_assessed: bool) -> dict:
    output = {}
    for color in (0, 1):
        hand = sorted((n for n in notes if n["c"] == color), key=lambda n: (n["b"], n["x"], n["y"]))
        groups = []
        for note in hand:
            if groups and not _geometry_is_unknown(note) and not _geometry_is_unknown(groups[-1][-1]) and _same_swing(groups[-1][-1], note, clock):
                groups[-1].append(note)
            else:
                groups.append([note])
        counts = Counter()
        examples = []
        for previous, current in zip(groups, groups[1:]):
            a, b = previous[-1], current[0]
            gap = clock.seconds(b["b"]) - clock.seconds(a["b"])
            if (_geometry_is_unknown(a) or _geometry_is_unknown(b) or a.get("a", 0) or b.get("a", 0)
                    or a.get("d") not in _ANGLES or b.get("d") not in _ANGLES):
                counts["unassessed_transitions"] += 1
                continue
            angle = abs(_ANGLES[a["d"]] - _ANGLES[b["d"]])
            angle = min(angle, 360 - angle)
            counts["assessed_transitions"] += 1
            category = "reversal_135_180" if angle >= 135 else ("ambiguous_90" if angle == 90 else "return_motion_0_45")
            counts[category] += 1
            if timebase_assessed and category == "return_motion_0_45" and 0 < gap <= SHORT_GAP_SECONDS:
                counts["short_gap_return_candidates"] += 1
            if category != "reversal_135_180" and len(examples) < _EXAMPLE_LIMIT:
                examples.append({"beat": b["b"], "seconds": round(clock.seconds(b["b"]), 4) if timebase_assessed else None,
                                 "gap_seconds": round(gap, 4) if timebase_assessed else None, "angle_degrees": angle, "category": category})
        transitions = max(len(groups) - 1, 0)
        counts["unassessed_transitions"] += max(0, transitions - counts["assessed_transitions"] - counts["unassessed_transitions"])
        signature = [(n["x"], n["y"], n["d"], round(clock.seconds(n["b"]) - clock.seconds(hand[i - 1]["b"]), 2))
                     for i, n in enumerate(hand) if i]
        periodicity = []
        for period in (1, 2, 4, 6, 8, 12, 16, 24, 32):
            if len(signature) >= 3 * period:
                fraction = sum(a == b for a, b in zip(signature[period:], signature[:-period])) / (len(signature) - period)
                periodicity.append((fraction, period))
        repeat, period = max(periodicity, default=(None, None), key=lambda pair: pair[0])
        output[str(color)] = {"notes": len(hand), "swing_groups": len(groups), "transitions": transitions,
                              "timebase_assessed": timebase_assessed,
                              **{key: counts[key] for key in ("assessed_transitions", "unassessed_transitions", "reversal_135_180",
                                 "ambiguous_90", "return_motion_0_45", "short_gap_return_candidates")},
                              "direction_counts": dict(Counter(n["d"] for n in hand)),
                              "periodic_match_fraction": repeat if timebase_assessed else None, "period_swing_notes": period if timebase_assessed else None, "examples": examples}
    return output


def _timing(times, features) -> dict:
    result = {}
    duration = features["duration"]
    shifts = np.random.default_rng(1138).uniform(0.25, max(0.251, duration - 0.25), 64)
    for name, attacks in features["attacks"].items():
        distances = _distances(times, attacks)
        if distances is None or not len(times):
            result[name] = {"available": False, "attacks": len(attacks)}
            continue
        support = float(np.mean(distances <= 0.05))
        chance = [float(np.mean(_distances(np.sort((times + shift) % duration), attacks) <= 0.05)) for shift in shifts]
        result[name] = {"available": True, "attacks": len(attacks), "within_50ms": support,
                        "within_100ms": float(np.mean(distances <= 0.1)), "median_distance_ms": float(np.median(distances) * 1000),
                        "shifted_baseline_mean_50ms": float(np.mean(chance)), "shifted_baseline_p95_50ms": float(np.quantile(chance, 0.95)),
                        "excess_over_baseline_50ms": support - float(np.mean(chance))}
    return result


def _lighting(data: dict) -> dict:
    events = data.get("basicBeatmapEvents", [])
    lights = [e for e in events if e.get("et") in (0, 1, 2, 3, 4, 6, 7, 10, 11)]
    on = [e for e in lights if e.get("i") in (1, 2, 3, 5, 6, 7, 9, 10, 11) and e.get("f", 1) > 0]
    boxes = {key: len(data.get(key, [])) for key in (
        "lightColorEventBoxGroups", "lightRotationEventBoxGroups", "lightTranslationEventBoxGroups")}
    custom = data.get("customData") or {}
    return {"basic_events": len(events), "standard_light_events": len(lights),
            "standard_light_event_value_counts": dict(Counter(e.get("i") for e in lights)),
            "standard_light_event_first_beat": min((e["b"] for e in lights), default=None),
            "standard_light_event_last_beat": max((e["b"] for e in lights), default=None),
            "standard_light_on_candidates": len(on), "event_box_groups": boxes,
            "custom_events": len(custom.get("customEvents", custom.get("_customEvents", []))),
            "authored_standard_lighting_present": bool(lights or any(boxes.values())),
            "visible_lighting_quality": None,
            "limitation": "Event presence is not visible brightness or musical lighting quality. Environment, custom events and game settings require gameplay review."}


def _palette(info: dict, profile: dict) -> dict:
    index = profile.get("_beatmapColorSchemeIdx", profile.get("beatmapColorSchemeIdx", -1))
    schemes = info.get("colorSchemes", [])
    scheme = {}
    if isinstance(index, int) and 0 <= index < len(schemes):
        wrapper = schemes[index]
        if wrapper.get("useOverride"):
            scheme = wrapper.get("colorScheme", wrapper)
    custom = profile.get("_customData", profile.get("customData", {})) or {}
    left = custom.get("_envColorLeft", scheme.get("environmentColor0"))
    right = custom.get("_envColorRight", scheme.get("environmentColor1"))
    valid = lambda color: (isinstance(color, dict) and all(isinstance(color.get(k), (int, float))
                           and math.isfinite(color[k]) and 0 <= color[k] <= 1 for k in ("r", "g", "b")))
    return {"native_scheme_index": index, "native_scheme_override": bool(scheme),
            "scene_left": left, "scene_right": right,
            "valid_scene_rgb_pair": valid(left) and valid(right),
            "saber_left": custom.get("_colorLeft", scheme.get("saberAColor")),
            "saber_right": custom.get("_colorRight", scheme.get("saberBColor")),
            "quality": None, "limitation": "Declared palette only; theme fit and in-game rendering are not inferred from RGB presence."}


def review_package(pkg: GeneratedPackage, target_nps: float | None = None, analyze_audio: bool = True) -> dict:
    """Return per-difficulty evidence, preserving unknowns and keeping rating null."""
    warnings = []
    features = None
    if analyze_audio:
        try:
            path = Path(pkg.audio_path).resolve()
            stat = path.stat()
            features = _audio_features(str(path), stat.st_size, stat.st_mtime_ns)
        except Exception as exc:
            warnings.append(f"audio unavailable: {type(exc).__name__}: {exc}")
    bpm = float(pkg.info.get("beatsPerMinute", 0) or 0)
    offset = float(pkg.info.get("songTimeOffset", 0) or 0)
    reviews = {}
    for name, data in pkg.difficulties.items():
        notes = data.get("colorNotes", [])
        hits = _playable_hits(data)
        issues = []
        limitations = []
        if not math.isfinite(bpm) or bpm <= 0:
            reviews[name] = {"assessed": False, "reason": "missing positive BPM"}
            continue
        clock = _BeatClock(bpm, [])
        timebase_assessed = True
        if offset or data.get("_BPMChanges") or (data.get("customData") or {}).get("_BPMChanges"):
            limitations.append("nonzero song offset / legacy editor BPM timebase is unsupported")
            timebase_assessed = False
        if any(e.get("et") == 100 for e in data.get("basicBeatmapEvents", [])):
            limitations.append("legacy BPM event timebase is unsupported")
            timebase_assessed = False
        try:
            clock = _BeatClock(bpm, data.get("bpmEvents", []))
        except (KeyError, TypeError, ValueError) as exc:
            limitations.append(f"invalid official BPM timebase: {exc}")
            timebase_assessed = False
        geometry_unknown = sum(_geometry_is_unknown(n) or bool(n.get("a", 0)) for n in notes)
        elements = _element_geometry(data)
        if geometry_unknown:
            limitations.append("animated/rotated notes require additional swing analysis")
        if elements["arcs"] or elements["chains"]:
            limitations.append(elements["limitation"])
        if not elements["valid"]:
            issues.append({"code": "invalid_arc_or_chain_geometry", "examples": elements["invalid"]})
        seen = set()
        duplicates = []
        for n in notes:
            key = (n["b"], n["x"], n["y"])
            if key in seen and len(duplicates) < _EXAMPLE_LIMIT:
                duplicates.append({"beat": n["b"], "x": n["x"], "y": n["y"]})
            seen.add(key)
        if duplicates:
            issues.append({"code": "duplicate_cell_candidates", "examples": duplicates, "geometry_unknown": bool(geometry_unknown)})
        flow = _flow(notes, clock, timebase_assessed)
        lighting = _lighting(data)
        if not lighting["authored_standard_lighting_present"]:
            issues.append({"code": "no_authored_standard_lighting", "note": "No standard light events or light event box groups; inspect custom lighting and actual visibility in game."})
        elif not lighting["standard_light_on_candidates"] and not lighting["event_box_groups"]["lightColorEventBoxGroups"]:
            issues.append({"code": "no_standard_light_on_candidates", "note": "Standard events do not appear to turn colored lights on; inspect custom lighting in game."})
        if lighting["standard_light_events"] == 1 and not any(lighting["event_box_groups"].values()) and not lighting["custom_events"]:
            issues.append({"code": "single_light_event", "note": "A single light event does not demonstrate a musical light show or visible lighting quality."})
        for hand, evidence in flow.items():
            if evidence["short_gap_return_candidates"]:
                issues.append({"code": "same_hand_return_motion", "hand": int(hand), "count": evidence["short_gap_return_candidates"],
                               "note": "Candidate requires checking bomb resets, setup and recovery; not a physics verdict."})
            if evidence["ambiguous_90"]:
                issues.append({"code": "same_hand_90_degree_transitions", "hand": int(hand), "count": evidence["ambiguous_90"],
                               "note": "90-degree transitions need contextual review."})
        profile = next((entry for group in pkg.info.get("difficultyBeatmapSets", [])
                        for entry in group.get("_difficultyBeatmaps", group.get("difficultyBeatmaps", []))
                        if entry.get("_beatmapFilename", entry.get("beatmapFilename")) == name), {})
        palette = _palette(pkg.info, profile)
        if not palette["valid_scene_rgb_pair"]:
            issues.append({"code": "scene_palette_not_declared", "note": "No valid per-difficulty scene RGB pair; environment defaults/custom lighting may still apply."})
        review = {"assessed": True, "note_count": len(notes), "playable_hit_count": len(hits),
                  "flow": flow, "lighting": lighting, "elements": elements,
                  "palette": palette,
                  "spawn": {"njs": profile.get("_noteJumpMovementSpeed", profile.get("noteJumpMovementSpeed")),
                            "beat_offset": profile.get("_noteJumpStartBeatOffset", profile.get("noteJumpStartBeatOffset")),
                            "quality": None, "limitation": "No universal NJS range or visibility/physics rating is assumed."},
                  "issues": issues, "limitations": limitations,
                  "content": {"light_events": len(data.get("basicBeatmapEvents", [])),
                              "bombs": len(data.get("bombNotes", [])), "walls": len(data.get("obstacles", [])),
                              "arcs": elements["arcs"], "chains": elements["chains"],
                              "chain_slices": elements["chain_slices"]}}
        if not hits:
            issues.append({"code": "empty_notes"})
        if features is not None and timebase_assessed:
            duration = features["duration"]
            full_duration = features["full_duration"]
            if full_duration > duration:
                limitations.append(f"audio timing/structure assessed only in first {duration:g} seconds; full-track NPS uses full duration")
            all_times = np.sort(np.asarray([clock.seconds(n["b"]) for n in hits]))
            times = np.unique(all_times[(all_times >= 0) & (all_times < duration)])
            counts, _ = np.histogram(all_times[(all_times >= 0) & (all_times < duration)], features["edges"])
            occupied = counts > 0
            audible = features["audible"]
            active_count = int(audible.sum())
            full_windows = int(duration // WINDOW_SECONDS)
            density_cv = _cv(counts[:full_windows])
            attacks, _ = np.histogram(features["attacks"]["full"], features["edges"])
            rhythm_cv = _cv(attacks[:full_windows])
            actual_nps = int(np.sum((all_times >= 0) & (all_times < full_duration))) / full_duration
            out_of_audio = int(np.sum((all_times < 0) | (all_times >= full_duration)))
            review["structure"] = {"window_seconds": WINDOW_SECONDS, "duration_seconds": full_duration,
                                   "analyzed_audio_seconds": duration, "timing_unique_hits_assessed": len(times),
                                   "actual_nps": actual_nps, "target_nps": target_nps,
                                   "nps_relative_error": abs(actual_nps - target_nps) / target_nps if target_nps and target_nps > 0 else None,
                                   "tail_ratio": float(all_times.max() / full_duration) if len(all_times) else 0,
                                   "occupied_audible_window_fraction": float(np.mean(occupied[audible])) if active_count else None,
                                   "audible_windows": active_count, "empty_audible_windows": int(np.sum(audible & ~occupied)),
                                   "note_count_cv": density_cv, "audio_attack_count_cv": rhythm_cv,
                                   "energy_density_correlation": _corr(features["rms"][:full_windows], counts[:full_windows]),
                                   "attack_density_correlation": _corr(attacks[:full_windows], counts[:full_windows]),
                                   "silence_threshold_dbfs": features["silence_threshold"], "notes_outside_audio": out_of_audio}
            review["timing"] = _timing(times, features)
            if out_of_audio:
                issues.append({"code": "notes_outside_audio", "count": out_of_audio})
            if active_count and np.any(audible & ~occupied):
                issues.append({"code": "empty_audible_windows", "count": int(np.sum(audible & ~occupied))})
            repeats = [v["periodic_match_fraction"] for v in flow.values() if v["periodic_match_fraction"] is not None]
            if density_cv is not None and density_cv < 0.08 and rhythm_cv is not None and rhythm_cv > 0.15 and len(repeats) == 2 and min(repeats) > 0.8:
                issues.append({"code": "rigid_pattern_over_changing_audio", "note": "Constant cadence and repeating hand patterns while audio attack activity changes; review musical intent."})
            if review["timing"]["full"].get("available") and review["timing"]["full"]["excess_over_baseline_50ms"] <= 0:
                issues.append({"code": "attack_alignment_not_above_shifted_baseline", "note": "Detector evidence is weak; soft attacks and chosen musical layer require human review."})
        reviews[name] = review
    return {"version": REVIEW_VERSION, "quality_score": None, "calibration_status": "uncalibrated",
            "scope": "json-and-audio" if analyze_audio else "json-only", "audio_assessed": features is not None,
            "verdict": "requires_playtest", "difficulty_reviews": reviews, "warnings": warnings,
            "parameters": {"short_gap_seconds": SHORT_GAP_SECONDS, "attack_tolerance_seconds": 0.05, "null_shifts": 64, "seed": 1138},
            "interpretation": "Raw diagnostic evidence. No component is a validated quality rating. Lighting is a separate experience axis; event/mod presence alone gives no quality bonus."}
