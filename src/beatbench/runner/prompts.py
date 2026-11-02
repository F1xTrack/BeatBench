"""Prompt builders for the two model calls: Info.dat generation and difficulty generation."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import librosa

from ..config import MAX_AUDIO_SECONDS
from ..types import TrackMeta

PROMPT_VERSION = "mapping-basics-v3-arcs-chains-colors"

_SYSTEM_PROMPT = (
    "Create a complete, enjoyable Standard Beat Saber map from the supplied signal. "
    "Use the benchmark analysis tool when available. Your FINAL reply must be EXACTLY ONE "
    "JSON object: no markdown fences, commentary, code or abbreviated arrays.\n"
    "Format: Info.dat fields use the convention shown in the example: version (\"2.2.0\"), "
    "songName, songSubName, songAuthorName, levelAuthorName, beatsPerMinute, songFilename, "
    "coverImageFilename, environmentName; difficultyBeatmapSets -> _difficultyBeatmaps "
    "with _difficulty, _difficultyRank, _beatmapFilename, _noteJumpMovementSpeed, "
    "_noteJumpStartBeatOffset. Choose jump speed/offset for readable approach and "
    "comfortable movement at the requested difficulty.\n"
    "V3 difficulty: version=\"3.3.0\", colorNotes {b,x,y,c,d}, bombNotes, "
    "obstacles {b,x,y,d,w,h}, basicBeatmapEvents {b,et,i,f}. "
    "Native V3 arcs go in sliders: {b,c,x,y,d,tb,tx,ty,tc,m}; tb is the tail beat, "
    "tx/ty the tail cell, tc the tail color and m is 0 or 1. Use an arc only for a "
    "musically justified sustained/curved swing, with compatible head and tail notes. "
    "Native chains (multiblocks) go in burstSliders: {b,c,x,y,d,tb,tx,ty,sc,s}; sc is "
    "the positive slice count and s is the positive squish factor. Chains create repeated "
    "playable slices between b and tb: count them toward density and leave recovery after them. "
    "Do not use chains as a cheap replacement for an entire melody or overlap incompatible objects. "
    "Note b is a beat, not a second; at constant BPM b = seconds * BPM / 60. "
    "x=0..3 left to right, y=0..2 bottom to top; c=0 left hand, c=1 right hand. "
    "d: 0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right, 6 down-left, "
    "7 down-right, 8 any direction.\n"
    "Musicality: infer timing from the supplied signal, including the first beat's position. "
    "Map a coherent audible layer (drums, bass, melody or vocals), its changing rhythms, "
    "accents and phrases. Constant BPM does not imply constant note spacing. "
    "Develop patterns with musical sections; repeat them when the music repeats, "
    "rather than looping one clockwork pattern over the entire song. "
    "Target NPS is a whole-track average: simultaneous notes count separately; "
    "local density and deliberate pauses should follow the music.\n"
    "Hand flow: track EACH hand's previous swing independently. Normally alternate "
    "forehand/backhand strokes so its next cut follows naturally from the previous endpoint. "
    "Repeated downstrokes or other same-direction cuts require a deliberate, comfortable return; "
    "alternating note colors does not fix either hand's reset. Allow recovery time and "
    "avoid awkward reach, hand collisions and notes hidden by other notes/walls.\n"
    "Lighting: author visible, music-responsive lighting throughout the track. "
    "For a legacy environment such as DefaultEnvironment, normal basicBeatmapEvents "
    "with et=0..4 control light groups: i=0 off; 1/5 on in secondary/primary color; "
    "2/6 flash then stay on; 3/7 flash then fade to black; f=1 normal brightness. "
    "Use groups, accents and section changes; an empty array or one event at the start "
    "does not provide a musical light show. Chroma/Noodle are optional; simple working "
    "lighting and readable gameplay come first.\n"
    "Palette: choose both saberColors and sceneColors from the supplied cover/theme. "
    "sceneColors {left:[r,g,b], right:[r,g,b]} controls the two scene light colors, "
    "separately from the sabers; RGB channels are integers 0..255. "
    "Honor an explicit visual palette in the track context. Example colors are not defaults.\n"
    "Safety: never place two notes in the same cell on the same beat, bombs on notes, "
    "or full-height center walls over notes, including during a wall's duration. "
    "Obstacles d is duration in beats, not cut direction. "
    "The JSON examples show syntax only, not sufficient content: generate the full map and light track."
)

_TEMPLATES_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prompt_manifest() -> dict:
    return {"version": PROMPT_VERSION,
            "system_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "templates_sha256": _TEMPLATES_SHA256}


_INFO_SKELETON = (
    '{"info": {"version": "2.2.0", "songName": "...", "songSubName": "", '
    '"songAuthorName": "...", "levelAuthorName": "...", "beatsPerMinute": 0.0, '
    '"songFilename": "song.ogg", "coverImageFilename": "cover.png", '
    '"environmentName": "DefaultEnvironment", '
    '"difficultyBeatmapSets": '
    '[{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": '
    '[{"_difficulty": "ExpertPlus", "_difficultyRank": 9, '
    '"_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 14, '
    '"_noteJumpStartBeatOffset": 0}]}]}, '
    '"saberColors": {"left": [255, 80, 80], "right": [40, 120, 255]}, '
    '"sceneColors": {"left": [255, 80, 80], "right": [40, 120, 255]}}'
)

_DIFFICULTY_SKELETON = (
    '{"version": "3.3.0", "colorNotes": [{"b": 0, "x": 1, "y": 0, "c": 0, "d": 1}], '
    '"bombNotes": [], "obstacles": [], "sliders": [], "burstSliders": [], '
    '"basicBeatmapEvents": [{"b": 0, "et": 0, "i": 1, "f": 1}]}'
)

_CONTINUATION_SKELETON = (
    '{"colorNotes": [{"b": 0, "x": 1, "y": 0, "c": 0, "d": 1}], '
    '"bombNotes": [], "obstacles": [], "sliders": [], "burstSliders": [], "basicBeatmapEvents": []}'
)


def build_info_messages(
    meta: TrackMeta,
    cover_path: Path | None,
    audio_path: Path,
    audio_fullrate: bool = False,
    signal_parts: list[dict] | None = None,
) -> list[dict]:
    user_content = list(signal_parts) if signal_parts is not None else [_audio_part(audio_path, fullrate=audio_fullrate)]
    cover = _cover_part(cover_path)
    if cover is not None:
        user_content.append(cover)
    user_content.append({"type": "text", "text": _context_text(meta) + (
        "\nAnalyze the audio and cover, then reply with a single JSON object of the form "
        '{"info": {...}, "saberColors": {...}, "sceneColors": {...}} containing complete Info.dat in the supplied field convention.\n'
        'REQUIRED top-level key: "saberColors": {"left": [r,g,b], "right": [r,g,b]} '
        "with integer 0-255 RGB values chosen to match the cover art palette. "
        "This key is MANDATORY and must appear at the top level of your JSON reply.\n"
        'Also REQUIRED: top-level "sceneColors": {"left": [r,g,b], "right": [r,g,b]} '
        'with integer 0-255 RGB values for the scene lighting, following the cover/theme '
        'and any explicit track palette. Optional keys: leftBoost, rightBoost, obstacles. '
        'Choose these colors yourself; do not copy example RGB values.\n'
        'The field difficultyBeatmapSets MUST be an ARRAY of objects like this exact shape:\n'
        + _INFO_SKELETON
    )})
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_difficulty_messages(
    meta: TrackMeta,
    info_json: dict,
    difficulty: str,
    cover_path: Path | None,
    audio_path: Path,
    audio_fullrate: bool = False,
    signal_parts: list[dict] | None = None,
) -> list[dict]:
    user_content = list(signal_parts) if signal_parts is not None else [_audio_part(audio_path, fullrate=audio_fullrate)]
    cover = _cover_part(cover_path)
    if cover is not None:
        user_content.append(cover)
    text = (
        _context_text(meta)
        + f"\nTarget difficulty: {difficulty}\n"
        + "Info.dat:\n"
        + json.dumps(info_json, indent=2)
        + "\nReply with a single JSON object: the V3 beatmap file for that difficulty only.\n"
        "This is ONE complete beatmap file, NOT Info.dat or a sample. "
        "Include notes and musical lighting across the full supplied track. "
        "Before replying, check timing, each hand's flow, section development and light events. "
        "This small example illustrates the JSON shape only:\n"
        + _DIFFICULTY_SKELETON
    )
    user_content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_continuation_messages(
    meta: TrackMeta,
    info_json: dict,
    difficulty: str,
    current_dat: dict,
    cover_path: Path | None,
    audio_path: Path,
    signal_parts: list[dict] | None = None,
    audio_fullrate: bool = False,
) -> list[dict]:
    user_content = list(signal_parts) if signal_parts is not None else [_audio_part(audio_path, fullrate=audio_fullrate)]
    cover = _cover_part(cover_path)
    if cover is not None:
        user_content.append(cover)
    bpm = float(info_json.get("beatsPerMinute", 0) or 0) or float(meta.bpm or 0.0)
    duration = _duration_seconds(audio_path)
    effective = min(duration, MAX_AUDIO_SECONDS)
    last_beat = max(
        [float(n["b"]) for n in current_dat.get("colorNotes", [])]
        + [float(s.get("tb", s.get("b", 0))) for s in current_dat.get("sliders", [])]
        + [float(s.get("tb", s.get("b", 0))) for s in current_dat.get("burstSliders", [])]
        + [0.0])
    last_sec = last_beat * 60.0 / bpm if bpm > 0 else 0.0
    text = (
        _context_text(meta)
        + f"\nTarget difficulty: {difficulty}\n"
        + "Info.dat:\n"
        + json.dumps(info_json, indent=2)
        + "\nCurrent map so far:\n"
        + json.dumps(current_dat)
        + f"\nThe map so far ends at beat {last_beat} "
          f"(≈{last_sec:.1f} seconds of {effective:.1f} seconds total). "
          "Continue both notes and lighting through the remaining musical sections, "
          "preserving each hand's swing flow at the join. Allow intentional musical pauses. "
          "Reply with a single JSON object containing ONLY the continuation: "
           '{"colorNotes": [...], "bombNotes": [...], "obstacles": [...], "sliders": [...], '
           '"burstSliders": [...], "basicBeatmapEvents": [...]} using ABSOLUTE beats '
          f"(continuing from beat {last_beat}). "
          f"Do not repeat notes before beat {last_beat}.\n"
        "Use exactly this shape:\n"
        + _CONTINUATION_SKELETON
    )
    user_content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _context_text(meta: TrackMeta) -> str:
    lines = [
        f"Track title: {meta.title}",
        f"Track author: {meta.author}",
    ]
    if meta.bpm is not None:
        lines.append(f"Desired BPM: {meta.bpm}")
    if meta.nps is not None:
        lines.append(
            f"Target note density: {meta.nps:.2f} notes per second on average "
            "across the whole track. Count both hands' notes, including simultaneous notes. "
            "Match this average while varying local density with the music."
        )
    lines.append("Difficulties: " + ", ".join(meta.difficulties))
    if meta.notes:
        lines.append("Track context: " + meta.notes)
    lines.append(
        "Choose left/right saber colors that match the cover art palette "
        "(complementary, high contrast between sabers)."
    )
    return "\n".join(lines)


def _audio_part(audio_path: Path, fullrate: bool = False) -> dict:
    fmt = audio_path.suffix.lower().lstrip(".")
    if (
        fmt in ("wav", "mp3")
        and _duration_seconds(audio_path) <= MAX_AUDIO_SECONDS
    ):
        return {
            "type": "input_audio",
            "input_audio": {"data": _b64(audio_path), "format": fmt},
        }
    return {
        "type": "input_audio",
        "input_audio": {
            "data": _convert_to_mp3(audio_path, fullrate=fullrate),
            "format": "mp3",
        },
    }


def _duration_seconds(audio_path: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(audio_path)))
    except Exception as exc:
        raise RuntimeError(f"cannot determine duration of {audio_path}: {exc}") from exc


def _convert_to_mp3(audio_path: Path, fullrate: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "converted.mp3"
        channels = "2" if fullrate else "1"
        sample_rate = "44100" if fullrate else "16000"
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                channels,
                "-ar",
                sample_rate,
                "-t",
                str(MAX_AUDIO_SECONDS),
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {audio_path}: {proc.stderr[-2000:]}"
            )
        return _b64(out)


def _cover_part(cover_path: Path | None) -> dict | None:
    if cover_path is None:
        return None
    data = _b64(cover_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{'png' if cover_path.suffix.lower() == '.png' else 'jpeg'};base64,{data}"},
    }


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
