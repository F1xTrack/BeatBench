"""End-to-end evaluation pipeline: generate, gate, score, package."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
import uuid
from subprocess import TimeoutExpired

import librosa
from openai import APITimeoutError

from ..config import MAX_AUDIO_SECONDS
from ..export.packager import build_zip
from ..export.colors import apply_palette
from ..export.packager import sanitize_folder
from ..export.v2 import build_v2_zip
from ..scoring.aggregate import run_scoring
from ..scoring.elements import map_end_beat
from ..scoring.schema_check import DIFF_SCHEMA, INFO_SCHEMA, check_schema
from ..types import GeneratedPackage, TrackMeta
from .client import ModelClient, extract_json
from .prompts import (
    build_continuation_messages,
    build_difficulty_messages,
    build_info_messages,
    prompt_manifest,
)

CONTINUATION_COVERAGE_TARGET = 0.92
MAX_CONTINUATION_ATTEMPTS = 4
_MERGE_ARRAY_KEYS = ("colorNotes", "bombNotes", "obstacles", "sliders",
                     "burstSliders", "basicBeatmapEvents")
logger = logging.getLogger(__name__)


class RawLog:
    def __init__(self, raw_dir: Path):
        self._raw_dir = raw_dir
        self._index = 0

    def append(self, kind: str, messages: list[dict], content: str,
               usage: dict | None, elapsed_s: float,
               reasoning: str = "", reasoning_metadata: dict | None = None) -> None:
        self._index += 1
        payload = {
            "call_index": self._index,
            "kind": kind,
            "messages": messages,
            "reply_content": content,
            "usage": usage or {},
            "elapsed_s": elapsed_s,
            "reasoning": reasoning,
            "reasoning_metadata": reasoning_metadata or {},
        }
        (self._raw_dir / f"call_{self._index}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if reasoning:
            (self._raw_dir / f"call_{self._index}.reasoning.md").write_text(
                reasoning, encoding="utf-8")


def _api_error_message(exc: Exception) -> str:
    code = getattr(exc, "status_code", None)
    prefix = f"API error (HTTP {code})" if code is not None else "API error"
    return f"{prefix}: {exc}"


def _write_api_failure(results_dir: Path, exc: Exception) -> Path:
    (results_dir / "status.json").write_text(json.dumps({
        "status": "failed_timeout" if isinstance(exc, (APITimeoutError, TimeoutError, TimeoutExpired)) else "failed_api",
        "errors": [_api_error_message(exc)],
    }), encoding="utf-8")
    return results_dir


def _call_with_retry(client: ModelClient, log: RawLog, kind: str,
                     messages: list[dict],
                     response_format: dict | None = None,
                     reasoning_effort: str | None = None) -> dict:
    reply = client.chat(messages, response_format=response_format,
                        reasoning_effort=reasoning_effort)
    log.append(kind, messages, reply.content, reply.usage, reply.elapsed_s,
               reply.reasoning, reply.reasoning_metadata)
    try:
        return extract_json(reply.content)
    except ValueError as exc:
        retry_messages = list(messages) + [{
            "role": "user",
            "content": f"Your previous reply was not valid JSON ({exc}). "
                       "Reply again with exactly one JSON object and nothing else.",
        }]
        retry_reply = client.chat(retry_messages, response_format=response_format,
                                  reasoning_effort=reasoning_effort)
        log.append(kind, retry_messages, retry_reply.content,
                   retry_reply.usage, retry_reply.elapsed_s,
                   retry_reply.reasoning, retry_reply.reasoning_metadata)
        return extract_json(retry_reply.content)


def _log_api_failure(log: RawLog, kind: str, messages: list[dict],
                     exc: Exception) -> None:
    log.append(kind, messages, f"API ERROR: {exc}", None, 0.0)


def _response_formats(constrained: bool) -> tuple[dict, dict]:
    if not constrained:
        return None, None
    info_fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": "info",
            "schema": {
                "type": "object",
                "required": ["info"],
                "properties": {"info": INFO_SCHEMA},
            },
        },
    }
    diff_fmt = {
        "type": "json_schema",
        "json_schema": {"name": "difficulty", "schema": DIFF_SCHEMA},
    }
    return info_fmt, diff_fmt


def _effective_bpm(info: dict, meta: TrackMeta) -> float:
    bpm = float(info.get("beatsPerMinute", 0) or 0)
    if bpm <= 0 and meta.bpm:
        bpm = float(meta.bpm)
    return bpm


def _audio_duration(audio_path: Path) -> float | None:
    try:
        return float(librosa.get_duration(path=str(audio_path)))
    except Exception:
        return None


def _coverage_fraction(dat: dict, bpm: float,
                       effective_duration: float) -> float:
    sec_per_beat = 60.0 / bpm
    last_beat = map_end_beat(dat)
    return min(last_beat * sec_per_beat / effective_duration, 1.0)


def _item_key(array_key: str, item: dict):
    if array_key == "colorNotes":
        return tuple(item.get(k) for k in ("b", "x", "y", "c"))
    return json.dumps(item, sort_keys=True)


def _merge_fragment(base: dict, fragment: dict) -> dict:
    merged = {k: v for k, v in base.items() if k not in _MERGE_ARRAY_KEYS}
    for key in _MERGE_ARRAY_KEYS:
        items = list(base.get(key) or [])
        seen = {_item_key(key, item) for item in items}
        for item in (fragment.get(key) or []):
            item_key = _item_key(key, item)
            if item_key not in seen:
                seen.add(item_key)
                items.append(item)
        if key in ("colorNotes", "sliders", "burstSliders", "basicBeatmapEvents"):
            items.sort(key=lambda n: (float(n.get("b", 0)),
                                      float(n.get("tb", n.get("b", 0)))))
        merged[key] = items
    return merged


def _continue_map(client: ModelClient, log: RawLog, meta: TrackMeta,
                  info: dict, difficulty: str, dat: dict,
                  cover_path: Path | None, audio_path: Path,
                  response_format: dict | None,
                  errors: list[str],
                  reasoning_effort: str | None = None,
                  signal_parts: list[dict] | None = None,
                  audio_fullrate: bool = False) -> dict:
    bpm = _effective_bpm(info, meta)
    duration = _audio_duration(audio_path)
    if bpm <= 0 or not duration or duration <= 0:
        return dat
    effective = min(duration, MAX_AUDIO_SECONDS)
    coverage = _coverage_fraction(dat, bpm, effective)
    attempts = 0
    while (coverage < CONTINUATION_COVERAGE_TARGET
           and attempts < MAX_CONTINUATION_ATTEMPTS):
        context = {"signal_parts": signal_parts} if signal_parts is not None else {}
        if audio_fullrate:
            context["audio_fullrate"] = True
        messages = build_continuation_messages(
            meta, info, difficulty, dat, cover_path, audio_path, **context)
        try:
            fragment = _call_with_retry(
                client, log, "difficulty_continue", messages,
                response_format=response_format,
                reasoning_effort=reasoning_effort)
        except ValueError:
            break
        except Exception as exc:
            _log_api_failure(log, "difficulty_continue", messages, exc)
            break
        dat = _merge_fragment(dat, fragment)
        coverage = _coverage_fraction(dat, bpm, effective)
        attempts += 1
    if coverage < CONTINUATION_COVERAGE_TARGET:
        last_beat = map_end_beat(dat)
        errors.append(
            f"map ends at {last_beat * 60 / bpm:.0f}s of {effective:.0f}s")
    return dat


def _evaluate_track(audio: Path, endpoint: str, key: str, model: str,
                   meta: TrackMeta, cfg: dict,
                   constrained: bool = False, v2: bool = False,
                   reasoning_effort: str | None = None,
                   audio_fullrate: bool = False,
                   signal_parts: list[dict] | None = None,
                    client=None, output_dir: Path | None = None,
                    model_call_timeout: float = 300) -> Path:
    results_dir = output_dir or Path("results") / model / audio.stem
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("*.json"):
        stale.unlink()
    for stale in raw_dir.glob("*.reasoning.md"):
        stale.unlink()
    log = RawLog(raw_dir)
    client = client or ModelClient(endpoint, key, model, timeout=model_call_timeout,
                                   live_dir=raw_dir if endpoint else None)
    cover_path = find_cover(audio)
    info_fmt, diff_fmt = _response_formats(constrained)

    context = {"signal_parts": signal_parts} if signal_parts is not None else {}
    info_messages = build_info_messages(meta, cover_path, audio,
                                        audio_fullrate=audio_fullrate, **context)
    try:
        parsed = _call_with_retry(
            client, log, "info", info_messages,
            response_format=info_fmt,
            reasoning_effort=reasoning_effort)
    except ValueError as exc:
        (results_dir / "status.json").write_text(json.dumps({
            "status": "failed_info",
            "errors": [str(exc)],
        }), encoding="utf-8")
        return results_dir
    except Exception as exc:
        _log_api_failure(log, "info", info_messages, exc)
        return _write_api_failure(results_dir, exc)

    info = parsed.get("info", {})
    colors = parsed.get("saberColors")
    if colors is not None and "saberColors" not in info:
        info = {**info, "saberColors": colors}
    scene = parsed.get("sceneColors", info.get("sceneColors"))
    if scene is not None:
        info = {**info, "sceneColors": scene}
    apply_palette(info, info.get("saberColors"), scene)
    info["levelAuthorName"] = model
    pkg = GeneratedPackage(info=info, difficulties={},
                           audio_path=audio, cover_path=cover_path)

    gen_errors: list[str] = []
    warnings: list[str] = []
    for difficulty in meta.difficulties:
        diff_messages = build_difficulty_messages(
            meta, pkg.info, difficulty, cover_path, audio,
            audio_fullrate=audio_fullrate, **context)
        try:
            dat = _call_with_retry(
                client, log, "difficulty", diff_messages,
                response_format=diff_fmt,
                reasoning_effort=reasoning_effort)
        except ValueError as exc:
            gen_errors.append(f"{difficulty}: {exc}")
            continue
        except Exception as exc:
            _log_api_failure(log, "difficulty", diff_messages, exc)
            return _write_api_failure(results_dir, exc)
        dat = _continue_map(
            client, log, meta, pkg.info, difficulty, dat,
            cover_path, audio, diff_fmt, warnings,
            reasoning_effort=reasoning_effort,
            signal_parts=signal_parts, audio_fullrate=audio_fullrate)
        pkg.difficulties[f"{difficulty}.dat"] = dat

    status = "partial" if gen_errors else "ok"
    gate_passed, gate_errors = check_schema(pkg)
    if not gate_passed:
        status = "invalid"

    try:
        run_scoring(pkg, meta, model, results_dir, cfg,
                    song_stem=audio.stem, gate_errors=gate_errors,
                    generation_prompt=prompt_manifest())
    except Exception as exc:
        (results_dir / "status.json").write_text(json.dumps({
            "status": "scoring_failed",
            "errors": [f"scoring failed: {exc}"],
        }), encoding="utf-8")
        return results_dir

    if gen_errors or warnings or status != "ok":
        _patch_results_status(results_dir, status, gen_errors + warnings)

    if gate_passed:
        build_zip(pkg, results_dir, audio.stem)
        if v2:
            try:
                build_v2_zip(pkg, results_dir, audio.stem)
            except Exception as exc:
                logger.warning("v2 export failed: %s", exc)

    return results_dir


def evaluate_track(audio: Path, endpoint: str, key: str, model: str,
                   meta: TrackMeta, cfg: dict,
                   constrained: bool = False, v2: bool = False,
                   reasoning_effort: str | None = None,
                   audio_fullrate: bool = False,
                   input_mode: str = "audio", backend: str = "api",
                   sandbox_distro: str = "Ubuntu-24.04",
                   output_dir: Path | None = None,
                   model_call_timeout: float = 300,
                   reasoning_summary: str = "detailed",
                   codex_working_dir: Path | None = None,
                   codex_cache_prefix: bool = False) -> Path:
    if not math.isfinite(model_call_timeout) or model_call_timeout <= 0:
        raise ValueError("model_call_timeout must be finite and positive")
    if input_mode not in ("audio", "spectrogram", "python"):
        raise ValueError("input_mode must be audio, spectrogram, or python")
    if backend not in ("api", "codex", "local"):
        raise ValueError("backend must be api, codex, or local")
    if backend == "local":
        from urllib.parse import urlsplit
        if urlsplit(endpoint).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("local backend must use a loopback endpoint")
        key = "local"
    if input_mode == "audio":
        if backend != "api":
            raise ValueError("Codex/local backends require spectrogram or python input mode")
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ") + "-" + uuid.uuid4().hex[:8]
        results_dir = output_dir or (Path("results/experiments/audio") / sanitize_folder(model) / audio.stem / run_id)
        results_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = results_dir / "experiment.json"
        manifest = {"run_id": run_id, "input_mode": "audio", "backend": backend,
                    "generation_prompt": prompt_manifest(), "endpoint": endpoint,
                    "model": model, "started_at": datetime.now(timezone.utc).isoformat(),
                    "status": "preparing", "track_meta": vars(meta), "weights": cfg.get("weights", {}),
                    "constrained": constrained, "v2": v2, "reasoning_effort": reasoning_effort,
                    "reasoning_capture": {"requested_summary": None,
                                          "kind": "openrouter_provider_reasoning",
                                          "reasoning_effort": reasoning_effort,
                                          "raw_internal_cot_available": False},
                    "model_call_timeout_seconds": model_call_timeout,
                    "notes": "Native audio input; provider-returned reasoning fields are captured when available."}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            out = _evaluate_track(audio, endpoint, key, model, meta, cfg,
                                  constrained, v2, reasoning_effort, audio_fullrate,
                                  output_dir=results_dir, model_call_timeout=model_call_timeout)
            status_path = out / "status.json"
            result_path = status_path if status_path.exists() else out / "results.json"
            manifest["status"] = json.loads(result_path.read_text(encoding="utf-8")).get("status")
            if (out / "results.json").exists():
                data = json.loads((out / "results.json").read_text(encoding="utf-8"))
                data.update({"input_mode": "audio", "backend": backend, "run_id": run_id})
                (out / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return out
        except Exception as exc:
            manifest["status"] = "failed_runtime"
            (results_dir / "status.json").write_text(json.dumps({
                "status": manifest["status"], "errors": [str(exc)]}), encoding="utf-8")
            return results_dir
        finally:
            from .reasoning import analyze_reasoning
            try:
                analysis, path = analyze_reasoning(results_dir)
                manifest["reasoning_capture"].update({
                    "status": analysis["capture_status"],
                    "summary_items": analysis["summary_items"],
                    "summary_characters": analysis["summary_characters"],
                    "provider_reasoning_calls": analysis.get("provider_reasoning_calls", 0),
                    "provider_reasoning_characters": analysis.get("provider_reasoning_characters", 0),
                    "openrouter_usage": analysis.get("openrouter_usage"),
                    "analysis": str(path),
                })
            except Exception as exc:
                manifest["reasoning_capture"].update({"status": "analysis_failed", "error": str(exc)})
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if audio_fullrate:
        raise ValueError("--audio-fullrate applies only to native audio input")
    from .representations import build_spectrograms, stage_signal
    from .sandbox import PythonSandbox
    from .tool_client import ToolClient

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    results_dir = output_dir or Path("results/experiments") / input_mode / sanitize_folder(model) / audio.stem / run_id
    results_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir()
    manifest_path = results_dir / "experiment.json"
    manifest = {"run_id": run_id, "input_mode": input_mode, "backend": backend,
                "generation_prompt": prompt_manifest(),
                "endpoint": endpoint if backend != "codex" else None,
                "model": model, "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "preparing", "track_meta": vars(meta), "weights": cfg.get("weights", {}),
                "constrained": constrained, "reasoning_effort": reasoning_effort,
                 "reasoning_capture": {"requested_summary": reasoning_summary if backend == "codex" else None,
                                       "kind": ("provider_reasoning_summary" if backend == "codex" else
                                                "openrouter_provider_reasoning" if backend == "api" else None),
                                       "reasoning_effort": reasoning_effort,
                                       "raw_internal_cot_available": False},
                "continuation_target": CONTINUATION_COVERAGE_TARGET,
                "max_continuations_per_difficulty": MAX_CONTINUATION_ATTEMPTS,
                "model_call_timeout_seconds": model_call_timeout,
                "codex_transport": {"version": "codex-prefix-v2" if codex_cache_prefix else "codex-transcript-v1",
                                    "shared_working_directory": str(codex_working_dir) if codex_working_dir else None,
                                    "fresh_ephemeral_session_per_call": True, "cross_map_history": False} if backend == "codex" else None,
                "notes": "Compare only runs with identical condition settings; Codex CLI adds its own system instructions."}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sandbox = None
    try:
        signal, signal_manifest = stage_signal(audio, results_dir / "signal")
        manifest["signal"] = signal_manifest
        if backend == "codex":
            from .codex_client import CodexClient
            client = CodexClient(model, raw_dir, timeout=model_call_timeout,
                                 reasoning_summary=reasoning_summary,
                                 working_dir=codex_working_dir, cache_prefix=codex_cache_prefix)
        else:
            client = ModelClient(endpoint, key, model, timeout=model_call_timeout,
                                 live_dir=raw_dir if backend == "api" else None)
        if input_mode == "spectrogram":
            parts, representation = build_spectrograms(signal, results_dir / "spectrograms")
            manifest["representation"] = representation
        else:
            sandbox = PythonSandbox(signal, find_cover(audio), distro=sandbox_distro)
            manifest["sandbox"] = sandbox.preflight()
            client = ToolClient(client, sandbox, raw_dir)
            parts = [{"type": "text", "text": (
                "The complete signal is available only through the python tool at /input/audio.wav. "
                "Analyze it yourself; no precomputed beats, onsets, music descriptions or map hints "
                "are supplied. All participants receive this same file and tool. "
                "Each Python call starts fresh. Save plots under /work/*.png. "
                "Run-wide limits: 12 Python calls, 180 seconds total Python wall time, "
                "30 seconds wall and 20 seconds CPU per call, 2 GiB address space per process, "
                "32 processes, 64 MiB per tmpfs workspace, 2 MiB transport output per call. "
                "Final answers must still be the requested map JSON."
            )}]
            manifest["model_request_limit"] = client.max_api_calls
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out = _evaluate_track(audio, endpoint, key, model, meta, cfg,
                              constrained, v2, reasoning_effort, False,
                              signal_parts=parts, client=client, output_dir=results_dir)
        status_path = out / "status.json"
        result_path = status_path if status_path.exists() else out / "results.json"
        manifest["status"] = json.loads(result_path.read_text(encoding="utf-8")).get("status")
        if (out / "results.json").exists():
            data = json.loads((out / "results.json").read_text(encoding="utf-8"))
            data.update({"input_mode": input_mode, "backend": backend, "run_id": run_id})
            (out / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return out
    except Exception as exc:
        manifest["status"] = "failed_setup" if manifest["status"] == "preparing" else "failed_runtime"
        (results_dir / "status.json").write_text(json.dumps({
            "status": manifest["status"], "errors": [str(exc)]}), encoding="utf-8")
        return results_dir
    finally:
        if sandbox is not None:
            manifest["python_calls"] = sandbox.calls
            manifest["python_wall_seconds"] = sandbox.elapsed_s
            sandbox.close()
        if backend in ("codex", "api"):
            from .reasoning import analyze_reasoning
            try:
                analysis, path = analyze_reasoning(results_dir)
                manifest["reasoning_capture"].update({"status": analysis["capture_status"],
                                                     "summary_items": analysis["summary_items"],
                                                     "summary_characters": analysis["summary_characters"],
                                                      "provider_reasoning_calls": analysis.get("provider_reasoning_calls", 0),
                                                      "provider_reasoning_characters": analysis.get("provider_reasoning_characters", 0),
                                                     "analysis": str(path)})
                if backend == "api":
                    manifest["reasoning_capture"]["openrouter_usage"] = analysis.get("openrouter_usage")
            except Exception as exc:
                manifest["reasoning_capture"].update({"status": "analysis_failed", "error": str(exc)})
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def find_cover(audio: Path) -> Path | None:
    for name in ("cover.jpg", "cover.png",
                 f"{audio.stem}.jpg", f"{audio.stem}.png"):
        candidate = audio.parent / name
        if candidate.exists():
            return candidate
    # A uniquely named dataset cover (e.g. Portal 2 Cover_2.png) is also input.
    images = sorted(p for p in audio.parent.iterdir()
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(images) == 1:
        return images[0]
    return None


def _patch_results_status(results_dir: Path, status: str,
                          errors: list[str]) -> None:
    path = results_dir / "results.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    data["errors"] = list(data.get("errors", [])) + list(errors)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
