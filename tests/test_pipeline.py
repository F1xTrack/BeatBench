"""Pipeline tests: generation flow, raw logging, statuses, packaging."""
from __future__ import annotations
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from beatbench.config import DEFAULT_WEIGHTS
from beatbench.runner import pipeline
from beatbench.runner.client import ChatReply
from beatbench.types import TrackMeta
from tests.fixtures_helpers import FIXTURES

INFO = json.loads((FIXTURES / "maps" / "valid" / "info.json").read_text(encoding="utf-8"))
DIFF = json.loads((FIXTURES / "maps" / "valid" / "ExpertPlus.dat").read_text(encoding="utf-8"))

GARBAGE = "sorry, here is my reasoning without any braces at all"

SABER_COLORS = {"left": [255, 80, 80], "right": [40, 120, 255]}


def info_reply():
    return json.dumps({"info": INFO})


def make_partial():
    return json.loads(json.dumps(DIFF))


TAIL_FRAGMENT = {
    "colorNotes": [
        {"b": 8.5, "x": 2, "y": 0, "c": 1, "d": 0},
        {"b": 12, "x": 1, "y": 0, "c": 0, "d": 1},
        {"b": 12.5, "x": 2, "y": 0, "c": 1, "d": 0},
        {"b": 14, "x": 1, "y": 0, "c": 0, "d": 1},
    ],
    "bombNotes": [],
    "obstacles": [],
    "basicBeatmapEvents": [{"b": 16, "et": 0, "i": 1}],
}

V3_ELEMENT_FRAGMENT = {
    "colorNotes": [], "bombNotes": [], "obstacles": [],
    "sliders": [{"b": 8, "c": 0, "x": 0, "y": 0, "d": 1,
                 "tb": 12, "tx": 1, "ty": 1, "tc": 0, "m": 1}],
    "burstSliders": [{"b": 12.5, "c": 1, "x": 2, "y": 0, "d": 1,
                      "tb": 16, "tx": 3, "ty": 1, "sc": 4, "s": 0.9}],
    "basicBeatmapEvents": [],
}


def write_wav(tmp_path, seconds):
    wav = tmp_path / "track.wav"
    sf.write(wav, np.zeros(int(seconds * 44100), dtype="float32"),
             44100, format="WAV")
    return wav


class FakeClient:
    last_instance = None

    def __init__(self, endpoint, key, model, replies):
        self.endpoint, self.key, self.model = endpoint, key, model
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        FakeClient.last_instance = self

    @classmethod
    def factory(cls, replies):
        def make(endpoint, key, model):
            return cls(endpoint, key, model, replies)
        return make

    def chat(self, messages, response_format=None, reasoning_effort=None):
        self.calls.append((messages, response_format))
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, ChatReply):
            return item
        return ChatReply(content=item,
                         usage={"prompt_tokens": 5, "completion_tokens": 7},
                         elapsed_s=0.02)

    @property
    def messages_only(self):
        return [m for m, _ in self.calls]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wav = tmp_path / "track.wav"
    sf.write(wav, np.zeros(2000, dtype="float32"), 44100, format="WAV")
    return wav


def meta(difficulties=None):
    return TrackMeta(title="Test", author="Artist", bpm=120.0,
                     difficulties=difficulties or ["ExpertPlus"])


def text_of_call(messages):
    parts = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    parts.append(part["text"])
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def raw_calls(results_dir: Path):
    return sorted((results_dir / "raw").glob("call_*.json"))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_success_builds_zip_and_results(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    expected = Path("results") / "m1" / "track"
    assert out == expected
    assert out.is_dir() and (out / "raw").is_dir()

    calls = raw_calls(out)
    assert [load_json(c)["kind"] for c in calls] == ["info", "difficulty"]
    for c in calls:
        payload = load_json(c)
        assert set(payload) >= {"call_index", "kind", "messages",
                                "reply_content", "usage", "elapsed_s"}

    fake = FakeClient.last_instance
    assert len(fake.calls) == 2
    assert "Info.dat:" in text_of_call(fake.messages_only[1])
    assert '"songName"' in text_of_call(fake.messages_only[1])
    assert "ExpertPlus" in text_of_call(fake.messages_only[1])

    results = load_json(out / "results.json")
    assert results["status"] == "ok"
    assert not any("map ends at" in err for err in results["errors"])
    assert results["final"] is not None
    assert results["subscores"] and all("details" in s for s in results["subscores"])

    zip_path = out / "track.zip"
    with zipfile.ZipFile(zip_path) as z:
        zipped_info = json.loads(
            z.read([n for n in z.namelist() if n.endswith("Info.dat")][0]))
    assert zipped_info["levelAuthorName"] == "m1"

    with zipfile.ZipFile(out / "track.zip") as z:
        names = z.namelist()
        assert "Artist - Test/Info.dat" in names
        assert "Artist - Test/ExpertPlus.dat" in names
        assert "Artist - Test/song.ogg" in names
        expected_info = {**INFO, "levelAuthorName": "m1"}
        assert load_json_text(z, "Artist - Test/Info.dat") == expected_info
        assert load_json_text(z, "Artist - Test/ExpertPlus.dat") == DIFF


def load_json_text(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))


def test_partial_status_when_one_difficulty_fails(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF),
                                            GARBAGE, GARBAGE]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1",
                                  meta(["ExpertPlus", "Expert"]),
                                  {"weights": DEFAULT_WEIGHTS})

    fake = FakeClient.last_instance
    assert len(fake.calls) == 4
    assert len(raw_calls(out)) == 4
    assert load_json(out / "results.json")["status"] == "partial"

    with zipfile.ZipFile(out / "track.zip") as z:
        assert "Artist - Test/ExpertPlus.dat" in z.namelist()


def test_failed_info_stops_before_scoring(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([GARBAGE, GARBAGE]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    assert len(raw_calls(out)) == 2
    assert not (out / "results.json").exists()
    assert not (out / "track.zip").exists()
    assert load_json(out / "status.json")["status"] == "failed_info"


def test_cover_discovered_and_packed(tmp_path, monkeypatch):
    wav = tmp_path / "track.wav"
    sf.write(wav, np.zeros(2000, dtype="float32"), 44100, format="WAV")
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\xff\xd8fakejpeg")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))

    out = pipeline.evaluate_track(wav, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    fake = FakeClient.last_instance
    assert '"image_url"' in json.dumps(fake.messages_only[0])
    with zipfile.ZipFile(out / "track.zip") as z:
        assert "Artist - Test/cover.jpg" in z.namelist()
        assert z.read("Artist - Test/cover.jpg") == b"\xff\xd8fakejpeg"


def test_partial_errors_accumulated_in_results(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF),
                                            GARBAGE, GARBAGE]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1",
                                  meta(["ExpertPlus", "Expert"]),
                                  {"weights": DEFAULT_WEIGHTS})

    results = load_json(out / "results.json")
    assert results["status"] == "partial"
    assert any("Expert" in err and "ExpertPlus.dat" not in err
               for err in results["errors"])


def test_full_coverage_skips_continuation(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    fake = FakeClient.last_instance
    assert len(fake.calls) == 2
    assert [load_json(c)["kind"] for c in raw_calls(out)] == ["info", "difficulty"]
    assert load_json(out / "results.json")["status"] == "ok"


def test_partial_map_gets_continuation_and_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wav = write_wav(tmp_path, 6.0)
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(),
                                            json.dumps(make_partial()),
                                            json.dumps(TAIL_FRAGMENT)]))
    out = pipeline.evaluate_track(wav, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    fake = FakeClient.last_instance
    assert len(fake.calls) == 3
    kinds = [load_json(c)["kind"] for c in raw_calls(out)]
    assert kinds == ["info", "difficulty", "difficulty_continue"]

    cont_text = text_of_call(fake.messages_only[2])
    assert "ends at beat 8.5" in cont_text
    assert "Continue both notes and lighting through the remaining musical sections" in cont_text
    assert "continuing from beat 8.5" in cont_text

    with zipfile.ZipFile(out / "track.zip") as z:
        merged = load_json_text(z, "Artist - Test/ExpertPlus.dat")
    beats = [n["b"] for n in merged["colorNotes"]]
    assert beats == sorted(beats) and max(beats) == 14
    seam = [n for n in merged["colorNotes"]
            if (n["b"], n["x"], n["y"], n["c"]) == (8.5, 2, 0, 1)]
    assert len(seam) == 1

    results = load_json(out / "results.json")
    assert results["status"] == "ok"


def test_continuation_merges_arcs_and_multiblocks():
    base = {"colorNotes": [{"b": 0, "x": 0, "y": 0, "c": 0, "d": 1}],
            "sliders": [], "burstSliders": [], "basicBeatmapEvents": []}
    merged = pipeline._merge_fragment(base, V3_ELEMENT_FRAGMENT)
    assert merged["sliders"] == V3_ELEMENT_FRAGMENT["sliders"]
    assert merged["burstSliders"] == V3_ELEMENT_FRAGMENT["burstSliders"]
    assert pipeline._coverage_fraction(merged, 120.0, 8.0) == 1.0


def test_continuation_garbage_yields_warning_but_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wav = write_wav(tmp_path, 6.0)
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(),
                                            json.dumps(make_partial()),
                                            GARBAGE]))
    out = pipeline.evaluate_track(wav, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    fake = FakeClient.last_instance
    assert len(fake.calls) == 4
    kinds = [load_json(c)["kind"] for c in raw_calls(out)]
    assert kinds == ["info", "difficulty",
                     "difficulty_continue", "difficulty_continue"]

    results = load_json(out / "results.json")
    assert results["status"] == "ok"
    assert any(err.startswith("map ends at") for err in results["errors"])
    assert (out / "track.zip").exists()


def test_build_zip_tolerates_missing_keys(tmp_path):
    from beatbench.export.packager import build_zip
    from beatbench.types import GeneratedPackage

    audio = tmp_path / "track.wav"
    sf.write(audio, np.zeros(1000, dtype="float32"), 44100, format="WAV")
    pkg = GeneratedPackage(
        info={"songName": "NoAuthor"},
        difficulties={"Easy.dat": DIFF},
        audio_path=audio,
        cover_path=None,
    )
    zip_path = build_zip(pkg, tmp_path, "track")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert "Unknown Artist - NoAuthor/Info.dat" in names
        assert "Unknown Artist - NoAuthor/song.ogg" in names


def test_invalid_gate_runs_scoring_but_no_zip(env, monkeypatch):
    bad_diff = {**DIFF, "colorNotes": [{"b": 0}]}
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(bad_diff)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    results = load_json(out / "results.json")
    assert results["status"] == "invalid"
    assert results["gate_passed"] is False
    assert results["errors"]
    assert not (out / "track.zip").exists()


class SequenceClient:
    last_instance = None

    def __init__(self, endpoint, key, model, steps):
        self.steps = list(steps)
        self.calls = []
        SequenceClient.last_instance = self

    @classmethod
    def factory(cls, steps):
        def make(endpoint, key, model):
            return cls(endpoint, key, model, steps)
        return make

    def chat(self, messages, response_format=None, reasoning_effort=None):
        self.calls.append((messages, response_format))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return ChatReply(content=step,
                         usage={"prompt_tokens": 1, "completion_tokens": 1},
                         elapsed_s=0.01)


def make_api_error(status_code=None):
    import httpx2
    from openai import APIError, APIStatusError

    request = httpx2.Request("POST", "http://ep/v1/chat/completions")
    if status_code is None:
        return APIError("connection reset", request=request, body=None)
    response = httpx2.Response(
        status_code, request=request, json={"error": {"message": "rate limited"}})
    return APIStatusError("rate limited", response=response, body=None)


def test_api_error_on_info_call_writes_failed_api(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        SequenceClient.factory([make_api_error(429)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    status = load_json(out / "status.json")
    assert status["status"] == "failed_api"
    assert "429" in status["errors"][0]
    assert "rate limited" in status["errors"][0]
    assert not (out / "results.json").exists()
    assert not (out / "track.zip").exists()

    calls = raw_calls(out)
    assert len(calls) == 1
    payload = load_json(calls[0])
    assert payload["kind"] == "info"
    assert payload["reply_content"].startswith("API ERROR:")
    assert "rate limited" in payload["reply_content"]


def test_api_error_without_http_code_writes_failed_api(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        SequenceClient.factory([info_reply(), make_api_error()]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    status = load_json(out / "status.json")
    assert status["status"] == "failed_api"
    assert "connection reset" in status["errors"][0]
    calls = raw_calls(out)
    assert [load_json(c)["kind"] for c in calls] == ["info", "difficulty"]
    assert load_json(calls[1])["reply_content"].startswith("API ERROR:")
    assert not (out / "results.json").exists()


def test_scoring_exception_persists_status_and_keeps_generation(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))

    def boom(*args, **kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(pipeline, "run_scoring", boom)
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    status = load_json(out / "status.json")
    assert status["status"] == "scoring_failed"
    assert "validator exploded" in status["errors"][0]
    assert len(raw_calls(out)) == 2
    assert not (out / "track.zip").exists()


def test_raw_cleared_on_rerun(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})
    (out / "raw" / "call_99.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                            {"weights": DEFAULT_WEIGHTS})

    names = sorted(p.name for p in raw_calls(out))
    assert names == ["call_1.json", "call_2.json"]


def test_raw_logs_capture_reasoning_field_and_markdown_file(env, monkeypatch):
    thinking_info = ChatReply(
        content=info_reply(),
        usage={"prompt_tokens": 5, "completion_tokens": 7},
        elapsed_s=0.02,
        reasoning="deliberating about colors",
    )
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([thinking_info, json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    calls = raw_calls(out)
    assert [load_json(c)["kind"] for c in calls] == ["info", "difficulty"]
    first = load_json(calls[0])
    assert first["reasoning"] == "deliberating about colors"
    md_path = out / "raw" / "call_1.reasoning.md"
    assert md_path.exists()
    assert md_path.read_text(encoding="utf-8") == "deliberating about colors"

    second = load_json(calls[1])
    assert second["reasoning"] == ""
    assert not (out / "raw" / "call_2.reasoning.md").exists()


def test_gate_validated_once_and_errors_not_duplicated(env, monkeypatch):
    bad_diff = {**DIFF, "colorNotes": [{"b": 0}]}
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(bad_diff)]))

    calls = []
    real_check_schema = pipeline.check_schema

    def counting(pkg):
        calls.append(pkg)
        return real_check_schema(pkg)

    monkeypatch.setattr(pipeline, "check_schema", counting)
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    assert len(calls) == 1
    results = load_json(out / "results.json")
    assert results["status"] == "invalid"
    for err in set(results["errors"]):
        assert results["errors"].count(err) == 1
    assert results["song"] == "track"


def test_results_song_is_audio_stem(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})
    assert load_json(out / "results.json")["song"] == "track"


def test_sanitize_folder_replaces_windows_illegal_chars():
    from beatbench.export.packager import sanitize_folder
    assert sanitize_folder('Bad<>:"/\\|?*Name') == "Bad_________Name"
    assert sanitize_folder("trailing dots... ") == "trailing dots"
    assert sanitize_folder("control\x01char") == "control_char"
    assert sanitize_folder("...") == "_"


def test_build_zip_sanitizes_folder_name(tmp_path):
    from beatbench.export.packager import build_zip
    from beatbench.types import GeneratedPackage

    audio = tmp_path / "track.wav"
    sf.write(audio, np.zeros(1000, dtype="float32"), 44100, format="WAV")
    pkg = GeneratedPackage(
        info={"songName": 'Weird/Name?', "coverImageFilename": "cover.jpg"},
        difficulties={"Easy.dat": DIFF},
        audio_path=audio,
        cover_path=None,
    )
    zip_path = build_zip(pkg, tmp_path, "track")

    with zipfile.ZipFile(zip_path) as z:
        assert "Unknown Artist - Weird_Name_/Info.dat" in z.namelist()


def test_v2_flag_builds_second_zip(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS}, v2=True)

    assert (out / "track.zip").exists()
    assert (out / "track-v2.zip").exists()

    with zipfile.ZipFile(out / "track-v2.zip") as z:
        names = z.namelist()
        assert "Artist - Test/Info.dat" in names
        assert "Artist - Test/ExpertPlus.dat" in names
        assert "Artist - Test/song.ogg" in names
        info = load_json_text(z, "Artist - Test/Info.dat")
        assert "_version" in info
        diff = load_json_text(z, "Artist - Test/ExpertPlus.dat")
        assert "_notes" in diff


def test_top_level_saber_colors_hoisted(env, monkeypatch):
    reply = json.dumps({"info": dict(INFO), "saberColors": SABER_COLORS})
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([reply, json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    with zipfile.ZipFile(out / "track.zip") as z:
        assert "Artist - Test/saber_colors.json" in z.namelist()
        sidecar = json.loads(z.read("Artist - Test/saber_colors.json")
                             .decode("utf-8"))
    assert sidecar == SABER_COLORS


def test_v2_zip_injects_saber_colors(env, monkeypatch):
    info = {**INFO, "saberColors": {"left": [255, 80, 80],
                                    "right": [40, 120, 255]}}
    monkeypatch.setattr(
        pipeline, "ModelClient",
        FakeClient.factory([json.dumps({"info": info}), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS}, v2=True)

    with zipfile.ZipFile(out / "track-v2.zip") as z:
        exported = load_json_text(z, "Artist - Test/Info.dat")
    profile = exported["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    cd = profile["_customData"]
    assert cd["_colorLeft"] == pytest.approx({"r": 1.0, "g": 80 / 255, "b": 80 / 255, "a": 1.0})
    assert cd["_colorRight"] == pytest.approx({"r": 40 / 255, "g": 120 / 255, "b": 1.0, "a": 1.0})
    assert exported["_colorSchemes"][profile["_beatmapColorSchemeIdx"]]["useOverride"] is True


def test_no_v2_zip_without_flag(env, monkeypatch):
    monkeypatch.setattr(pipeline, "ModelClient",
                        FakeClient.factory([info_reply(), json.dumps(DIFF)]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    assert (out / "track.zip").exists()
    assert not (out / "track-v2.zip").exists()


V3_INFO_WITH_CUSTOM_BEATMAP_NAME = {
    "songName": "T",
    "songAuthorName": "A",
    "songFilename": "song.ogg",
    "coverImageFilename": "cover.png",
    "difficultyBeatmapSets": [{
        "_beatmapCharacteristicName": "Standard",
        "_difficultyBeatmaps": [{
            "_difficulty": "ExpertPlus",
            "_difficultyRank": 9,
            "_beatmapFilename": "CustomName.dat",
            "_noteJumpMovementSpeed": 18,
            "_noteJumpStartBeatOffset": 0,
        }],
    }],
}


def make_pkg_with_info(info, difficulties, tmp_path):
    from beatbench.types import GeneratedPackage

    audio = tmp_path / "track.wav"
    sf.write(audio, np.zeros(1000, dtype="float32"), 44100, format="WAV")
    return GeneratedPackage(info=info, difficulties=difficulties,
                            audio_path=audio, cover_path=None)


def test_build_v2_zip_reads_beatmap_filename_from_info(tmp_path):
    from beatbench.export.v2 import build_v2_zip

    pkg = make_pkg_with_info(V3_INFO_WITH_CUSTOM_BEATMAP_NAME,
                             {"ExpertPlus.dat": DIFF}, tmp_path)

    zip_path = build_v2_zip(pkg, tmp_path, "track")

    with zipfile.ZipFile(zip_path) as z:
        assert "A - T/CustomName.dat" in z.namelist()
        assert load_json_text(z, "A - T/CustomName.dat")["_version"] == "2.1.0"


def test_build_v2_zip_converts_custom_named_payload(tmp_path):
    from beatbench.export.v2 import build_v2_zip

    pkg = make_pkg_with_info(V3_INFO_WITH_CUSTOM_BEATMAP_NAME,
                             {"CustomName.dat": DIFF}, tmp_path)

    zip_path = build_v2_zip(pkg, tmp_path, "track")

    with zipfile.ZipFile(zip_path) as z:
        assert "A - T/CustomName.dat" in z.namelist()
        assert "_notes" in load_json_text(z, "A - T/CustomName.dat")


def test_build_v2_zip_falls_back_to_difficulty_dat_when_key_missing(tmp_path):
    from beatbench.export.v2 import build_v2_zip

    info = json.loads(json.dumps(V3_INFO_WITH_CUSTOM_BEATMAP_NAME))
    del info["difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_beatmapFilename"]
    pkg = make_pkg_with_info(info, {"ExpertPlus.dat": DIFF}, tmp_path)

    zip_path = build_v2_zip(pkg, tmp_path, "track")

    with zipfile.ZipFile(zip_path) as z:
        assert "A - T/ExpertPlus.dat" in z.namelist()
        assert "_notes" in load_json_text(z, "A - T/ExpertPlus.dat")


def test_load_config_returns_weight_copy(tmp_path):
    from beatbench.config import DEFAULT_WEIGHTS, load_config

    cfg = load_config(cwd=tmp_path)
    cfg["weights"]["playability"] = 999.0
    assert DEFAULT_WEIGHTS["playability"] == 0.30

    (tmp_path / "beatbench.toml").write_text(
        '[scoring.weights]\nplayability = 0.5\n', encoding="utf-8")
    cfg2 = load_config(cwd=tmp_path)
    assert cfg2["weights"]["sync"] == DEFAULT_WEIGHTS["sync"]
    cfg2["weights"]["density"] = 0.0
    assert DEFAULT_WEIGHTS["density"] == 0.17

