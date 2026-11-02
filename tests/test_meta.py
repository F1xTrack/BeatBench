import yaml
from pathlib import Path
from beatbench.meta import load_meta


def test_sidecar_and_overrides(tmp_path: Path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    (tmp_path / "song.yaml").write_text(yaml.safe_dump(
        {"title": "T", "author": "A", "bpm": 174, "difficulties": ["ExpertPlus", "expert"]}))
    meta = load_meta(audio, {})
    assert meta.title == "T" and meta.bpm == 174.0
    assert meta.difficulties == ["ExpertPlus", "Expert"]

    meta2 = load_meta(audio, {"title": "Override"})
    assert meta2.title == "Override"


def test_defaults_without_sidecar(tmp_path: Path):
    audio = tmp_path / "cool_track.ogg"
    audio.write_bytes(b"x")
    meta = load_meta(audio, {})
    assert meta.title == "cool_track" and meta.author == "Unknown"
    assert meta.difficulties == ["ExpertPlus"]


def test_nps_sidecar_and_override(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    (tmp_path / "song.yaml").write_text("nps: 2.5\n")
    assert load_meta(audio, {}).nps == 2.5
    assert load_meta(audio, {"nps": 4.0}).nps == 4.0
    assert load_meta(audio, {"nps": None}).nps == 2.5
