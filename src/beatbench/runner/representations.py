"""Deterministic signal views shared by all participants in an input condition."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..config import MAX_AUDIO_SECONDS

SPECTROGRAM_SETTINGS = {
    "version": 2, "sample_rate": 44100, "n_fft": 1024, "hop_length": 128,
    "segment_seconds": 4.0, "width": 1400, "height": 480,
    "db_min": -80.0, "db_max": 0.0, "frequency_min": 40.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_part(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64,{data}", "detail": "high"}}


def stage_signal(audio: Path, directory: Path) -> tuple[Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    signal, sr = librosa.load(audio, sr=44100, mono=False, duration=MAX_AUDIO_SECONDS)
    if signal.size == 0:
        raise ValueError("audio is empty")
    canonical = directory / "audio.wav"
    sf.write(canonical, signal.T, sr, subtype="PCM_16")
    manifest = {
        "source_sha256": sha256(audio), "canonical_sha256": sha256(canonical),
        "duration_seconds": signal.shape[-1] / sr, "sample_rate": sr,
        "channels": 1 if signal.ndim == 1 else signal.shape[0],
        "max_audio_seconds": MAX_AUDIO_SECONDS, "encoding": "WAV PCM_16",
    }
    return canonical, manifest


def build_spectrograms(audio: Path, directory: Path) -> tuple[list[dict], dict]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    settings = dict(SPECTROGRAM_SETTINGS)
    signal, sr = librosa.load(audio, sr=settings["sample_rate"], mono=True)
    magnitude = np.abs(librosa.stft(signal, n_fft=settings["n_fft"],
                                    hop_length=settings["hop_length"]))
    db = librosa.amplitude_to_db(magnitude, ref=max(float(magnitude.max()), 1e-8))
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr,
                                   hop_length=settings["hop_length"])
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=settings["n_fft"])
    duration = len(signal) / sr
    directory.mkdir(parents=True, exist_ok=True)
    parts = [{"type": "text", "text": (
        "The following images are fixed spectrograms of the complete supplied signal. "
        "Horizontal axis: absolute seconds; vertical axis: frequency in Hz on a log scale; "
        "brightness: magnitude in dB relative to the maximum for the entire track. "
        "All panels use the same scale. No BPM, beat grid, detected onsets, section labels "
        "or proposed notes have been added. Infer music timing and structure yourself. "
        f"Signal duration: {duration:.6f} seconds. Overview is followed by consecutive "
        "4-second panels; panel images do not change the requested full-track generation."
    )}]
    images = []
    intervals = [("overview", 0.0, duration)] + [
        (f"segment_{i:03d}", float(start), min(float(start + 4), duration))
        for i, start in enumerate(np.arange(0, duration, settings["segment_seconds"]))
    ]
    for name, start, end in intervals:
        fig, ax = plt.subplots(figsize=(settings["width"] / 100, settings["height"] / 100), dpi=100)
        selected = np.flatnonzero((times >= start) & (times <= end))
        selected = selected[::max(1, int(np.ceil(len(selected) / settings["width"])))]
        ax.pcolormesh(times[selected], frequencies, db[:, selected], shading="auto", cmap="magma",
                      vmin=settings["db_min"], vmax=settings["db_max"], rasterized=True)
        ax.set_yscale("log")
        ax.set_ylim(settings["frequency_min"], sr / 2)
        ax.set_xlim(start, max(start + .001, end))
        ax.set_xlabel("Absolute time (seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{name}: {start:.3f}–{end:.3f} s | fixed scale −80…0 dB")
        if name != "overview":
            ax.set_xticks(np.arange(start, end + .0001, .2))
            ax.tick_params(axis="x", labelsize=9)
        fig.tight_layout()
        path = directory / f"{name}.png"
        fig.savefig(path, metadata={"Software": "BeatBench spectrogram v2"})
        plt.close(fig)
        parts.append(image_part(path))
        images.append({"file": path.name, "start_seconds": start,
                       "end_seconds": end, "sha256": sha256(path)})
    manifest = {"settings": settings, "images": images}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return parts, manifest
