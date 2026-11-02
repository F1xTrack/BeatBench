# Input modes

BeatBench supports three model-context modes. They are evaluated separately;
keep the track, metadata, requested density, difficulty, timeout and scoring
configuration fixed when comparing models.

## Audio

The model receives the audio representation supported by the selected API
client. Audio longer than 240 seconds is truncated for the model context. This
is the simplest mode, but providers can apply different audio preprocessing.

## Spectrogram

The harness sends one overview image and sequential four-second panels. The
STFT settings, image dimensions and SHA-256 values are recorded in the run
manifest. No onset list, section labels or reference-map note times are sent.

```powershell
uv run beatbench evaluate path\to\track.ogg `
  --backend api --endpoint https://api.example.com/v1 `
  --key "$env:BEATBENCH_KEY" --model MODEL_NAME `
  --input-mode spectrogram --nps 2.5 --difficulty ExpertPlus
```

## Python analysis

The model can request signal analysis through the sandboxed `python(code)`
tool. The sandbox receives a read-only PCM WAV and cover image, with bounded
CPU, wall time, memory, output and request counts. It cannot access the host
repository, API keys, previous runs or the network.

```powershell
uv run beatbench evaluate path\to\track.ogg `
  --backend local --endpoint http://127.0.0.1:8097/v1 `
  --model MODEL_NAME --input-mode python --nps 2.5 `
  --difficulty ExpertPlus
```

The local backend is optional and is intended for an OpenAI-compatible loopback
server. The public repository does not include model weights or a server.

## Backend and evidence

`api` uses an OpenAI-compatible HTTP endpoint, `codex` uses an authenticated
Codex CLI, and `local` uses a loopback endpoint. API keys are read from a
command argument or environment variable and are never written to manifests.

Each run is written under `results/experiments/<mode>/<model>/<track>/<id>/`.
It may contain the manifest, generated map, diagnostic result, raw provider
responses and reasoning summaries. Treat that directory as private evidence:
do not commit it, and redact credentials before sharing an excerpt.

The backend, model identifier, prompt version, representation settings,
source hashes and timeout are recorded so that later comparisons can reject
accidental condition changes. A successful `ok` status means the format and
pipeline gates passed; it is not a gameplay-quality verdict.
