import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .judges.human import SCORE_KEYS, collect_ratings
from .meta import load_meta
from .runner.pipeline import evaluate_track
from .scoring.review_io import review_source

app = typer.Typer(help="BeatBench: benchmark multimodal LLMs on BeatSaber mapping.")
console = Console()


@app.command()
def evaluate(
    audio: str,
    endpoint: str = typer.Option("https://openrouter.ai/api/v1"),
    key: str = typer.Option(None),
    model: str = typer.Option(...),
    title: str = typer.Option(None),
    author: str = typer.Option(None),
    bpm: float = typer.Option(None),
    nps: float = typer.Option(None, help="Target average notes-per-second density"),
    difficulty: list[str] = typer.Option(None, "--difficulty", "-d"),
    constrained: bool = typer.Option(False, "--constrained"),
    v2: bool = typer.Option(False, "--v2"),
    reasoning_effort: str = typer.Option(None, "--reasoning-effort",
                                         help="Extended thinking effort for models that support it (e.g. high)"),
    audio_fullrate: bool = typer.Option(False, "--audio-fullrate",
                                        help="Send audio at original quality (44.1 kHz stereo mp3) instead of 16 kHz mono"),
    input_mode: str = typer.Option("audio", help="audio, spectrogram, or python; common modes use fresh experiment directories"),
    backend: str = typer.Option("api", help="api, codex (subscription CLI), or local (loopback llama.cpp)"),
    sandbox_distro: str = typer.Option("Ubuntu-24.04", help="WSL distribution for the Python sandbox"),
    model_call_timeout: float = typer.Option(300, help="Seconds per model request in audio and common modes"),
    reasoning_summary: str = typer.Option("detailed", help="Codex summary: auto, concise, detailed, none; not full internal CoT"),
):
    """Generate, score and package a full map for one track."""
    resolved_key = key or os.environ.get("BEATBENCH_KEY")
    if input_mode not in ("audio", "spectrogram", "python") or backend not in ("api", "codex", "local"):
        raise typer.BadParameter("choose --input-mode audio/spectrogram/python and --backend api/codex/local")
    if backend in ("codex", "local") and input_mode == "audio":
        raise typer.BadParameter("Codex/local require --input-mode spectrogram or python")
    if backend == "api" and not resolved_key:
        console.print("[red]No API key: pass --key or set the BEATBENCH_KEY environment variable.[/red]")
        raise typer.Exit(code=2)
    audio_path = Path(audio)
    meta = load_meta(audio_path, {
        "title": title,
        "author": author,
        "bpm": bpm,
        "nps": nps,
        "difficulties": list(difficulty) if difficulty else None,
    })
    cfg = load_config()
    mode_options = {}
    if input_mode != "audio" or backend != "api":
        mode_options = {"input_mode": input_mode, "backend": backend, "sandbox_distro": sandbox_distro}
    if model_call_timeout != 300:
        mode_options["model_call_timeout"] = model_call_timeout
    if backend == "codex":
        mode_options["reasoning_summary"] = reasoning_summary
    out_dir = evaluate_track(audio_path, endpoint, resolved_key or "", model, meta, cfg,
                             constrained=constrained, v2=v2,
                             reasoning_effort=reasoning_effort,
                             audio_fullrate=audio_fullrate, **mode_options)

    data = _load_status(out_dir)
    final = data.get("experimental_score")
    table = Table(title="BeatBench evaluation")
    table.add_column("Model")
    table.add_column("Song")
    table.add_column("Status")
    table.add_column("Experimental v6.0 proxy / 100")
    table.add_row(
        str(data.get("model", model)),
        str(data.get("song", audio_path.stem)),
        str(data.get("status", "unknown")),
        f"{final:.1f}/100" if isinstance(final, (int, float)) else "-",
    )
    console.print(table)
    console.print("Experimental axis proxies require playtesting/calibration. Status 'ok' means format checks passed.")
    console.print(f"Results: {out_dir}")

    if v2:
        v2_path = out_dir / f"{audio_path.stem}-v2.zip"
        if v2_path.exists():
            console.print(f"V2 zip: {v2_path}")
        else:
            console.print("[yellow]v2 export failed; see log warnings[/yellow]")

    if data.get("status") != "ok":
        raise typer.Exit(code=1)


@app.command(name="analyze-reasoning")
def analyze_reasoning_command(source: Path, output_root: Path = typer.Option(None)):
    """Index exposed summaries, code and actions from a saved run without model calls."""
    from .runner.reasoning import analyze_reasoning
    if not (source / "raw").is_dir():
        raise typer.BadParameter("Provide a run directory containing raw/")
    analysis, path = analyze_reasoning(source, output_root)
    console.print(f"Reasoning summaries: {analysis['capture_status']}; {analysis['summary_items']} items, "
                  f"{analysis['summary_characters']} characters. Python actions: {len(analysis['observable_python_actions'])}.")
    console.print(f"Analysis: {path}")


@app.command(name="judge-human")
def judge_human(results_dir: str):
    """Collect absolute 0-100 player ratings for the primary map ZIP in a run directory."""
    answers = {}
    labels = {
        "musicality": "Musical timing, accents and note choice",
        "phrase_response": "Response to sections, changes and recurring phrases",
        "density_and_rest": "Whether notes and pauses feel intentional; no filler for NPS",
        "playability": "Pattern quality, hand comfort and readability",
        "presentation": "Lighting, colors and environment in the headset",
    }
    for key in SCORE_KEYS:
        while True:
            try:
                value = typer.prompt(f"{labels[key]} (0-100)", type=int)
            except ValueError:
                console.print("[red]Enter an integer between 0 and 100.[/red]")
                continue
            if 0 <= value <= 100:
                answers[key] = value
                break
            console.print("[red]Enter an integer between 0 and 100.[/red]")
    comment = typer.prompt("Comment", default="", show_default=False)

    out_path = collect_ratings(Path(results_dir), {**answers, "comment": comment})
    data = json.loads(out_path.read_text(encoding="utf-8"))
    console.print(f"Player-rated quality: [bold]{data['overall_score_100']:.1f}/100[/bold]")
    if data.get("source_binding_status") != "bound_to_primary_zip":
        console.print("[yellow]Rating saved, but no unique primary ZIP was found; it will not attach to a map review.[/yellow]")


@app.command(name="review-map")
def review_map(
    source: Path = typer.Argument(..., help="Existing V2/V3 ZIP, Info.dat, map or run directory"),
    output_root: Path = typer.Option(Path("results/scoring-reviews"), help="Fresh review directories; source files are read only"),
    target_nps: float = typer.Option(None, help="Optional density requirement, not a quality rating"),
    json_only: bool = typer.Option(False, help="Inspect only map JSON; no audio feature analysis"),
):
    """Review existing maps offline; no model calls or legacy result changes."""
    try:
        review, path = review_source(source, output_root, target_nps, analyze_audio=not json_only)
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Experimental v6.0 proxy / 100 + diagnostics")
    for label in ("Difficulty", "Score", "Notes", "NPS", "Return candidates L/R", "Lighting authored", "Issues"):
        table.add_column(label)
    for name, data in review["difficulty_reviews"].items():
        if not data.get("assessed"):
            table.add_row(name, "-", "-", "-", "-", "unknown", data.get("reason", "unassessed"))
            continue
        flow = data["flow"]
        nps = data.get("structure", {}).get("actual_nps")
        score = review["assessment"]["difficulty_assessments"][name].get("experimental_score")
        table.add_row(name, f"{score:.1f}" if score is not None else "unrated", str(data["note_count"]), f"{nps:.2f}" if nps is not None else "unknown",
                      f"{flow['0']['short_gap_return_candidates']}/{flow['1']['short_gap_return_candidates']}"
                      if all(v["timebase_assessed"] for v in flow.values()) else "unknown",
                      "yes" if data["lighting"]["authored_standard_lighting_present"] else "absent",
                      ", ".join(issue["code"] for issue in data["issues"]))
    console.print(table)
    player_rating = review.get("player_rating")
    if player_rating:
        console.print(f"Player-rated quality: [bold]{player_rating['overall_score_100']:.1f}/100[/bold] "
                      f"({player_rating['rating_count']} rating(s), {player_rating['calibration_status']})")
    else:
        console.print("Player-rated quality: unrated. Experimental axis values are diagnostics, not absolute quality scores.")
    console.print(f"Review: {path}")


def _load_status(out_dir: Path) -> dict:
    paths = [out_dir / name for name in ("status.json", "results.json")]
    existing = [path for path in paths if path.exists()]
    if existing:
        path = max(existing, key=lambda candidate: candidate.stat().st_mtime_ns)
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


if __name__ == "__main__":
    app()
