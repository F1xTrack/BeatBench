# Reasoning capture v2

Codex participants explicitly request `model_reasoning_summary="detailed"` and
`model_supports_reasoning_summaries=true`, with the requested effort and model.
The model catalog's default summary may be `none`, independently of reasoning
effort. The participant prompt and permitted tools are unchanged.

Codex/OpenAI exposes summaries rather than full internal reasoning:
[reasoning summaries](https://developers.openai.com/api/docs/guides/reasoning),
[Codex settings](https://learn.chatgpt.com/docs/config-file/config-reference).
No raw-CoT extraction or fabricated reconstruction is performed. Summaries may
be brief or unavailable even when reasoning-token counters are nonzero.

Every CLI call retains original stdout JSONL and stderr, plus a `.reasoning.json`
record containing requested settings, thread ID, availability, source hash and
summary item IDs/line numbers. Nonempty summaries also get a Markdown copy.
File-byte SHA256 and normalized-stdout SHA256 are separate: Windows may translate
newlines on write. Earlier v1 records used the latter under the source-hash key;
fresh analysis recomputes the actual source hash and flags discrepancies without
rewriting the old records.
Updated/completed revisions of an item are deduplicated by ID. Timeout output is
preserved, including any partial summary; errors do not destroy preceding data.
CLI sessions remain ephemeral: exported evidence is sufficient for this path.

OpenRouter API runs send `reasoning.effort` and preserve returned reasoning fields,
per-call token usage, provider identity and cost in `raw/call_N.json`. The analyzer
indexes these provider-exposed traces and their topic mentions. Providers may omit,
summarize or transform reasoning fields; the capture is not assumed to be a complete
internal CoT. The final report should summarize themes and decisions instead of
reproducing long private reasoning verbatim.

`beatbench analyze-reasoning RUN_DIRECTORY` writes a fresh timestamped analysis.
Common-mode Codex and API generation also invoke it on completion/failure. It includes
exposed reasoning text, reproducible topic mentions, observed Python code/results,
failures, map lighting/palette observations and the saved scoring assessment.
Observable Python action records include stderr, elapsed time and truncation flags,
so exceptions with an empty error field can still be inspected from their traceback.
It also records capture settings per call, malformed/truncated input, and counts
text outside standalone Markdown headings. Heading-only summaries are explicitly
visible; this formatting count is not a rating of reasoning depth.
Topic mentions are not claims of understanding or successful execution. Capture
failure and unavailable text are explicit, including for historical Astra runs.
Analysis performs no model calls, does not influence scoring, and is never fed
back into the participant conversation. No judge-model expenses are introduced.

Debug all four conditions through the subscription CLI:

```powershell
.venv\Scripts\python.exe tools/run_input_pilots.py --backend codex --codex-model gpt-6-luna --reasoning-effort high --reasoning-summary detailed --repeats 1 --model-timeout 900
```

For a later Astra series, change only `--codex-model gpt-6-astra`. Freeze prompt,
input conditions and generation settings before claiming a controlled comparison.

## Sequential model series and cache accounting

One launcher invocation processes four conditions per selected model (BB/YPM,
spectrogram/Python). Models, conditions and all underlying metadata/difficulty/tool
requests execute sequentially. Repeating `--codex-model` queues another model.

```powershell
.venv\Scripts\python.exe tools/run_input_pilots.py --backend codex --codex-model gpt-6-luna --codex-model gpt-6.1-sol --reasoning-effort xhigh --reasoning-summary detailed --codex-cache-prefix --repeats 1 --model-timeout 900
```

`--codex-cache-prefix` enables transport `codex-prefix-v2`. It keeps one isolated
temporary working directory throughout the launcher series and sends the unchanged
benchmark system instructions as a stable developer prefix. The Python contract
and changing per-track transcript follow that prefix. Individual calls remain
fresh ephemeral sessions; no previous map, summary or evaluation is supplied to
another condition. A workspace lock rejects overlapping requests, and the native
Windows executable is preferred over a Node wrapper to avoid an orphan on timeout.

Each call saves `.transport.json` with prefix hash, workspace and image hashes.
Offline analysis and `series.json` include provider-reported input, cached input,
uncached input and cached fraction. When the CLI exposes cache-write and reasoning
token counters, fresh analysis preserves them; missing usage stays explicit.
Cache reuse is best effort and requires a
matching rendered prefix; a different model or the parent Codex conversation
cannot be assumed to share it. See [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).

The parent mapping prompt stays P3, but v2 changes its placement in the transport.
Comparisons with saved v1 runs also differ in transport; comparisons between xhigh
and high runs differ in effort. Record both differences and avoid attributing
an observed score change solely to model size or effort.

`tools/compare_scoring_v4.py --series SERIES.json` accepts the entire series,
preserves failed conditions, and adds fresh reasoning/cache analysis alongside
map assessments. It never rewrites source maps or their earlier scores.
