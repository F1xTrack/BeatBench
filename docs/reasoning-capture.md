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

For a controlled series, run each condition sequentially with the same prompt
version, input mode, timeout and requested difficulty. Record the provider,
model, reasoning setting and cache mode in the manifest. A cache hit is a
transport optimization, not evidence of better mapping quality; changing the
prefix or model can invalidate reuse. Preserve failed conditions and never
rewrite earlier raw calls or score files.
