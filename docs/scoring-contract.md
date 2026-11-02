# Evaluation contract: diagnostic v3

## Experimental v6.0 implementation

New results now also include a separate, explicitly uncalibrated experimental
assessment and the CLI displays that proxy on a 0–100 scale. See
[scoring-v6.md](scoring-v6.md) for the player rubric, weights and limitations.
The v3 diagnostic contract
and null calibrated quality rating remain in effect; legacy numbers are not
promoted back to quality ratings.

## Why the previous number is not a quality rating

The legacy weighted average lets independent bonuses compensate for fundamental
gameplay failures. A map can receive large contributions from format validity,
endpoint coverage and a simplistic duplicate-cell check before musicality is
considered. Those signals cannot establish that a map is enjoyable.

The legacy reset detector compares adjacent notes across both hands and ignores
cut direction. Other defects include density windows that depend on declared
BPM, endpoint-only coverage, accidental onset matches and treating mod presence
as mod quality. These old metrics remain reproducibility data only.

The existing v2 proposal is not promoted: a revised weighted sum still permits
compensation, a bottom audio percentile is not silence, and arbitrary event
diversity or NJS ranges cannot establish quality.

## Output contract

New generation results explicitly label `final` as `legacy-v1`,
`score_kind=uncalibrated_heuristic`. It is retained for compatibility. The CLI
calls it **Legacy proxy (uncalibrated)**. `quality_score=null` and
`calibration_status` prevents the automated proxy from being presented as a
calibrated gameplay rating. A map remains unrated until a player score is bound
to its exact ZIP hash; a single player rating is still marked unvalidated.
`status=ok` describes the pipeline/format gate, not musical or gameplay quality.

`diagnostics` contains `diagnostic-v3.0`, per-file observations and issue examples.
There is no automatic averaging across difficulties or bonus for mod support.
Existing results, analytics reports and source maps are not rewritten.

```powershell
beatbench review-map "path\to\map.zip" --target-nps 2.5
beatbench review-map "path\to\map.zip" --json-only
```

The offline command accepts V2/V3 Standard maps, including human maps. Every
review goes into a new `results/scoring-reviews/<UTC>-<id>/review.json` with source
hashes. V2 is normalized in memory, not rejected as a poor map because of its
format. Other characteristics and V4 require future support and are not rated.
No model calls or API spending are needed.

`--json-only` skips audio analysis explicitly and records that scope. It cannot
establish musical timing. Lighting diagnostics also expose event value counts
and declared native/per-difficulty scene palettes. A valid RGB pair is
authoring evidence, not a theme-fit score; color selection by light events must
also be inspected.

## Six experimental axes and a separate validity gate

| Axis | Evidence in experimental v6 | Needed for a quality judgement |
| --- | --- | --- |
| Musicality | Full-spectrum attacks, timing proximity, chance baseline and interval response | Listen for selected sound layer, offset, phrasing and accents |
| Movement | Independent hand transitions, fast returns, ambiguous turns and planar recovery proxy | Playtest setup/recovery, parity, bombs, walls, comfort and speed |
| Pattern development | Four-second audio/map structure agreement and rigidity mismatch | Check whether motifs and changes follow musical sections |
| Density response | Four-second note counts compared with audio level and attack counts | Inspect rests and intentional low-density phrases in context |
| Readability | NJS/offset metadata, face-note candidates, duplicates and local density jumps | Playtest visibility, reaction time and accumulated fatigue |
| Presentation | Authored standard-light coverage and attack alignment; declared palette contrast is diagnostic only | Observe visibility, brightness, palette/theme fit and environment in game |

JSON/package correctness and source integrity are prerequisites. A format pass
is never converted into free quality points. Target NPS is a requested density
constraint, not a reward for making constant-density maps. Chroma/Noodle are
capabilities; neither is required for good normal lighting or gameplay.

Lighting is part of the complete experience. A black map cannot be described as
well presented. Event presence alone cannot establish that the lights are
visible or good. Strong lighting cannot compensate for bad movement or timing;
high overall quality must require acceptable results on every core axis.

## Diagnostic limits

- Same-hand direction differences of 0/45 degrees suggest return motion;
  90 degrees needs context, and 135/180 degrees supports reversal. A short-gap
  candidate uses a one-second diagnostic window, not a validated fail threshold.
  Intentional resets, bombs, speed and setup require review. See
  [BSMG parity guidance](https://bsmg.wiki/mapping/basic-mapping.html#parity).
- Dots, rotated/animated geometry and explicit arcs/chains can leave transitions
  unassessed. This is not a physics simulation or full collision/visibility check.
- Repetition is not intrinsically bad. A repeated musical section can justify a
  repeated pattern; variability is not rewarded merely for being variable. See
  [BSMG emphasis and consistency guidance](https://bsmg.wiki/mapping/basic-mapping.html).
- Spectral attacks are imperfect evidence, particularly for vocals and sustained
  instruments. Shifted schedules estimate accidental proximity, not human
  judgement. No winning band is silently selected to inflate a rating.
- Audibility uses a conservative RMS threshold, not the quietest quartile. An
  occupied four-second window measures coarse activity, not completeness or
  quality. Pauses and undermapping may be intentional.
- Audio timing/structure is limited to the first 240 seconds; full-track NPS uses
  the actual full duration. Official V3 BPM events are integrated into note
  timestamps. Nonzero song offsets and legacy editor BPM changes currently make
  audio assessment unavailable; short-gap hand counts are also unassessed in
  those cases. Unsupported data never earns a perfect score.
- Light counts are authoring evidence. Custom lighting and environment/game
  settings can change visible lighting. Missing standard lighting is a warning,
  not proof that every possible environment is black.

## Calibration before publishing a new overall number

1. Freeze a versioned collection with multiple songs, musical styles and density
   levels. Include human maps, these four Luna pilots and genuine BeatSage and
   InfernoSaber outputs. Record generator/version/settings and hashes. Baselines
   must be generated from the same audio cut; do not compare a human longer mix
   or another difficulty as if conditions matched. Do not use ratings or maps as
   hints in model generation.
2. Play blind randomized pairs on the same headset, with the same environment
   settings, modifiers and audio latency. Record each axis, overall preference,
   ties, inability to finish, problematic timestamps and confidence. Keep repeat
   sessions and different players identifiable without revealing model labels.
3. Anchor the human scale with played examples: 0 = broken/unusable, around
   0.3–0.4 = poor with frequent movement/musical failures, around 0.7 = consistently
   good and worth playing, 1 = an excellent reference. These are proposed anchors,
   not scores assigned to today's artifacts. The user's rejection of all four
   pilots is qualitative playtest evidence; the stated ceiling is not four
   fabricated numeric ratings.
4. Validate candidate automated metrics against these labels and pairwise
   preferences. Use held-out songs and players, uncertainty intervals and axis
   disagreement checks. Include failure cases: one note at the end, alternating
   hands with repeated downstrokes, clockwork patterns over changing music,
   shifted timing, absent lights, many meaningless lights and good deliberate
   repetition. Human V2 maps are positive references, not format negatives.
5. Choose a calibrated non-compensating overall rule only after this evidence.
   Learn any weights/axis floors from human data; do not lower a weight or impose
   an arbitrary cap merely to force today's maps below 0.4. Publish the four axes
   and confidence alongside an overall rating. Missing essential evidence means
   unrated, never weight renormalization that inflates the remaining metrics.

Until this protocol is completed, automated output is diagnostic and overall
quality remains unrated. The current implementation does not claim a calibrated
quality model, a successful new playtest, or a measured win over either baseline.
