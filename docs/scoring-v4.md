# Experimental assessment v4.1

This implements model-blind offline proxies for the proposed four gameplay axes.
It is not a fitted or validated human quality scale. `quality_score` remains null,
`experimental_score` is explicit, and the CLI shows it as the primary proxy.
Legacy `final`/subscores stay labeled legacy-v1 for historical reproducibility.
Existing reports, run results and maps are not rewritten; offline rescoring uses
fresh directories with source hashes. During development, v4.0 pilot assessments
are provisional; rescore them with v4.1 to compare using one implementation.

| Axis | Weight | Evidence |
|---|---:|---|
| Musicality | .35 | Full-spectrum 50 ms attack proximity, chance-corrected support, matched interval sequence and reused-attack penalty |
| Movement | .30 | Independent hand swing groups, return speed, contextual bomb exclusion, ambiguous turns and planar recovery effort |
| Pattern development | .20 | Four-second audio/map descriptors, pairwise structural correspondence against 32 fixed-seed permutations, unchanged patterns across audio changes |
| Readability | .15 | Spawn time from NJS/BPM/offset, conservative face-note candidates, duplicate cells and local demand spikes |

The gameplay aggregate is `M^.35 * F^.30 * P^.20 * R^.15`. Missing essential
evidence produces null; weights are never renormalized. Multiple difficulties
are shown separately, with the minimum as the conservative package proxy.

Musicality uses `.45 * proximity + .30 * chance_corrected + .25 * interval_support`.
The full-spectrum layer stays fixed; band diagnostics are visible, and no best
band is selected separately per note. This is not instrument recognition or a
validated measure of emphasis. Quiet vocals and sustained instruments can be
missed; no automatic offset correction moves a mistimed map into alignment.

Movement groups compatible stacks/sliders, then checks each hand independently.
Fast-return burden uses `1-gap/.6` for supported 0/45-degree transitions; bombs
between swings exclude candidate resets. Ambiguous 90-degree turns receive a
small speed-dependent penalty, not a fail verdict. Recovery estimates previous
stroke endpoint to next start with half-cell strokes and a soft scale of six
cells/second. It does not model wrists, body motion or physical collision paths.

Pattern development compares normalized pairwise audio/map section differences.
It uses `.4 * agreement + .6 * correlation_excess_over_permutations`, then a
rigidity penalty only when high repetition coincides with changing audio and
unchanged map descriptors. Constant repetition over stable audio is allowed.
Random diversity is not a standalone reward. Coarse spectral features cannot
establish thematic phrase understanding; thresholds require held-out validation.

Readability reports actual half-jump times, without a universal good NJS range.
Vision candidates are conservative; a face note is not automatically unplayable.
Spikes use two-second counts and a threefold increase above eight local notes.
Reaction capacity, fatigue and actual occlusion remain unverified.

Provisional core caps are disclosed in output: widespread fast returns with
burden >.35, fewer than half of audible windows occupied, or severe rigidity
without timing support can cap the gameplay proxy at .39. These are engineering
hypotheses, not confirmed gameplay verdicts. Deliberate pauses and technical
patterns require context. Caps bound rather than multiply penalties repeatedly.

Lighting is separate. Standard on-state coverage and event/attack proximity form
an authored-show proxy; one event is capped at .2. No events means no authored
standard show, not proof of an actually black scene. Fade/flash durations and
environment rendering are not simulated. Event-box/custom lighting is unrated
until supported. `experience_proxy = G * (.8 + .2 * lighting_proxy)` is reported
only when both inputs exist. Neither event count, valid RGB, Chroma/Noodle nor
theme keywords earns direct points. Theme fit and actual visibility need review.

Target NPS is a separate requirement with a 25% tolerance, never a quality bonus.
Format/schema correctness is a separate gate. Endpoint coverage, bottom-quartile
"silence", constant local NPS and mod presence are removed from the primary
score. CoT/summaries, token counts and model identity have no scoring influence.

Unsupported offsets, legacy BPM changes, animated/rotated geometry, explicit
arcs/chains, missing audio and truncated audio prevent a complete automated
assessment. Dots may leave too few directional transitions. Unknowns do not
become zero-quality judgments or perfect scores.

Validation includes adversarial timing shifts, repeated downstrokes, fast returns,
one tail note, missing audio, unsupported geometry, mod flags, target changes,
lighting removal and BPM reencoding. Freeze these rules before the next model
series, then calibrate with blind played pairs and held-out songs/players.
Do not tune thresholds to force the desired Astra/Luna gap on today's examples.

```powershell
beatbench review-map MAP.zip
python tools/compare_scoring_v4.py RUN1 RUN2 --baseline-comparison BASELINE.json
```
