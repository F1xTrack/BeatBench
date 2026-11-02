# Scoring v6.0: 0–100 indicators and player ratings

v6.0 puts every automated indicator and its aggregate on a 0–100 display scale.
This is a scale change for the existing offline evidence; it does **not** turn the
automated proxy into a human quality rating. `quality_score` remains null until
players rate maps against the absolute rubric below. The CLI labels automated
results as proxies. Prior scoring files and reports are historical and unchanged.

## Absolute quality rubric

Rate each axis from 0 to 100 after playing the map. Suggested anchor points are
0 (broken or absent), 25 (poor), 50 (weak but partly playable), 75 (strong), and
100 (exceptional reference quality). Use intermediate values when the map falls
between anchors.

| Player-rated axis | Weight | What to judge |
|---|---:|---|
| Musicality | .25 | Timing, accent choice, and whether notes represent the audible music |
| Phrase response | .25 | Changes across sections, coherent motifs, and response to fills, drops, vocals, and rests |
| Density and rests | .15 | Whether each note is justified; whether quiet phrases and empty space are respected |
| Playability | .20 | Hand flow, pattern quality, readability, and sustained comfort |
| Presentation | .15 | Lighting, colors, and environment as they appear in game |

Use these axis-specific anchors; do not infer one axis from another:

- **Musicality:** 0 = timing and note choices mostly unrelated to audible events;
  25 = occasional hits land, but accents/sound layers are often wrong; 50 = basic
  beat alignment with uneven emphasis; 75 = notes usually follow the intended
  accents and musical parts; 100 = note-by-note choices consistently express the
  audible parts, accents, and timing. A detector hit within 50 ms is evidence,
  not a human musicality score.
- **Phrase response:** 0 = one repeated pattern regardless of song changes;
  25 = changes are mostly ignored or arbitrary; 50 = some sections are reflected,
  but motifs and transitions are inconsistent; 75 = coherent motifs evolve with
  sections, fills, vocals, and rests; 100 = changes, callbacks, and deliberate
  pauses consistently express the song's structure. Variety alone earns no credit.
- **Density and rests:** 0 = obvious silence/rests are filled or large active
  passages are left empty; 25 = frequent forced notes or missed phrases; 50 = broad
  density is plausible but local note economy is uneven; 75 = notes and empty
  space mostly match musical activity; 100 = every note feels justified and
  rests/dynamic changes are deliberately respected. Hitting target NPS earns no
  points.
- **Playability:** 0 = broken or repeatedly unsafe/unreadable; 25 = frequent
  awkward returns, conflicting hands, or tiring patterns; 50 = playable but often
  repetitive or uncomfortable; 75 = coherent, readable, varied hand flow; 100 = a
  consistently comfortable, intentional challenge. Automated planar estimates do
  not substitute for playing the map.
- **Presentation:** 0 = absent, broken, or visually confusing; 25 = sparse or
  mostly unrelated effects; 50 = functional but weakly connected to the music;
  75 = coordinated lighting, colors, and environment support musical sections;
  100 = a clear visual identity and well-timed show enrich the whole song. Event
  counts and RGB values alone do not earn points.

The absolute interpretation for an overall score is: 0–19 unusable, 20–39 poor,
40–59 weak but partly playable, 60–69 good, 70–84 very good, 85–94 excellent,
and 95–100 reference-level. These anchors follow the user's stated judgement of
the current card sets; they remain provisional until blind repeat ratings.

The overall player score is the weighted mean, with disclosed non-compensation
ceilings: musicality or phrase response at 20 or below caps the total at 39; two
or more core axes below 40 cap it at 49. NPS target compliance is recorded
separately and never earns quality points. Missing evidence is `unrated`, never
zero.

## Automated evidence

The six offline axes from v5 remain model-blind, but each `score` now uses the
0–100 range and also records its `raw_proxy_0_1`. The weighted geometric aggregate
is shown as `experimental_score` on the same range. It is still a mathematical
proxy, not the absolute quality rubric. In particular, timing detection,
four-second section correlations, planar recovery estimates, and standard-light
coverage cannot determine semantic phrasing, deliberate note economy, actual
comfort, or rendered visual quality.

Use `beatbench judge-human <experiment-run-directory>` after playing a map to record
the five player axes. Ratings are saved under a new, timestamped
`human-ratings/` directory and do not overwrite existing ratings. The CLI does
not display the model name; use anonymous map IDs during review to reduce
expectation bias. When the directory contains exactly one primary ZIP, each
rating stores its SHA-256. `review-map` attaches only ratings whose hash matches
the reviewed ZIP, averages repeated ratings, and exposes that value as
`quality_score`; stale or unbound ratings are ignored. A single rating is marked
unvalidated; repeated blind ratings are needed before treating the scale as a
reliable benchmark.

The prior v5 result archive remains available for reproducibility. Rescore maps
into a fresh directory when comparing the automated v6 proxy; never overwrite a
prior review set.
