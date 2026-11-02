# Experimental scoring v5.0

v5.0 replaces the experimental v4.1 aggregate for new CLI runs and offline reviews.
It is a disclosed, model-blind proxy, not a calibrated human quality rating;
`quality_score` remains null. Existing run files and prior score reviews remain
historical records. Re-score into a fresh directory with `beatbench review-map`.

## What the score measures

| Axis | Weight | Evidence |
|---|---:|---|
| Musicality | .25 | Fixed full-spectrum attack proximity, chance correction, interval response and repeated-attack penalty |
| Movement | .15 | Per-hand transitions, fast returns, ambiguous turns and estimated planar recovery |
| Pattern development | .25 | Four-second audio/map structure correspondence and rigid-pattern mismatch |
| Density response | .15 | Notes per four-second section compared with audio level and spectral-attack counts, corrected against 64 fixed-seed permutations |
| Readability | .05 | Spawn time, conservative face-note candidates, duplicate cells and local density jumps |
| Presentation | .15 | Authored standard-light coverage and attack alignment, with a .35 floor |

The aggregate is the weighted geometric mean. Missing essential evidence produces
null; weights are not renormalized. Multiple difficulties remain separate, and
the package score uses the minimum assessed difficulty.

## Changes from v4.1

- Removed the `.39` hard caps for sparse audible-window occupancy, repeated
  patterns, and fast-return candidates. Those conditions are represented in the
  relevant axes rather than overriding the total with a single heuristic.
- Removed presentation from a post-score multiplier and gave the lighting proxy
  a disclosed 15% aggregate weight. A `.35` floor stops absent authored standard
  events from forcing the geometric aggregate to zero.
- Added `density_response`: section note counts should broadly rise and fall with
  both audio level and detected spectral attacks. This does not reward a target
  NPS or penalize every quiet section as silence. NPS tolerance remains a
  separate requirement.
- Reduced movement and readability weights because their current offline
  implementations are rough physical/visibility proxies. Musical timing and
  section-level responsiveness now carry more weight.
- Added scene-palette pair contrast as an unscored diagnostic. RGB contrast alone
  cannot establish whether the palette fits the song or renders well in game.

## Limits

Four-second level and attack counts do not recognize lyrics, instruments,
semantic phrases, drops, or intentional rests. Pattern development and density
response are related coarse signals and may partly double-count section changes.
The standard-light proxy does not render fades, brightness, custom event boxes,
environment geometry, or perceived visibility. Theme fit, obstacles, arcs/chains,
long-note path quality, and actual hand comfort still need rendered/playtest
review. The palette contrast diagnostic is not part of the score.

This scale has not been fitted to blind player ratings. Its weights are a
transparent first pass based on the observed benchmark failure modes; scores
should be read with the axis evidence and source-map hashes, not as universal
skill ratings. No score term uses model identity, reasoning traces, token use,
input mode, endpoint coverage, or exact target-NPS attainment.
