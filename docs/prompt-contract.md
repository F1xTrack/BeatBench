# Shared mapping prompt

The mapping prompt is shared by the `Info.dat`, difficulty and continuation
requests. Keeping one versioned prompt prevents a model from receiving
different mapping rules at different stages of the same run.

## Requirements covered by the prompt

- Beats versus seconds, grid coordinates, hand colors and cut directions;
- an audible musical layer, first-beat alignment, rhythm and phrase changes;
- forehand/backhand flow, recovery, readability and comfortable movement;
- ordinary musical lighting, event semantics and environment compatibility;
- whole-track NPS counting both hands, local variation and intentional pauses;
- complete output for the requested difficulty;
- optional V3 arcs (`sliders`), burst sliders (`burstSliders`), Chroma and
  Noodle Extensions without treating their presence as a quality bonus;
- model-selected saber and scene colors exported through native V3 fields.

The prompt does not provide a reference map, precomputed note times, a hidden
track identity or a track-specific color answer. A model must infer musical
structure from the supplied input representation. JSON snippets in the prompt
demonstrate syntax only and never define the scope of the output.

## Versioning and comparison

Each manifest records the prompt version and SHA-256 values for the system and
template text. Compare participants only within the same prompt version and
input condition. A prompt change, input fix, transport change or model setting
change starts a new experiment.

Continuation calls must preserve the existing hand flow, musical sections,
lighting intent and map format. They are repairs or extensions of one map, not
permission to replace its pattern with a new unrelated style.

## Interpreting prompt experiments

Prompt-only comparisons require identical audio, cover, metadata, requested
density, difficulty, timeouts and output constraints. Store raw provider calls
and private evidence outside the public repository; publish only sanitized
summaries and reproducible configuration. A higher automated proxy does not
prove a better map, and a successful schema gate does not prove that the map is
playable or visually effective.

The event and cut-direction terminology follows the public
[Beat Saber map format](https://bsmg.wiki/mapping/map-format/beatmap.html) and
[lightshow format](https://bsmg.wiki/mapping/map-format/lightshow.html).
