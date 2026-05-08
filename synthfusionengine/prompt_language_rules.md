# Prompt Language Rules (Fragment-First)

## Principle

Long prose does not reliably improve synthesis quality. Prefer compact, sloted fragments built from visible evidence.

## Core Lanes

- `cdp`: character detail prompt
- `acp`: atmosphere and context prompt
- `sip`: style and image properties prompt

## Slot Model

Default order:

1. `character`
2. `objects`
3. `scene`
4. `atmosphere`
5. `camera`
6. `style`

Target SDXL budget:

- around `75` tokens total unless profile overrides

Default slot budget:

- character: 20
- objects: 12
- scene: 12
- atmosphere: 14
- camera: 7
- style: 10

## Composition Format

Compact bracket style:

`[character fragments] [object fragments] [scene fragments] [atmosphere fragments] [camera fragments] [style fragments]`

Example:

`[black mask, gloves, heavy boots] [hammer, damaged plating, oil spray] [open battlefield, broken metal debris] [ember smoke, windy haze] [low angle, medium shot] [cinematic fantasy realism, gritty contrast]`

## Selection Rules

- Prefer observed facts over inferred facts.
- Keep high-confidence fragments first.
- Drop contradictory fragments.
- Preserve character-lock fragments when enabled.
- Maintain minimum atmosphere/camera budget when constrained.

## Anti-Drift Rules

- Never confirm unsupported assumptions.
- No invented identities/brands.
- If uncertain, mark uncertainty rather than inventing detail.
- If model output is sparse, apply deterministic fallback composition.

## RNG Semantics

`Resynthesize` is RNG behavior only (seed/strength randomization), not style compilation.

