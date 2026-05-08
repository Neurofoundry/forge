# SynthFusion Engine Schema Pack

This folder is the working source of truth for prompt-language schemas and routing contracts for Forge synth/fusion.

## Files

- `synthfusion_contracts.json`: Endpoint contracts, request/response schemas, routing logic, token budgeting.
- `style_profiles.json`: Style profile definitions, slot weights, defaults, and negative baselines.
- `prompt_language_rules.md`: Language method for fragment-first prompting and compact composition.
- `examples.json`: Practical payload and output examples.

## Core Direction

- Focus on image-first decomposition.
- Keep three analysis lanes (`cdp`, `acp`, `sip`) and fuse late.
- Favor compact structured fragments over long prose.
- Target around 75 tokens for SDXL variants unless profile overrides.
- Keep `Resynthesize` semantics as RNG only (seed/strength randomization), not style compile.

