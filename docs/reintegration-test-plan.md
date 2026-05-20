# Forge Reintegration Test Plan

This plan keeps the live static Forge page as the source of truth. Apprentice is reference material only until a cleaner replacement contract is designed and proven.

## Local Static Layout

1. From `D:\0___TESTZONE\_theneurofoundry\import_scripts\Forge`, run:

   ```powershell
   python -m http.server 8008 --bind 127.0.0.1
   ```

2. Open `http://127.0.0.1:8008/`.
3. Verify these viewport sizes:
   - Desktop: `1440x1000`
   - Tablet: `900x1024`
   - Mobile: `390x844`
4. Confirm:
   - Desktop keeps the live Forge composition.
   - Mobile has no document-level horizontal scroll.
   - Mobile panels stack in order: Origin, Fusion, Synthesize, Recent Crafts.
   - Forge button remains reachable.
   - Recent Crafts scrolls horizontally on mobile.

## Live Endpoint Smoke Checks

These checks should run before any backend reshaping.

```powershell
Invoke-WebRequest https://extract.theneurofoundry.com/health -UseBasicParsing
Invoke-WebRequest https://nfcr-synthesize.onrender.com/features/4/images -UseBasicParsing
```

For authenticated Worker routes, use the configured local key outside git and test:

- `https://prmptrndr.csirico9.workers.dev/fusion/cdp`
- `https://prmptrndr.csirico9.workers.dev/fusion/acp`
- `https://prmptrndr.csirico9.workers.dev/fusion/sip`
- `https://synth.csirico9.workers.dev`

Expected result: reachable services return either healthy JSON or a clear auth/validation error, not DNS/TLS/network failure.

## Live Page Workflow

1. Origin path:
   - Upload one image in Origin.
   - Confirm it displays in the center viewer.
   - Confirm Fusion disables or resets as currently designed.

2. Fusion path:
   - Upload Subject and Scene.
   - Confirm subject extraction starts.
   - Confirm CDP/ACP/SIP status messages progress.
   - Confirm Fusion reaches `Fusion ready` or shows a clear failure status.

3. Prompt path:
   - Enter Positive Prompt and Scene Prompt.
   - Confirm field checks activate.
   - Confirm Forge readiness updates.

4. Render path:
   - Click Forge.
   - Confirm loading popup appears.
   - Confirm successful renders populate Recent Crafts.
   - On failure, confirm the UI exits loading state and reports the error.

## Apprentice Reference Policy

Do not cut the live page over to Apprentice as-is.

Reusable ideas to inspect later:

- Endpoint grouping under one Worker.
- Health and diagnostics endpoints.
- Prompt/style schema concepts.
- Model routing options.

Rewrite any useful Apprentice behavior into a cleaner contract after the live page is stable.
