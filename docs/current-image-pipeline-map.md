# Forge Current Image Pipeline Map

This reflects the current live Forge flow after the Origin captioning, Fusion routing, and Cloudflare CORS fixes.

## High-Level Flow

```mermaid
flowchart TD
  A[User opens Forge] --> B{Choose input path}

  B -->|Origin image| O1[Upload Origin image]
  O1 --> O2[Preview shown]
  O2 --> O3[Fusion input disabled]
  O3 --> O4[POST nfcr-synthesize /features/4/caption]
  O4 --> O5[Origin caption returned]
  O5 --> O6[Positive Prompt filled with originCaption]
  O6 --> R0[Forge button allowed when Origin caption + prompt checks are ready]

  B -->|Fusion images| F1[Upload Subject image]
  B -->|Fusion images| F2[Upload Scene image]
  F1 --> F3[POST extract.theneurofoundry.com rembg]
  F3 --> F4[Extracted subject image]
  F4 --> F5[POST prmptrndr /fusion/cdp]
  F5 --> F6[cdpContext subject-only caption]

  F2 --> F7[POST prmptrndr /fusion/acp]
  F7 --> F8[acpContext scene-only caption]

  F4 --> F9[Canvas merge subject over scene]
  F8 --> F9
  F9 --> F10[mergedImage preview]
  F10 --> F11[POST prmptrndr /fusion/sip]
  F11 --> F12[sipContext style/camera/palette caption]
  F12 --> F13[Scene Prompt filled with sipContext]
  F12 --> F14[Closest Style dropdown option selected]

  F6 --> F15[POST synth.csirico9.workers.dev]
  F8 --> F15
  F12 --> F15
  F15 --> F16[synthCaption compact final caption]
  F16 --> F17[Positive Prompt filled with synthCaption]
  F17 --> R0

  R0 --> R1[buildPrompt]
  R1 --> R2[basePrompt + selected Style + Positive Prompt + Scene Prompt]
  R2 --> R3[POST nfcr-synthesize /features/4/render method prompt]
  R3 --> R4[Render API forwards prompt to Cloudflare SDXL worker]
  R4 --> R5[PNG returned as image_base64 JSON]
  R5 --> R6[Forge displays output]
  R6 --> R7[Local recent cache updated]
  R6 --> R8[Optional cloud recent save POST api.theneurofoundry.com /api/forge/renders]
```

## Origin Path

Origin is now caption-first:

1. User uploads one Origin image.
2. Forge previews the image and disables Fusion inputs.
3. Forge calls `https://nfcr-synthesize.onrender.com/features/4/caption`.
4. Request includes:

```json
{
  "image_base64": "data:image/...",
  "instruction": "ORIGIN_CAPTION_INSTRUCTION"
}
```

5. Returned caption becomes `originCaption`.
6. `originCaption` fills Positive Prompt.
7. Forge render uses `originCaption` as `basePrompt`.

Origin does not send the original image into the final render call right now. It sends the caption/prompt only.

## Fusion Path

Fusion currently uses Subject and Scene. Style image input exists visually but remains locked.

1. Subject image goes through background extraction.
2. Extracted subject is captioned through CDP:

```text
POST https://prmptrndr.csirico9.workers.dev/fusion/cdp
```

3. Scene image is captioned through ACP:

```text
POST https://prmptrndr.csirico9.workers.dev/fusion/acp
```

4. Forge merges the extracted subject over the scene locally in a canvas.
5. The merged preview is captioned through SIP:

```text
POST https://prmptrndr.csirico9.workers.dev/fusion/sip
```

6. SIP output fills Scene Prompt.
7. SIP output also tries to select the closest Style dropdown option.
8. CDP, ACP, and SIP are sent to Synth:

```text
POST https://synth.csirico9.workers.dev
```

9. Synth returns `synthCaption`.
10. `synthCaption` fills Positive Prompt.

## Final Render Prompt

Both Origin and Fusion call `buildPrompt(basePrompt)` before rendering.

Prompt order:

```text
basePrompt, selected Style, Positive Prompt, Scene Prompt
```

Duplicate copies of `basePrompt` are skipped when Positive Prompt or Scene Prompt exactly match it.

Final render request:

```text
POST https://nfcr-synthesize.onrender.com/features/4/render
```

Request shape:

```json
{
  "method": "prompt",
  "prompt": "final built prompt",
  "negative_prompt": "negative prompt text",
  "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
  "width": 1024,
  "height": 576,
  "guidance_scale": 7.5,
  "num_inference_steps": 25
}
```

Response shape:

```json
{
  "ok": true,
  "image_base64": "...",
  "saved_as": "Features/images/...",
  "prompt": "...",
  "method": "prompt"
}
```

## Current Known Side Paths

- `api.theneurofoundry.com/api/forge/renders` is the authenticated cloud recent-render save path. It stores up to 10 renders per user through the Neurofoundry API/R2 route when auth and R2 are available.
- Local Recent Crafts still update using browser storage and generated thumbnails, including for logged-out visitors.
- Auto-download is disabled in the live UI. Render outputs display in the Forge result popup and Recent Crafts; authenticated cloud save is attempted separately.
- `shared/page-transitions.js` ignores `download`, `data:`, and `blob:` links so manual/generated download links are not intercepted by page transitions.
- Loading cog art is served from `assets/loading/c2.png`, `c3.png`, and `c4.png`.
