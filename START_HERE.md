# Forge Quick Start

Forge's current public UI is `index.html`, served at:

```text
https://forge.theneurofoundry.com/
```

## Current Live Flow

- Origin path: upload one image, caption it through `nfcr-synthesize`, then render from the generated prompt.
- Fusion path: upload subject and scene images, extract/caption/analyze them through the configured external services, then render from the synthesized prompt.
- Recent Crafts: keep up to 10 local thumbnails in browser storage, with optional cloud save/load for authenticated users.

## Local Static UI Check

From this folder:

```powershell
python -m http.server 4179 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:4179/index.html
```

## Subject Extraction Service

Run this only when testing the extraction service directly:

```powershell
python subject_extractor.py
```

Local endpoint:

```text
http://127.0.0.1:5001/health
```

Production Forge uses:

```text
https://extract.theneurofoundry.com
```

## Legacy Reference Files

The older ControlNet/composition files remain reference material unless explicitly revived:

- `controlnet_service.py`
- `controlnet_integration.py`
- `START_ALL_SERVICES.bat`
- `START_EVERYTHING.bat`
- `start_controlnet.bat`
- `sdxl_fusion_endpoint.py`
