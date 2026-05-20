# Legacy Neuroforge Feature Pocket

These old Neuroforge V2 feature files are reference material for the Forge reshape. They are not copied into the live runtime because they include old worker URLs, local model paths, and backend assumptions that should not be treated as production configuration.

## Source Files

- `D:\0___TESTZONE\Neuroforge - V2\features\feature_04_textimage.py`
- `D:\0___TESTZONE\Neuroforge - V2\features\feature_14_recalibrate.py`

## Useful Ideas To Reuse

### Feature 04: Text/Image

- FastAPI slot for `/features/4/render`, `/features/4/caption`, `/features/4/video`, and `/features/4/images`.
- Caption-first flow: image input becomes a caption, then caption becomes the render prompt.
- Model-to-worker routing for base, lightning, img2img, and inpainting variants.
- Local placeholder fallback when remote rendering fails.

### Feature 14: Recalibrate

- Recipe object for subject, scene, style, weights, SD parameters, outputs, and optional persisted image payloads.
- Multi-stage pipeline shape: subject extraction, scene understanding, composition, harmonization, style transfer, variation.
- Late prompt refresh from captions plus optional user hint.
- Useful endpoint shape: create/update recipe, then generate one or more recalibrated outputs.

## Reshape Notes

- Replace hard-coded keys and worker URLs with environment variables or Worker bindings.
- Treat U2Net, MiDaS, BLIP, Torch, and OpenCV paths as optional local/sidecar capabilities, not baseline static-site dependencies.
- Prefer folding the recipe/slot model into the Apprentice Worker contract before adding another backend service.
