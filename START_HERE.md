# 🎨 AI Image Fusion - Quick Start Guide

## What You Need

**1 Service Only:**
- `subject_extractor.py` (Port 5001)
- Your SDXL proxy (Port 8080) - should already be running

## Quick Start

### Step 1: Start Service
```bash
# Double-click:
start_extractor.bat

# OR command line:
python subject_extractor.py
```

### Step 2: Test Fusion
Open in browser: **`test_forgeintegration.html`**

Go to **Fusion Panel (#2)** and:
1. Upload **Subject** image (background auto-removes)
2. Upload **Scene** image
3. Upload **Style** image
4. Click **"⚡ Compose Images"**

## How It Works

```
1. Subject → rembg removes BG → SDXL img2img → Result A
2. Scene → SDXL img2img → Result B
3. Style → SDXL img2img → Result C
4. Layer A + B + C together → FINAL IMAGE
```

**Each image gets AI-processed through your Cloudflare Worker!**

## Files You Need

### Core Files:
- ✅ `subject_extractor.py` - Main service
- ✅ `requirements.txt` - Dependencies
- ✅ `start_extractor.bat` - Quick launcher
- ✅ `test_forgeintegration.html` - Your main UI

### Optional Test Files:
- `test_composition_pipeline.html` - Standalone test UI
- `test_extraction.html` - Background removal test

### Ignore These (old test files):
- ❌ `controlnet_service.py` - Not needed anymore
- ❌ `controlnet_integration.py` - Not needed
- ❌ `START_ALL_SERVICES.bat` - Old
- ❌ `START_EVERYTHING.bat` - Old
- ❌ `start_controlnet.bat` - Not needed
- ❌ `sdxl_fusion_endpoint.py` - Reference only

## Troubleshooting

**"Composition failed"?**
- Make sure `localhost:8080` (SDXL proxy) is running
- Check the terminal logs for detailed error

**"Service not starting"?**
- Run: `pip install -r requirements.txt`
- Check if port 5001 is already in use

**Slow performance?**
- Normal! 3 img2img passes take ~60-90 seconds total
- Each pass is ~20-30 seconds through Cloudflare Worker

## For Production

When deploying online:
1. Deploy `subject_extractor.py` to your server
2. Update Cloudflare Worker URL in the code (currently localhost:8080)
3. Done! Frontend calls your deployed endpoint

---

**Need help?** Check the detailed logs in the terminal window.
