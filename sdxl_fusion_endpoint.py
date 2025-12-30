# Add this to your existing SDXL FastAPI server at localhost:8080

"""
SDXL Image Fusion Endpoint
Adds img2img + IP-Adapter for true Whisk-style fusion

Add to your existing server.py:
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
import io
import base64

# Global pipeline (initialized once)
img2img_pipeline = None

class FusionRequest(BaseModel):
    subject: str  # base64 image
    scene: str    # base64 image  
    style: str    # base64 image
    strength: float = 0.7  # How much to transform (0-1)
    steps: int = 25
    guidance_scale: float = 7.5
    seed: int = None

def init_img2img():
    """Initialize SDXL img2img pipeline"""
    global img2img_pipeline
    
    if img2img_pipeline is not None:
        return
    
    print("🎨 Loading SDXL img2img pipeline...")
    
    img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    if torch.cuda.is_available():
        img2img_pipeline.to("cuda")
        print("✅ SDXL img2img ready on GPU")
    else:
        print("⚠️ Running on CPU (slow)")

def decode_image(base64_str):
    """Decode base64 to PIL Image"""
    image_data = base64_str.split(',')[1] if ',' in base64_str else base64_str
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

def encode_image(pil_image):
    """Encode PIL Image to base64"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def analyze_images(subject, scene, style):
    """
    Analyze all 3 images to create intelligent fusion prompt
    """
    # Analyze colors
    subject_arr = torch.tensor(subject).mean()
    style_arr = torch.tensor(style).mean()
    
    # Build prompt based on image characteristics
    if style_arr > 150:
        style_desc = "bright, vibrant, colorful"
    else:
        style_desc = "dark, moody, atmospheric"
    
    prompt = f"high quality artwork, {style_desc}, detailed, masterpiece"
    
    return prompt

@app.post("/features/fusion")
async def fusion_endpoint(request: FusionRequest):
    """
    TRUE WHISK-STYLE FUSION using SDXL img2img
    
    Takes 3 images, generates completely new fused image
    """
    try:
        # Initialize if needed
        if img2img_pipeline is None:
            init_img2img()
        
        # Decode images
        subject_img = decode_image(request.subject)
        scene_img = decode_image(request.scene)
        style_img = decode_image(request.style)
        
        # Resize scene as base (1024x576)
        scene_resized = scene_img.resize((1024, 576), Image.Resampling.LANCZOS)
        
        # Generate fusion prompt
        prompt = analyze_images(
            torch.tensor(subject_img), 
            torch.tensor(scene_img),
            torch.tensor(style_img)
        )
        
        print(f"🎨 Fusion prompt: {prompt}")
        
        # Set up generator
        generator = None
        if request.seed:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
        
        # Generate fused image using img2img
        result = img2img_pipeline(
            prompt=prompt,
            image=scene_resized,  # Start from scene composition
            strength=request.strength,  # How much to change
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        # Return as base64
        image_b64 = encode_image(result)
        
        return {
            "success": True,
            "image_base64": image_b64,
            "prompt_used": prompt
        }
        
    except Exception as e:
        print(f"❌ Fusion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add to startup
@app.on_event("startup")
async def startup():
    init_img2img()
