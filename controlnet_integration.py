# ControlNet Integration for SDXL
# Add this to your existing FastAPI server at localhost:8080

"""
INSTALLATION:
pip install diffusers transformers accelerate controlnet_aux opencv-python torch torchvision

MODELS NEEDED (auto-download on first run):
- diffusers/controlnet-canny-sdxl-1.0
- stabilityai/stable-diffusion-xl-base-1.0
"""

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import cv2
import torch
import io
import base64

# Global pipeline (loaded once)
controlnet_pipeline = None

def initialize_controlnet():
    """Initialize ControlNet + SDXL pipeline (call once on startup)"""
    global controlnet_pipeline
    
    if controlnet_pipeline is not None:
        return
    
    print("🎨 Loading ControlNet + SDXL models...")
    
    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    
    # Create pipeline
    controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16
    )
    
    # Enable optimizations
    controlnet_pipeline.enable_model_cpu_offload()
    
    print("✅ ControlNet ready!")

def get_canny_image(image, low_threshold=100, high_threshold=200):
    """Extract canny edges from image for ControlNet"""
    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB PIL image
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    canny_image = Image.fromarray(edges_rgb)
    
    return canny_image

def controlnet_transform(
    input_image,
    prompt,
    negative_prompt="",
    style_reference=None,
    controlnet_conditioning_scale=0.7,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=None
):
    """
    Transform an image using ControlNet to preserve structure
    
    Args:
        input_image: PIL Image or base64 string
        prompt: Text description of desired output
        negative_prompt: What to avoid
        style_reference: Optional PIL Image to match style
        controlnet_conditioning_scale: How much to follow structure (0-1)
        num_inference_steps: Quality (more = better but slower)
        guidance_scale: How much to follow prompt
        seed: Random seed for reproducibility
    """
    global controlnet_pipeline
    
    # Initialize if needed
    if controlnet_pipeline is None:
        initialize_controlnet()
    
    # Decode input if base64
    if isinstance(input_image, str):
        input_image = decode_base64_image(input_image)
    
    # Resize to SDXL dimensions (must be multiple of 8)
    width, height = input_image.size
    width = (width // 8) * 8
    height = (height // 8) * 8
    input_image = input_image.resize((width, height))
    
    # Get canny edges for structure preservation
    canny_image = get_canny_image(input_image)
    
    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate with ControlNet
    result = controlnet_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    return result

def decode_base64_image(data_str):
    """Decode base64 image string to PIL Image"""
    image_data = data_str.split(',')[1] if ',' in data_str else data_str
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

def encode_image_to_base64(image):
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


# ==========================================
# FASTAPI ENDPOINT (add to your server)
# ==========================================

"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ControlNetRequest(BaseModel):
    input_image: str  # base64
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.7  # How much to preserve structure
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int = None

@app.post("/controlnet/transform")
async def controlnet_transform_endpoint(request: ControlNetRequest):
    try:
        result_image = controlnet_transform(
            input_image=request.input_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            controlnet_conditioning_scale=request.strength,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        return {
            "success": True,
            "image": encode_image_to_base64(result_image)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_controlnet()
"""
