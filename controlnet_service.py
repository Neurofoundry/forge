"""
ControlNet + SDXL Service
Port 5002 (to avoid conflict with subject_extractor on 5001)

Transforms images while preserving composition:
- Anime → Photorealistic
- Photo → Anime/Cartoon
- Change art style while keeping pose/layout
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global pipeline
pipeline = None

def initialize_pipeline():
    """Load models on startup"""
    global pipeline
    
    if pipeline is not None:
        return
    
    logger.info("🎨 Loading ControlNet + SDXL models (this may take a few minutes)...")
    
    # Load ControlNet for Canny edge detection
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    
    # Load optimized VAE
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    
    # Create pipeline
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16
    )
    
    # Optimize for speed
    if torch.cuda.is_available():
        # Move entire pipeline to GPU (faster than CPU offload)
        pipeline.to("cuda")
        # Enable memory-efficient attention for speed
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("✅ Using GPU with xformers optimization")
    else:
        logger.warning("⚠️  No GPU found, using CPU (will be VERY slow)")
    
    logger.info("✅ ControlNet pipeline ready!")

def get_canny_edges(image, low_threshold=100, high_threshold=200):
    """Extract structure/edges from image"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def decode_base64_image(data_str):
    """Decode base64 to PIL Image"""
    image_data = data_str.split(',')[1] if ',' in data_str else data_str
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'controlnet-sdxl',
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/transform', methods=['POST'])
def transform_image():
    """
    Transform image style while preserving composition
    
    Request:
    {
        "image": "base64...",
        "prompt": "hyper realistic photo",
        "negative_prompt": "anime, cartoon",
        "strength": 0.7,  # How much to preserve structure (0-1)
        "steps": 30,
        "guidance": 7.5,
        "seed": 12345
    }
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        logger.info(f"🎨 Transform request: {data.get('prompt', '')[:50]}...")
        
        # Initialize pipeline if needed
        if pipeline is None:
            initialize_pipeline()
        
        # Decode input image
        input_image = decode_base64_image(data['image'])
        
        # Resize to valid SDXL dimensions (multiple of 8)
        width, height = input_image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Extract edges for structure preservation
        canny_image = get_canny_edges(input_image)
        
        # Get parameters
        prompt = data['prompt']
        negative_prompt = data.get('negative_prompt', 'blurry, low quality, distorted')
        controlnet_strength = data.get('strength', 0.7)
        num_steps = data.get('steps', 30)
        guidance_scale = data.get('guidance', 7.5)
        seed = data.get('seed', None)
        
        # Set up generator
        generator = None
        if seed is not None:
            if torch.cuda.is_available():
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed)
        
        logger.info(f"Generating with strength={controlnet_strength}, steps={num_steps}")
        
        # Generate NEW IMAGE (not just transform)
        # Lower strength = more creative freedom to redraw
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            controlnet_conditioning_scale=controlnet_strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            # Key parameters for fusion/redrawing:
            # Lower strength = AI redraws more freely
            # Higher guidance = follows prompt better
        ).images[0]
        
        # Encode result
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("✅ Transform complete!")
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'canny_edges': f'data:image/png;base64,{base64.b64encode(io.BytesIO(canny_image.tobytes()).getvalue()).decode()}'  # For debugging
        })
        
    except Exception as e:
        logger.error(f"Transform error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🎨 ControlNet + SDXL Transform Service")
    print("="*60)
    print("📍 Local:   http://localhost:5002")
    print("📍 Network: http://0.0.0.0:5002")
    print("="*60)
    print("💡 Endpoints:")
    print("   POST /transform  - Transform image style")
    print("   GET  /health     - Check service status")
    print("="*60)
    print("\n⏳ Initializing models on first request...")
    print("💾 Models will auto-download (~6GB) on first run\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
