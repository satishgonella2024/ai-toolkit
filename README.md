# AI Toolkit for Ubuntu Workstation

A comprehensive collection of AI/ML scripts for your RTX 4080 SUPER workstation.

## System Requirements

- **OS**: Ubuntu 24.04 LTS
- **GPU**: NVIDIA RTX 4080 SUPER (16GB VRAM)
- **RAM**: 124 GB
- **CUDA**: 12.4+
- **Python**: 3.12

## Quick Start

```bash
# Activate the virtual environment
source ~/pytorch-env/bin/activate

# Run any script
python scripts/02_image_generation.py
```

## Available Capabilities

| # | Category | Script | Models | VRAM |
|---|----------|--------|--------|------|
| 01 | LLM Text Generation | `01_llm_text_generation.py` | Mistral-7B, Llama-3.1-8B, GPT-2 | 4-6 GB |
| 02 | Image Generation | `02_image_generation.py` | SDXL, SD3, FLUX.1, Playground v2.5 | 6.5 GB |
| 03 | Video Generation | `03_video_generation.py` | Stable Video Diffusion | CPU offload |
| 04 | Music Generation | `04_music_generation.py` | MusicGen Small/Medium | 2-7.5 GB |
| 05 | Speech-to-Text | `05_speech_to_text.py` | Whisper Large-v3 | 2.9 GB |
| 06 | Text-to-Speech | `06_text_to_speech.py` | Bark, Parler-TTS | 2-3.3 GB |
| 07 | Image Captioning | `07_image_captioning.py` | BLIP, LLaVA-v1.6 | 0.9-4.4 GB |
| 08 | Object Detection | `08_object_detection.py` | YOLOv8 (detect/segment/pose) | ~1 GB |
| 09 | Depth Estimation | `09_depth_estimation.py` | Depth Anything V2 | 0.63 GB |
| 10 | Segmentation | `10_segment_anything.py` | SAM (Segment Anything) | 2.45 GB |
| 11 | Inpainting & Background | `11_inpainting_background.py` | SD 1.5/SDXL Inpaint, rembg | 2-6.5 GB |
| 12 | Image Upscaling | `12_image_upscaling.py` | Swin2SR, SD x4 Upscaler | 0.05-1.6 GB |
| 13 | ControlNet & Img2Img | `13_controlnet_img2img.py` | SDXL ControlNet (Canny, Depth) | 8-9 GB |

## Script Details

### 01. LLM Text Generation

Generate text with large language models using 4-bit quantization.

```python
from scripts.01_llm_text_generation import load_model, generate_text

model, tokenizer = load_model("mistral")  # or "llama", "gpt2"
response = generate_text(model, tokenizer, "Explain quantum computing")
print(response)
```

### 02. Image Generation

Create images from text prompts.

```python
from scripts.02_image_generation import load_playground, generate_image_playground

pipe = load_playground()
generate_image_playground(pipe, "A dragon flying over mountains", output_name="dragon.png")
```

**Available models:**
- `load_sdxl_turbo()` - Fast (4 steps, 512x512)
- `load_sdxl()` - High quality (30 steps, 1024x1024)
- `load_playground()` - Aesthetic (30 steps, 1024x1024)
- `load_sd3()` - SD3 Medium (28 steps, 1024x1024) *requires HF approval*
- `load_flux()` - FLUX.1-schnell (4 steps, 512x512) *requires HF approval*

### 03. Video Generation

Generate short video clips from images.

```python
from scripts.03_video_generation import load_svd, generate_video

pipe = load_svd()
generate_video(pipe, "path/to/image.png", output_name="animated.mp4", motion_bucket_id=150)
```

### 04. Music Generation

Generate music from text descriptions.

```python
from scripts.04_music_generation import load_musicgen, generate_music

model, processor = load_musicgen("medium")
generate_music(model, processor, "epic orchestral music with drums", output_name="epic.wav")
```

### 05. Speech-to-Text

Transcribe audio files to text.

```python
from scripts.05_speech_to_text import load_whisper, transcribe

model, processor = load_whisper("large-v3")
text = transcribe(model, processor, "audio.wav", language="en")
print(text)
```

### 06. Text-to-Speech

Generate natural speech from text.

```python
# Bark (supports emotions like [laughs], [sighs])
from scripts.06_text_to_speech import load_bark, generate_speech_bark

model, processor = load_bark()
generate_speech_bark(model, processor, "Hello! [laughs] How are you?", output_name="greeting.wav")

# Parler-TTS (natural language style control)
from scripts.06_text_to_speech import load_parler, generate_speech_parler

model, tokenizer = load_parler()
generate_speech_parler(model, tokenizer, "Hello world!",
    description="A female speaker with a warm voice", output_name="hello.wav")
```

### 07. Image Captioning & Visual Q&A

Describe images or ask questions about them.

```python
# Fast captioning with BLIP
from scripts.07_image_captioning import load_blip, caption_blip

model, processor = load_blip()
caption = caption_blip(model, processor, "image.png")

# Detailed Q&A with LLaVA
from scripts.07_image_captioning import load_llava, ask_llava

model, processor = load_llava()
answer = ask_llava(model, processor, "image.png", "What is happening in this image?")
```

### 08. Object Detection

Detect, segment, and estimate poses.

```python
from scripts.08_object_detection import load_yolo, detect_objects, segment_objects, estimate_pose

# Object detection
model = load_yolo("detect", "x")
detections = detect_objects(model, "image.jpg")

# Instance segmentation
model = load_yolo("segment", "x")
segments = segment_objects(model, "image.jpg")

# Pose estimation
model = load_yolo("pose", "x")
num_people = estimate_pose(model, "image.jpg")
```

### 09. Depth Estimation

Generate depth maps from images.

```python
from scripts.09_depth_estimation import load_depth_anything, estimate_depth

model, processor = load_depth_anything("Large")
depth_map = estimate_depth(model, processor, "image.png")
```

### 10. Segment Anything (SAM)

Universal image segmentation.

```python
# Automatic segmentation
from scripts.10_segment_anything import segment_automatic

outputs, image = segment_automatic("image.png")
print(f"Found {len(outputs['masks'])} segments")

# Point-based segmentation
from scripts.10_segment_anything import load_sam, segment_with_points

model, processor = load_sam()
masks, image = segment_with_points(model, processor, "image.png", points=[[100, 200]])
```

### 11. Image Inpainting & Background Removal

Edit image regions with AI and remove backgrounds.

```python
# Background removal
from scripts.11_inpainting_background import remove_background

output_img = remove_background("photo.jpg", "photo_nobg.png")

# Inpainting with SD 1.5 (512x512)
from scripts.11_inpainting_background import load_inpainting_model, inpaint, create_mask_circle

pipe = load_inpainting_model("sd15")
mask = create_mask_circle((512, 512), radius=100)
result = inpaint(pipe, "image.png", mask, prompt="a beautiful flower")

# Inpainting with SDXL (1024x1024, higher quality)
pipe = load_inpainting_model("sdxl")
result = inpaint(pipe, "image.png", mask, prompt="a golden dragon")

# Outpainting (extend image)
from scripts.11_inpainting_background import outpaint

result = outpaint(pipe, "image.png", direction="right", prompt="forest landscape")
```

### 12. Image Upscaling / Super Resolution

Enhance image resolution with AI upscalers.

```python
# Swin2SR (fast, transformer-based)
from scripts.12_image_upscaling import load_swin2sr, upscale_swin2sr

model, processor = load_swin2sr(scale=4)  # 4x upscaling
result = upscale_swin2sr(model, processor, "low_res.png", max_size=512)

# SD x4 Upscaler (diffusion-based, high quality)
from scripts.12_image_upscaling import load_sd_upscaler, upscale_sd

pipe = load_sd_upscaler()
result = upscale_sd(pipe, "image.png", prompt="detailed photo, sharp")

# Tiled upscaling for large images
from scripts.12_image_upscaling import upscale_tiled

result = upscale_tiled(model, processor, "large_image.png", tile_size=256)
```

### 13. ControlNet & Image-to-Image

Transform images with AI guidance and control signals.

```python
# Image-to-Image transformation
from scripts.13_controlnet_img2img import load_img2img, transform_image

pipe = load_img2img()
result = transform_image(pipe, "photo.png", prompt="oil painting style", strength=0.6)

# ControlNet with Canny edges
from scripts.13_controlnet_img2img import load_controlnet_canny, image_to_controlnet

pipe = load_controlnet_canny()
control_img, result = image_to_controlnet(
    pipe, "photo.png",
    prompt="cyberpunk city at night",
    control_type="canny"
)

# ControlNet with depth map
from scripts.13_controlnet_img2img import load_controlnet_depth, generate_with_controlnet

pipe = load_controlnet_depth()
result = generate_with_controlnet(pipe, "depth_map.png", prompt="underwater scene")
```

## Output Directories

Generated files are saved to:
- `outputs/images/` - Generated images
- `outputs/video/` - Generated videos
- `outputs/audio/` - Generated audio (music, speech)
- `outputs/depth/` - Depth maps
- `outputs/segmentation/` - Segmentation results

## Tips

### Managing VRAM

Clear GPU memory between models:
```python
import torch

del model
torch.cuda.empty_cache()
```

### Using 4-bit Quantization

For large models that don't fit in VRAM:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
```

### Gated Models

Some models require HuggingFace login:
```python
from huggingface_hub import login
login(token="your_token_here")
```

Models requiring access:
- `meta-llama/Llama-3.1-8B-Instruct` - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- `stabilityai/stable-diffusion-3-medium` - https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
- `black-forest-labs/FLUX.1-schnell` - https://huggingface.co/black-forest-labs/FLUX.1-schnell

## Jupyter Notebook

Start Jupyter for interactive experimentation:
```bash
source ~/pytorch-env/bin/activate
jupyter lab
```

## Troubleshooting

### Out of Memory
- Use smaller model variants (e.g., `yolov8s` instead of `yolov8x`)
- Enable 4-bit quantization for LLMs
- Use `pipe.enable_model_cpu_offload()` for diffusion models
- Clear cache with `torch.cuda.empty_cache()`

### Model Download Issues
- Check internet connection
- Try HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`
- Download manually: `huggingface-cli download model_name`

### CUDA Errors
- Verify CUDA: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## License

This toolkit uses various open-source models with their respective licenses:
- Stable Diffusion: CreativeML Open RAIL-M
- Llama: Meta Llama License
- Whisper: MIT
- YOLOv8: AGPL-3.0
- SAM: Apache 2.0

---

Created: January 2026
System: Ubuntu 24.04 LTS | RTX 4080 SUPER | CUDA 12.4 | PyTorch 2.6.0
