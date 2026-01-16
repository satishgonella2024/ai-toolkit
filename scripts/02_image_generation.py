#!/usr/bin/env python3
"""
Image Generation with Stable Diffusion XL, SD3, FLUX, and Playground v2.5
"""

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
)
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sdxl_turbo():
    """Load SDXL Turbo for fast generation (4 steps)"""
    print("Loading SDXL Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to("cuda")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def load_sdxl():
    """Load full SDXL 1.0 for high quality"""
    print("Loading SDXL 1.0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def load_playground():
    """Load Playground v2.5 for aesthetic images"""
    print("Loading Playground v2.5...")
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def load_sd3():
    """Load Stable Diffusion 3 Medium (requires HF access approval)"""
    print("Loading SD3 Medium...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()  # Required for 16GB VRAM
    print("SD3 loaded with CPU offload")
    return pipe


def load_flux():
    """Load FLUX.1-schnell (requires HF access approval)"""
    print("Loading FLUX.1-schnell...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("FLUX loaded with sequential CPU offload")
    return pipe


def generate_image_turbo(pipe, prompt, output_name="turbo_output.png"):
    """Generate with SDXL Turbo (fast, 512x512)"""
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    image.save(output_path)
    print(f"Saved: {output_path}")
    return image


def generate_image_sdxl(pipe, prompt, negative_prompt="blurry, low quality", output_name="sdxl_output.png"):
    """Generate with SDXL (high quality, 1024x1024)"""
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    image.save(output_path)
    print(f"Saved: {output_path}")
    return image


def generate_image_playground(pipe, prompt, negative_prompt="blurry, low quality", output_name="playground_output.png"):
    """Generate with Playground v2.5 (aesthetic, 1024x1024)"""
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=3.0,
        width=1024,
        height=1024,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    image.save(output_path)
    print(f"Saved: {output_path}")
    return image


def generate_image_sd3(pipe, prompt, negative_prompt="blurry, low quality", output_name="sd3_output.png"):
    """Generate with SD3 (high quality, 1024x1024)"""
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        width=1024,
        height=1024,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    image.save(output_path)
    print(f"Saved: {output_path}")
    return image


def generate_image_flux(pipe, prompt, output_name="flux_output.png", size=512):
    """Generate with FLUX.1-schnell (fast, high quality)"""
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,  # FLUX schnell is very fast
        guidance_scale=0.0,
        width=size,
        height=size,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    image.save(output_path)
    print(f"Saved: {output_path}")
    return image


if __name__ == "__main__":
    # Example: Generate with Playground v2.5
    pipe = load_playground()

    prompt = "A majestic dragon flying over a medieval castle at sunset, fantasy art, highly detailed, 8k"
    generate_image_playground(pipe, prompt, output_name="dragon_castle.png")
