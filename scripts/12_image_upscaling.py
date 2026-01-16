#!/usr/bin/env python3
"""
Image Upscaling and Super Resolution
- Swin2SR (transformer-based, fast)
- Stable Diffusion x4 Upscaler (diffusion-based, high quality)
- Tiled processing for large images
"""

import torch
from PIL import Image
from pathlib import Path
import numpy as np


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_swin2sr(scale=4):
    """
    Load Swin2SR super resolution model

    Args:
        scale: Upscaling factor (2 or 4)

    Returns:
        model, processor tuple
    """
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    if scale == 2:
        model_name = "caidas/swin2SR-classical-sr-x2-64"
    elif scale == 4:
        model_name = "caidas/swin2SR-classical-sr-x4-64"
    else:
        raise ValueError(f"Unsupported scale: {scale}. Use 2 or 4.")

    print(f"Loading Swin2SR x{scale}...")
    processor = Swin2SRImageProcessor.from_pretrained(model_name)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
    model = model.to("cuda")
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return model, processor


def upscale_swin2sr(model, processor, image_path, output_name=None, max_size=512):
    """
    Upscale image using Swin2SR

    Args:
        model: Swin2SR model
        processor: Swin2SR processor
        image_path: Path to input image
        output_name: Optional output filename
        max_size: Max input dimension (resize if larger to avoid OOM)

    Returns:
        PIL Image with upscaled result
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"Input size: {original_size[0]}x{original_size[1]}")

    # Resize if too large
    if max(original_size) > max_size:
        ratio = max_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"Resized to: {image.size[0]}x{image.size[1]}")

    # Process
    print("Upscaling with Swin2SR...")
    inputs = processor(image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    output = outputs.reconstruction.squeeze().cpu().clamp(0, 1).numpy()
    output = (output.transpose(1, 2, 0) * 255).astype("uint8")
    result = Image.fromarray(output)

    print(f"Output size: {result.size[0]}x{result.size[1]}")

    # Save
    if output_name is None:
        output_name = Path(image_path).stem + "_swin2sr.png"
    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")

    return result


def load_sd_upscaler():
    """
    Load Stable Diffusion x4 Upscaler

    Returns:
        Diffusion pipeline
    """
    from diffusers import StableDiffusionUpscalePipeline

    print("Loading SD x4 Upscaler...")
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return pipe


def upscale_sd(
    pipe,
    image_path,
    prompt="high quality, detailed, sharp",
    negative_prompt="blurry, low quality, artifacts",
    output_name=None,
    num_inference_steps=20,
    guidance_scale=7.5,
    max_input_size=256,
):
    """
    Upscale image using Stable Diffusion x4 Upscaler

    Args:
        pipe: SD Upscaler pipeline
        image_path: Path to input image
        prompt: Quality guidance prompt
        negative_prompt: What to avoid
        output_name: Optional output filename
        num_inference_steps: Diffusion steps
        guidance_scale: Prompt guidance strength
        max_input_size: Max input dimension (SD upscaler works best with small inputs)

    Returns:
        PIL Image with upscaled result
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"Input size: {original_size[0]}x{original_size[1]}")

    # SD upscaler works best with smaller inputs
    if max(original_size) > max_input_size:
        ratio = max_input_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"Resized to: {image.size[0]}x{image.size[1]}")

    print(f"Upscaling with SD x4 Upscaler (prompt: '{prompt[:30]}...')...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    print(f"Output size: {result.size[0]}x{result.size[1]}")

    # Save
    if output_name is None:
        output_name = Path(image_path).stem + "_sd_upscaled.png"
    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")

    return result


def upscale_tiled(model, processor, image_path, tile_size=256, overlap=32, output_name=None):
    """
    Upscale large image using tiled processing (for Swin2SR)

    Args:
        model: Swin2SR model
        processor: Swin2SR processor
        image_path: Path to input image
        tile_size: Size of each tile
        overlap: Overlap between tiles for seamless blending
        output_name: Optional output filename

    Returns:
        PIL Image with upscaled result
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"Input size: {w}x{h}")

    # Determine scale from model config
    scale = 4  # Default for x4 model

    # Calculate output size
    out_w, out_h = w * scale, h * scale
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

    # Process tiles
    step = tile_size - overlap
    tiles_x = (w + step - 1) // step
    tiles_y = (h + step - 1) // step
    total_tiles = tiles_x * tiles_y
    print(f"Processing {total_tiles} tiles...")

    tile_count = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract tile
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            tile = image.crop((x1, y1, x2, y2))

            # Process tile
            inputs = processor(tile, return_tensors="pt").to("cuda")
            with torch.no_grad():
                tile_output = model(**inputs).reconstruction.squeeze().cpu().clamp(0, 1).numpy()

            tile_output = tile_output.transpose(1, 2, 0)

            # Place in output with blending weights
            ox1, oy1 = x1 * scale, y1 * scale
            ox2, oy2 = ox1 + tile_output.shape[1], oy1 + tile_output.shape[0]

            # Create weight mask (feathered edges)
            tw, th = tile_output.shape[1], tile_output.shape[0]
            weight = np.ones((th, tw, 1), dtype=np.float32)

            # Feather edges
            feather = overlap * scale // 2
            if x > 0:
                weight[:, :feather, :] *= np.linspace(0, 1, feather)[None, :, None]
            if y > 0:
                weight[:feather, :, :] *= np.linspace(0, 1, feather)[:, None, None]
            if x2 < w:
                weight[:, -feather:, :] *= np.linspace(1, 0, feather)[None, :, None]
            if y2 < h:
                weight[-feather:, :, :] *= np.linspace(1, 0, feather)[:, None, None]

            output[oy1:oy2, ox1:ox2] += tile_output * weight
            weights[oy1:oy2, ox1:ox2] += weight

            tile_count += 1
            if tile_count % 10 == 0:
                print(f"  Processed {tile_count}/{total_tiles} tiles")

    # Normalize
    output = output / np.maximum(weights, 1e-8)
    output = (output * 255).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(output)

    print(f"Output size: {result.size[0]}x{result.size[1]}")

    # Save
    if output_name is None:
        output_name = Path(image_path).stem + "_tiled_upscaled.png"
    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")

    return result


if __name__ == "__main__":
    print("=== Swin2SR Super Resolution ===\n")

    # Test Swin2SR
    model, processor = load_swin2sr(scale=4)

    test_image = "/home/satish/ai-toolkit/outputs/images/sdxl_test_output.png"
    if Path(test_image).exists():
        upscale_swin2sr(model, processor, test_image, "upscale_swin2sr_test.png", max_size=256)

    # Clear VRAM
    del model
    torch.cuda.empty_cache()

    print("\n=== SD x4 Upscaler ===\n")

    # Test SD Upscaler
    pipe = load_sd_upscaler()

    if Path(test_image).exists():
        upscale_sd(
            pipe, test_image,
            prompt="a majestic lion in a savanna, highly detailed, 4k",
            output_name="upscale_sd_test.png",
            max_input_size=256
        )

    print("\nDone!")
