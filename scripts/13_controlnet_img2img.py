#!/usr/bin/env python3
"""
ControlNet and Image-to-Image Generation
- Image-to-Image transformation with SDXL
- ControlNet with Canny edges
- ControlNet with Depth maps
- ControlNet with OpenPose (human poses)
"""

import torch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Image-to-Image
# =============================================================================

def load_img2img():
    """Load SDXL image-to-image pipeline"""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    print("Loading SDXL img2img...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def transform_image(
    pipe,
    image_path,
    prompt,
    negative_prompt="blurry, low quality",
    strength=0.6,
    output_name="img2img_output.png",
    num_inference_steps=30,
    guidance_scale=7.5,
):
    """
    Transform an image based on a text prompt

    Args:
        pipe: img2img pipeline
        image_path: Path to source image
        prompt: What to transform into
        negative_prompt: What to avoid
        strength: How much to change (0=no change, 1=complete change)
        output_name: Output filename
        num_inference_steps: Diffusion steps
        guidance_scale: Prompt guidance strength

    Returns:
        PIL Image with transformed result
    """
    image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    print(f"Source: {Path(image_path).name}")
    print(f"Transforming with strength={strength}: '{prompt[:50]}...'")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")
    return result


# =============================================================================
# ControlNet
# =============================================================================

def load_controlnet_canny():
    """Load ControlNet with Canny edge detection"""
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    print("Loading ControlNet Canny for SDXL...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def load_controlnet_depth():
    """Load ControlNet with depth maps"""
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    print("Loading ControlNet Depth for SDXL...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def extract_canny_edges(image_path, low_threshold=100, high_threshold=200):
    """
    Extract Canny edges from an image

    Args:
        image_path: Path to input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        PIL Image with edges (white on black)
    """
    image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    image_np = np.array(image)

    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges = np.stack([edges] * 3, axis=-1)  # Convert to RGB

    return Image.fromarray(edges)


def generate_with_controlnet(
    pipe,
    control_image,
    prompt,
    negative_prompt="blurry, low quality",
    controlnet_conditioning_scale=0.5,
    output_name="controlnet_output.png",
    num_inference_steps=30,
    guidance_scale=7.5,
):
    """
    Generate image guided by ControlNet

    Args:
        pipe: ControlNet pipeline
        control_image: PIL Image with control signal (edges, depth, etc.)
        prompt: What to generate
        negative_prompt: What to avoid
        controlnet_conditioning_scale: How much ControlNet influences (0-1)
        output_name: Output filename
        num_inference_steps: Diffusion steps
        guidance_scale: Prompt guidance strength

    Returns:
        PIL Image with generated result
    """
    if isinstance(control_image, (str, Path)):
        control_image = Image.open(control_image).convert("RGB").resize((1024, 1024))

    print(f"Generating with ControlNet: '{prompt[:50]}...'")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")
    return result


def image_to_controlnet(
    pipe,
    image_path,
    prompt,
    control_type="canny",
    negative_prompt="blurry, low quality",
    controlnet_conditioning_scale=0.5,
    output_name=None,
):
    """
    Full pipeline: extract control signal from image and generate new image

    Args:
        pipe: ControlNet pipeline
        image_path: Path to source image
        prompt: What to generate
        control_type: "canny" or "depth"
        negative_prompt: What to avoid
        controlnet_conditioning_scale: ControlNet influence strength
        output_name: Output filename

    Returns:
        Tuple of (control_image, result_image)
    """
    if control_type == "canny":
        control_image = extract_canny_edges(image_path)
        control_suffix = "canny"
    elif control_type == "depth":
        # For depth, you need to run depth estimation first
        # This assumes you have a depth map already
        raise ValueError("For depth ControlNet, provide a depth map directly")
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    # Save control image
    control_path = OUTPUT_DIR / f"control_{control_suffix}.png"
    control_image.save(control_path)
    print(f"Saved control image: {control_path}")

    if output_name is None:
        output_name = f"controlnet_{control_suffix}_output.png"

    result = generate_with_controlnet(
        pipe,
        control_image,
        prompt,
        negative_prompt,
        controlnet_conditioning_scale,
        output_name,
    )

    return control_image, result


if __name__ == "__main__":
    print("=== Image-to-Image ===\n")

    # Test img2img
    test_image = "/home/satish/ai-toolkit/outputs/images/sdxl_test_output.png"
    if Path(test_image).exists():
        pipe = load_img2img()
        transform_image(
            pipe,
            test_image,
            prompt="a lion made of ice and snow, frozen, crystalline",
            strength=0.7,
            output_name="img2img_ice_lion.png",
        )

        # Clear memory
        del pipe
        torch.cuda.empty_cache()

    print("\n=== ControlNet Canny ===\n")

    # Test ControlNet Canny
    test_image = "/home/satish/ai-toolkit/outputs/images/playground_v25_output.png"
    if Path(test_image).exists():
        pipe = load_controlnet_canny()
        image_to_controlnet(
            pipe,
            test_image,
            prompt="a steampunk mechanical city, brass and copper, Victorian era",
            control_type="canny",
            controlnet_conditioning_scale=0.5,
            output_name="controlnet_steampunk.png",
        )

    print("\nDone!")
