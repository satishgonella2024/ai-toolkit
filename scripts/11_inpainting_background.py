#!/usr/bin/env python3
"""
Image Inpainting and Background Removal
- Remove backgrounds with rembg
- Inpaint/edit regions with Stable Diffusion
"""

import torch
from PIL import Image, ImageDraw
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def remove_background(image_path, output_name=None):
    """
    Remove background from an image using rembg

    Args:
        image_path: Path to input image
        output_name: Optional output filename (default: input_nobg.png)

    Returns:
        PIL Image with transparent background
    """
    from rembg import remove

    print(f"Removing background: {Path(image_path).name}")

    input_img = Image.open(image_path)
    output_img = remove(input_img)

    if output_name is None:
        output_name = Path(image_path).stem + "_nobg.png"

    output_path = OUTPUT_DIR / output_name
    output_img.save(output_path)
    print(f"Saved: {output_path}")

    return output_img


def load_inpainting_model(model="sd15"):
    """
    Load inpainting model

    Args:
        model: "sd15" for SD 1.5 inpainting, "sdxl" for SDXL inpainting
    """
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    if model == "sd15":
        print("Loading SD 1.5 Inpainting...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        )
    elif model == "sdxl":
        print("Loading SDXL Inpainting...")
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    pipe.to("cuda")
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return pipe


def create_mask_circle(size, center=None, radius=None):
    """Create a circular mask"""
    w, h = size
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(w, h) // 4

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x, y = center
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)
    return mask


def create_mask_rectangle(size, box=None):
    """Create a rectangular mask"""
    w, h = size
    if box is None:
        # Default: center rectangle
        margin = min(w, h) // 4
        box = [margin, margin, w - margin, h - margin]

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    return mask


def inpaint(
    pipe,
    image_path,
    mask,
    prompt,
    negative_prompt="blurry, low quality, distorted",
    output_name="inpainted.png",
    num_inference_steps=30,
    guidance_scale=7.5,
):
    """
    Inpaint an image region

    Args:
        pipe: Inpainting pipeline
        image_path: Path to input image
        mask: PIL Image mask (white = inpaint, black = keep)
        prompt: What to generate in masked region
        negative_prompt: What to avoid
        output_name: Output filename
        num_inference_steps: Diffusion steps
        guidance_scale: Prompt guidance strength

    Returns:
        PIL Image with inpainted result
    """
    image = Image.open(image_path).convert("RGB")

    # Resize for SD 1.5 (512x512) or keep for SDXL (1024x1024)
    original_size = image.size

    # Determine target size based on pipeline
    if hasattr(pipe, 'unet') and pipe.unet.config.sample_size == 64:
        target_size = (512, 512)  # SD 1.5
    else:
        target_size = (1024, 1024)  # SDXL

    image = image.resize(target_size)
    mask = mask.resize(target_size)

    print(f"Inpainting with prompt: '{prompt[:50]}...'")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")

    return result


def outpaint(pipe, image_path, direction="right", prompt="", output_name="outpainted.png"):
    """
    Extend an image in a direction (outpainting)

    Args:
        pipe: Inpainting pipeline
        image_path: Path to input image
        direction: "left", "right", "top", "bottom"
        prompt: What to generate in extended region
        output_name: Output filename
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Create extended canvas
    extend_amount = w // 2 if direction in ["left", "right"] else h // 2

    if direction == "right":
        new_size = (w + extend_amount, h)
        paste_pos = (0, 0)
        mask_box = [w - 10, 0, w + extend_amount, h]
    elif direction == "left":
        new_size = (w + extend_amount, h)
        paste_pos = (extend_amount, 0)
        mask_box = [0, 0, extend_amount + 10, h]
    elif direction == "bottom":
        new_size = (w, h + extend_amount)
        paste_pos = (0, 0)
        mask_box = [0, h - 10, w, h + extend_amount]
    elif direction == "top":
        new_size = (w, h + extend_amount)
        paste_pos = (0, extend_amount)
        mask_box = [0, 0, w, extend_amount + 10]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Create new canvas and paste original
    canvas = Image.new("RGB", new_size, (128, 128, 128))
    canvas.paste(image, paste_pos)

    # Create mask
    mask = Image.new("L", new_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_box, fill=255)

    print(f"Outpainting {direction}: '{prompt[:50]}...'")

    result = pipe(
        prompt=prompt,
        image=canvas,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    result.save(output_path)
    print(f"Saved: {output_path}")

    return result


if __name__ == "__main__":
    print("=== Background Removal ===\n")

    # Test background removal
    test_image = "/home/satish/ai-toolkit/outputs/images/sdxl_test_output.png"
    if Path(test_image).exists():
        remove_background(test_image, "lion_nobg.png")

    print("\n=== Image Inpainting ===\n")

    # Load inpainting model
    pipe = load_inpainting_model("sd15")

    # Test inpainting
    test_image = "/home/satish/ai-toolkit/outputs/images/sdxl_full_output.png"
    if Path(test_image).exists():
        # Create circular mask in center
        image = Image.open(test_image)
        mask = create_mask_circle(image.size, radius=150)

        # Inpaint with different prompts
        inpaint(
            pipe, test_image, mask,
            prompt="a beautiful koi fish swimming in clear water",
            output_name="inpaint_koi.png"
        )

        inpaint(
            pipe, test_image, mask,
            prompt="a stone lantern with warm glowing light",
            output_name="inpaint_lantern.png"
        )

    print("\nDone!")
