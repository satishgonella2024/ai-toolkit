#!/usr/bin/env python3
"""
Video Generation with Stable Video Diffusion
Generates short video clips from input images
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "video"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_svd():
    """Load Stable Video Diffusion pipeline"""
    print("Loading Stable Video Diffusion...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()  # Saves VRAM
    print("Model loaded!")
    return pipe


def generate_video(
    pipe,
    image_path,
    output_name="output.mp4",
    num_frames=25,
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=0.1,
    seed=42
):
    """
    Generate video from an image

    Args:
        pipe: SVD pipeline
        image_path: Path to input image
        output_name: Output filename
        num_frames: Number of frames (default 25)
        fps: Frames per second (default 7)
        motion_bucket_id: Amount of motion (0-255, higher = more motion)
        noise_aug_strength: Noise augmentation (0.0-1.0)
        seed: Random seed for reproducibility
    """
    # Load and resize image (SVD needs 1024x576)
    image = Image.open(image_path)
    image = image.resize((1024, 576))

    print(f"Input: {image_path}")
    print(f"Generating {num_frames} frames...")

    generator = torch.manual_seed(seed)
    frames = pipe(
        image,
        decode_chunk_size=8,
        generator=generator,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength
    ).frames[0]

    output_path = OUTPUT_DIR / output_name
    export_to_video(frames, str(output_path), fps=fps)

    duration = num_frames / fps
    print(f"Saved: {output_path} ({duration:.1f}s, {fps}fps)")
    return output_path


if __name__ == "__main__":
    pipe = load_svd()

    # Example: Generate video from an image
    # Replace with your image path
    image_path = "/home/satish/ai-toolkit/outputs/images/playground_output.png"

    if Path(image_path).exists():
        generate_video(
            pipe,
            image_path,
            output_name="animated.mp4",
            motion_bucket_id=150,  # Medium-high motion
        )
    else:
        print(f"Image not found: {image_path}")
        print("Run 02_image_generation.py first to create an image")
