#!/usr/bin/env python3
"""
Depth Estimation with Depth Anything V2
Generate depth maps from images
"""

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "depth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_depth_anything(size="Large"):
    """
    Load Depth Anything V2 model

    Args:
        size: "Small", "Base", or "Large"
    """
    model_id = f"depth-anything/Depth-Anything-V2-{size}-hf"
    print(f"Loading Depth Anything V2 {size}...")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")

    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def estimate_depth(model, processor, image_path, colormap="inferno", save_comparison=True):
    """
    Estimate depth from an image

    Args:
        model: Depth Anything model
        processor: Image processor
        image_path: Path to image
        colormap: Matplotlib colormap for visualization
        save_comparison: Save side-by-side comparison

    Returns:
        Depth map as numpy array (normalized 0-1)
    """
    image = Image.open(image_path).convert("RGB")
    print(f"Processing: {Path(image_path).name}")

    # Prepare input
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Normalize depth
    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Save grayscale depth
    depth_gray = (depth * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_gray)
    gray_path = OUTPUT_DIR / f"{Path(image_path).stem}_depth.png"
    depth_image.save(gray_path)
    print(f"Saved: {gray_path}")

    # Save colorized depth
    plt.figure(figsize=(10, 10))
    plt.imshow(depth, cmap=colormap)
    plt.axis("off")
    color_path = OUTPUT_DIR / f"{Path(image_path).stem}_depth_color.png"
    plt.savefig(color_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved: {color_path}")

    # Save side-by-side comparison
    if save_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(depth, cmap=colormap)
        axes[1].set_title("Depth Map (closer = brighter)", fontsize=14)
        axes[1].axis("off")

        plt.tight_layout()
        comparison_path = OUTPUT_DIR / f"{Path(image_path).stem}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {comparison_path}")

    return depth


if __name__ == "__main__":
    model, processor = load_depth_anything("Large")

    # Test images
    test_images = [
        "/home/satish/sdxl_full_output.png",
        "/home/satish/playground_v25_output.png",
        "/home/satish/test_yolo.jpg",
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            estimate_depth(model, processor, img_path)
            print()
        else:
            print(f"Not found: {img_path}")
