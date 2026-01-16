#!/usr/bin/env python3
"""
Segment Anything Model (SAM) for universal image segmentation
Segment any object with points, boxes, or automatic detection
"""

import torch
from transformers import SamModel, SamProcessor, pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sam():
    """Load SAM ViT-Huge model (2.45GB VRAM)"""
    print("Loading SAM (Segment Anything)...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to("cuda")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def segment_with_points(model, processor, image_path, points, labels=None):
    """
    Segment objects using point prompts

    Args:
        model: SAM model
        processor: SAM processor
        image_path: Path to image
        points: List of [x, y] coordinates [[x1,y1], [x2,y2], ...]
        labels: List of labels (1=foreground, 0=background), optional

    Returns:
        List of masks
    """
    image = Image.open(image_path).convert("RGB")

    # Format points for SAM
    input_points = [[points]]

    inputs = processor(image, input_points=input_points, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    return masks[0][0], image


def segment_automatic(image_path, points_per_batch=64):
    """
    Automatically segment all objects in an image

    Args:
        image_path: Path to image
        points_per_batch: Points to process per batch (lower = less VRAM)

    Returns:
        Dictionary with masks and scores
    """
    print("Generating automatic masks...")
    generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
    image = Image.open(image_path).convert("RGB")
    outputs = generator(image, points_per_batch=points_per_batch)
    print(f"Found {len(outputs['masks'])} segments")
    return outputs, image


def visualize_masks(image, masks, output_path, title="Segmentation"):
    """Visualize masks overlaid on image"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # With masks
    axes[1].imshow(image)
    np.random.seed(42)

    if isinstance(masks, dict):
        # Automatic segmentation output
        mask_list = masks['masks'][:30]  # Show top 30
    else:
        # Point-based output
        mask_list = [masks[i].numpy() for i in range(min(len(masks), 10))]

    for mask in mask_list:
        if isinstance(mask, np.ndarray):
            mask_array = mask
        else:
            mask_array = mask

        color = np.concatenate([np.random.random(3), [0.4]])
        h, w = mask_array.shape[-2:]
        mask_image = np.zeros((h, w, 4))
        mask_image[mask_array.squeeze() > 0.5] = color
        axes[1].imshow(mask_image)

    num_masks = len(masks['masks']) if isinstance(masks, dict) else len(masks)
    axes[1].set_title(f"{title} ({num_masks} segments)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Test with automatic segmentation
    image_path = "/home/satish/playground_v25_output.png"

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Run 02_image_generation.py first")
        exit(1)

    # Automatic segmentation
    print("=== Automatic Segmentation ===")
    outputs, image = segment_automatic(image_path)
    output_path = OUTPUT_DIR / f"{Path(image_path).stem}_sam_auto.png"
    visualize_masks(image, outputs, output_path, "SAM Auto-Segmentation")

    # Point-based segmentation
    print("\n=== Point-based Segmentation ===")
    model, processor = load_sam()

    # Example: segment center of image
    img = Image.open(image_path)
    center_point = [[img.size[0] // 2, img.size[1] // 2]]

    masks, image = segment_with_points(model, processor, image_path, center_point)
    print(f"Generated {len(masks)} masks from point prompt")

    # Visualize point-based result
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for i, mask in enumerate(masks[:3]):  # Show top 3 masks
        color = np.array([1, 0, 0, 0.4]) if i == 0 else np.concatenate([np.random.random(3), [0.3]])
        mask_array = mask.numpy()
        h, w = mask_array.shape
        mask_image = np.zeros((h, w, 4))
        mask_image[mask_array > 0.5] = color
        ax.imshow(mask_image)

    # Mark the point
    ax.scatter([center_point[0][0]], [center_point[0][1]], c='green', s=200, marker='*')
    ax.axis("off")
    ax.set_title("Point-based Segmentation (green star = input point)")

    point_output_path = OUTPUT_DIR / f"{Path(image_path).stem}_sam_point.png"
    plt.savefig(point_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {point_output_path}")
