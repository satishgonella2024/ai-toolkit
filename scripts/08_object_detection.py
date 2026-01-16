#!/usr/bin/env python3
"""
Object Detection, Segmentation, and Pose Estimation with YOLOv8
"""

from ultralytics import YOLO
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_yolo(task="detect", size="x"):
    """
    Load YOLOv8 model

    Args:
        task: "detect", "segment", or "pose"
        size: "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (extra large)
    """
    if task == "detect":
        model_name = f"yolov8{size}.pt"
    elif task == "segment":
        model_name = f"yolov8{size}-seg.pt"
    elif task == "pose":
        model_name = f"yolov8{size}-pose.pt"
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Loading YOLOv8 {task} ({size})...")
    model = YOLO(model_name)
    return model


def detect_objects(model, image_path, conf=0.25, save=True):
    """
    Detect objects in an image

    Args:
        model: YOLO model
        image_path: Path to image
        conf: Confidence threshold
        save: Save annotated image

    Returns:
        List of detections
    """
    results = model(image_path, conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[cls]
        detections.append({"class": name, "confidence": confidence})

    if save:
        output_path = OUTPUT_DIR / f"{Path(image_path).stem}_detected.jpg"
        results[0].save(str(output_path))
        print(f"Saved: {output_path}")

    return detections


def segment_objects(model, image_path, conf=0.25, save=True):
    """
    Segment objects in an image (instance segmentation)

    Args:
        model: YOLO segmentation model
        image_path: Path to image
        conf: Confidence threshold
        save: Save annotated image

    Returns:
        List of segmentations
    """
    results = model(image_path, conf=conf, verbose=False)

    segmentations = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[cls]
        segmentations.append({"class": name, "confidence": confidence})

    if save:
        output_path = OUTPUT_DIR / f"{Path(image_path).stem}_segmented.jpg"
        results[0].save(str(output_path))
        print(f"Saved: {output_path}")

    return segmentations


def estimate_pose(model, image_path, conf=0.25, save=True):
    """
    Estimate human poses in an image

    Args:
        model: YOLO pose model
        image_path: Path to image
        conf: Confidence threshold
        save: Save annotated image

    Returns:
        Number of people detected with poses
    """
    results = model(image_path, conf=conf, verbose=False)

    num_people = len(results[0].keypoints) if results[0].keypoints is not None else 0

    if save:
        output_path = OUTPUT_DIR / f"{Path(image_path).stem}_pose.jpg"
        results[0].save(str(output_path))
        print(f"Saved: {output_path}")

    return num_people


if __name__ == "__main__":
    # Test image
    image_path = "/home/satish/test_yolo.jpg"

    if not Path(image_path).exists():
        # Download sample image
        import urllib.request
        print("Downloading sample image...")
        urllib.request.urlretrieve(
            "https://ultralytics.com/images/bus.jpg",
            image_path
        )

    # Object Detection
    print("=== Object Detection ===")
    detect_model = load_yolo("detect", "x")
    detections = detect_objects(detect_model, image_path)
    for det in detections:
        print(f"  {det['class']}: {det['confidence']:.1%}")

    # Instance Segmentation
    print("\n=== Instance Segmentation ===")
    segment_model = load_yolo("segment", "x")
    segmentations = segment_objects(segment_model, image_path)
    for seg in segmentations:
        print(f"  {seg['class']}: {seg['confidence']:.1%}")

    # Pose Estimation
    print("\n=== Pose Estimation ===")
    pose_model = load_yolo("pose", "x")
    num_people = estimate_pose(pose_model, image_path)
    print(f"  Detected {num_people} people with pose keypoints")
