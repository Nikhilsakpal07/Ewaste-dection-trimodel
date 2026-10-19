import os
import sys
import argparse
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv5 inference on test images")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--weights", type=str, default="runs/yolov5/train/weights/best.pt", help="Path to trained YOLOv5 weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--project", type=str, default="runs/yolov5", help="Output project folder")
    parser.add_argument("--name", type=str, default="predict", help="Output folder name")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated images")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, args.weights) if not os.path.isabs(args.weights) else args.weights

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    print("=" * 60)
    print("  YOLOv5 Inference (Model 3 Baseline)")
    print(f"  Weights   : {weights_path}")
    print(f"  Source    : {args.source}")
    print(f"  Conf Thresh: {args.conf}")
    print("=" * 60)

    model = YOLO(weights_path)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        project=os.path.join(base_dir, args.project) if not os.path.isabs(args.project) else args.project,
        name=args.name,
        save=args.save,
        verbose=True
    )

    for i, r in enumerate(results):
        boxes = r.boxes
        print(f"\n--- Result {i+1} ---")
        if len(boxes) == 0:
            print("  No objects detected.")
        else:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"Class-{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                print(f"  Detected: {cls_name:30s} | Confidence: {conf:.2%} | Box: {[round(c, 1) for c in xyxy]}")

    print("\nInference complete! Results saved to directory.")


if __name__ == "__main__":
    main()
