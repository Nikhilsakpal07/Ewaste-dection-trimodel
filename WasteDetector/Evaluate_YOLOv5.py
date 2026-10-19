import os
import sys
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 performance on validation/test set")
    parser.add_argument("--weights", type=str, default="runs/yolov5/train/weights/best.pt", help="Path to weights file")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--project", type=str, default="runs/yolov5", help="Project directory")
    parser.add_argument("--name", type=str, default="evaluate", help="Evaluation folder name")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, args.weights) if not os.path.isabs(args.weights) else args.weights
    data_path = os.path.join(base_dir, args.data) if not os.path.isabs(args.data) else args.data

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    print("=" * 60)
    print("  YOLOv5 Model Evaluation")
    print(f"  Weights   : {weights_path}")
    print(f"  Data      : {data_path}")
    print(f"  Split     : {args.split}")
    print("=" * 60)

    model = YOLO(weights_path)
    metrics = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=os.path.join(base_dir, args.project) if not os.path.isabs(args.project) else args.project,
        name=args.name,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  mAP@50     : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95  : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
