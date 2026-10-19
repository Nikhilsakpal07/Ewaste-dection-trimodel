import os
import sys
import time
import json
import argparse
from datetime import timedelta
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5 on E-Waste Dataset (Model 3 Baseline)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--weights", type=str, default="yolov5nu.pt", help="Pretrained model weights (default: yolov5nu.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (default: auto-detected)")
    parser.add_argument("--project", type=str, default="runs/yolov5", help="Directory to save runs (default: runs/yolov5)")
    parser.add_argument("--name", type=str, default="train", help="Run name inside project (default: train)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--device", type=str, default=None, help="Device to use: '0', 'cpu', etc. (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve data.yaml path
    if args.data is None:
        data_yaml = os.path.join(base_dir, "data.yaml")
    else:
        data_yaml = os.path.abspath(args.data) if not os.path.isabs(args.data) else args.data

    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found at {data_yaml}")
        sys.exit(1)

    # Determine device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("  MODEL 3: YOLOv5 (Single-Stage CSPDarknet53 Detector)")
    print("  E-Waste Detection Training")
    print("=" * 70)
    print(f"  Base Weights   : {args.weights}")
    print(f"  Dataset Config : {data_yaml}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch Size     : {args.batch_size}")
    print(f"  Image Size     : {args.imgsz}")
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f"  Project Dir    : {args.project}")
    print(f"  Run Name       : {args.name}")
    print("=" * 70)

    # Initialize model
    weights_path = args.weights
    if not os.path.isabs(weights_path) and os.path.exists(os.path.join(base_dir, weights_path)):
        weights_path = os.path.join(base_dir, weights_path)

    model = YOLO(weights_path)

    start_time = time.time()

    # Train
    train_results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        workers=args.workers,
        project=os.path.join(base_dir, args.project) if not os.path.isabs(args.project) else args.project,
        name=args.name,
        device=device,
        resume=args.resume,
        verbose=True,
        save=True,
        exist_ok=True
    )

    total_duration = time.time() - start_time
    formatted_duration = str(timedelta(seconds=int(total_duration)))

    print("\n" + "=" * 70)
    print("  YOLOv5 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"  Total Duration: {formatted_duration}")
    print("=" * 70)

    # Check for saved weights
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        last_pt = os.path.join(save_dir, "weights", "last.pt")
        print(f"  Save Directory : {save_dir}")
        if os.path.exists(best_pt):
            print(f"  Best Weights   : {best_pt} ({os.path.getsize(best_pt) / (1024*1024):.2f} MB)")
        if os.path.exists(last_pt):
            print(f"  Last Weights   : {last_pt} ({os.path.getsize(last_pt) / (1024*1024):.2f} MB)")

        # Save model metadata
        metadata = {
            "model_name": "YOLOv5",
            "architecture": "Single-stage, anchor-based CSPDarknet53",
            "base_weights": args.weights,
            "dataset": data_yaml,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "duration": formatted_duration,
            "save_dir": str(save_dir),
            "best_weights": best_pt if os.path.exists(best_pt) else None,
            "last_weights": last_pt if os.path.exists(last_pt) else None,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        meta_file = os.path.join(save_dir, "model_info.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"  Metadata saved : {meta_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()
