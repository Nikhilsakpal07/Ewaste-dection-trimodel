import os
import sys
import time
import json
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset_faster_rcnn import EwasteDataset, collate_fn, load_data_yaml


def compute_iou(box1, box2):
    """
    Computes IoU between two sets of boxes [x1, y1, x2, y2].
    box1: [N, 4]
    box2: [M, 4]
    returns: [N, M] matrix of IoU
    """
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / np.clip(union, 1e-6, None)
    return iou


def calculate_ap(recalls, precisions):
    """
    Computes 11-point or area-under-curve AP.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_dataset(model, data_loader, device, num_classes, iou_thresholds=[0.5, 0.75]):
    """
    Evaluates detection metrics (mAP@0.5, mAP@0.75, mAP@0.5:0.95) over the data_loader.
    """
    model.eval()
    
    # Store all ground truths and predictions per class
    # class_gt[c][img_id] = list of boxes
    # class_pred[c] = list of dict(img_id, score, box)
    class_gts = defaultdict(lambda: defaultdict(list))
    class_preds = defaultdict(list)
    total_latency = 0.0
    total_images = 0

    print("[*] Running inference across evaluation samples...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images_dev = [img.to(device) for img in images]
            
            t0 = time.time()
            predictions = model(images_dev)
            t1 = time.time()
            total_latency += (t1 - t0)
            total_images += len(images)

            for i in range(len(images)):
                img_id = targets[i]["image_id"].item()
                gt_boxes = targets[i]["boxes"].cpu().numpy()
                gt_labels = targets[i]["labels"].cpu().numpy()

                for b, l in zip(gt_boxes, gt_labels):
                    class_gts[int(l)][img_id].append(b)

                pred = predictions[i]
                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
                    if s >= 0.05:  # Evaluation confidence threshold
                        class_preds[int(l)].append({
                            "img_id": img_id,
                            "box": b,
                            "score": float(s)
                        })

    # Calculate AP for each IoU threshold and each class
    # Evaluating IoU from 0.50 to 0.95 with step 0.05
    all_ious = np.linspace(0.5, 0.95, 10)
    class_ap_50 = {}
    class_ap_75 = {}
    class_ap_50_95 = {}

    for c in range(1, num_classes):
        preds = class_preds[c]
        preds.sort(key=lambda x: x["score"], reverse=True)
        gts = class_gts[c]
        num_gt_boxes = sum(len(b) for b in gts.values())

        if num_gt_boxes == 0:
            continue

        if len(preds) == 0:
            class_ap_50[c] = 0.0
            class_ap_75[c] = 0.0
            class_ap_50_95[c] = 0.0
            continue

        ap_per_iou = []
        for iou_thresh in all_ious:
            detected = defaultdict(set)
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for p_idx, p in enumerate(preds):
                img_id = p["img_id"]
                p_box = p["box"][None, :]
                img_gt = gts.get(img_id, [])

                if len(img_gt) == 0:
                    fp[p_idx] = 1
                    continue

                gt_boxes_arr = np.array(img_gt)
                ious = compute_iou(p_box, gt_boxes_arr)[0]
                best_gt_idx = np.argmax(ious)
                best_iou = ious[best_gt_idx]

                if best_iou >= iou_thresh:
                    if best_gt_idx not in detected[img_id]:
                        tp[p_idx] = 1
                        detected[img_id].add(best_gt_idx)
                    else:
                        fp[p_idx] = 1
                else:
                    fp[p_idx] = 1

            cumsum_tp = np.cumsum(tp)
            cumsum_fp = np.cumsum(fp)
            recalls = cumsum_tp / num_gt_boxes
            precisions = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, 1e-6)

            ap = calculate_ap(recalls, precisions)
            ap_per_iou.append(ap)

            if np.isclose(iou_thresh, 0.50):
                class_ap_50[c] = ap
            if np.isclose(iou_thresh, 0.75):
                class_ap_75[c] = ap

        class_ap_50_95[c] = float(np.mean(ap_per_iou))

    map_50 = float(np.mean(list(class_ap_50.values()))) if class_ap_50 else 0.0
    map_75 = float(np.mean(list(class_ap_75.values()))) if class_ap_75 else 0.0
    map_50_95 = float(np.mean(list(class_ap_50_95.values()))) if class_ap_50_95 else 0.0
    avg_latency_ms = (total_latency / max(1, total_images)) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "mAP@50": map_50,
        "mAP@75": map_75,
        "mAP@50:95": map_50_95,
        "evaluated_classes": len(class_ap_50),
        "total_images": total_images,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "per_class_ap50": class_ap_50,
        "per_class_ap50_95": class_ap_50_95
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on E-Waste Dataset")
    parser.add_argument("--weights", type=str, default="runs/faster_rcnn/weights/best.pth", help="Path to checkpoint .pth")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--img-size", type=int, default=640, help="Input image resolution")
    parser.add_argument("--output", type=str, default="runs/faster_rcnn/eval_report.json", help="Path to output JSON")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("📊 EVALUATING MODEL 3: Faster R-CNN (Two-stage RPN)")
    print(f"[*] Weights: {args.weights}")
    print(f"[*] Device : {device}")
    print("=" * 80)

    # Load checkpoint
    checkpoint = torch.load(args.weights, map_location=device)
    num_classes = checkpoint.get("num_classes", 78)
    class_names = checkpoint.get("classes", [])

    if not class_names:
        yaml_info = load_data_yaml(args.data)
        class_names = yaml_info["class_names"]

    # Reconstruct model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Dataset
    yaml_info = load_data_yaml(args.data)
    eval_dir = yaml_info["test_path"] if args.split == "test" else yaml_info["val_path"]
    print(f"[*] Evaluating on {args.split.upper()} set ({eval_dir})...")

    eval_dataset = EwasteDataset(eval_dir, is_train=False, img_size=(args.img_size, args.img_size))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    metrics = evaluate_dataset(model, eval_loader, device, num_classes)

    print("\n" + "=" * 80)
    print("🏁 FASTER R-CNN EVALUATION RESULTS")
    print("=" * 80)
    print(f"  • mAP @ 0.50        : {metrics['mAP@50']:.4f} ({metrics['mAP@50']*100:.2f}%)")
    print(f"  • mAP @ 0.75        : {metrics['mAP@75']:.4f} ({metrics['mAP@75']*100:.2f}%)")
    print(f"  • mAP @ 0.50:0.95   : {metrics['mAP@50:95']:.4f} ({metrics['mAP@50:95']*100:.2f}%)")
    print(f"  • Evaluated Images  : {metrics['total_images']}")
    print(f"  • Average Latency   : {metrics['avg_latency_ms']:.2f} ms per image")
    print(f"  • Inference Speed   : {metrics['fps']:.2f} FPS")
    print("=" * 80)

    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        # Convert keys to string for JSON serialization
        json_metrics = {
            "model": "Faster R-CNN (Two-stage RPN)",
            "weights": args.weights,
            "split": args.split,
            "mAP@50": metrics["mAP@50"],
            "mAP@75": metrics["mAP@75"],
            "mAP@50:95": metrics["mAP@50:95"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "fps": metrics["fps"],
            "evaluated_images": metrics["total_images"]
        }
        json.dump(json_metrics, f, indent=2)

    print(f"[*] Full report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
