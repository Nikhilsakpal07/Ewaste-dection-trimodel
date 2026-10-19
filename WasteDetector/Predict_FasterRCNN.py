import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset_faster_rcnn import load_data_yaml


def load_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 78)
    class_names = checkpoint.get("classes", [])

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def predict_image(model, img_path, class_names, device, conf_thresh=0.4, output_dir="runs/faster_rcnn/predictions"):
    os.makedirs(output_dir, exist_ok=True)
    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Resize to 640x640 for model input
    resized_img = F.resize(orig_img, [640, 640])
    img_tensor = F.to_tensor(resized_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    draw = ImageDraw.Draw(orig_img)
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    boxes = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    detected_items = []

    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
        
        # 0 is background, so classes are 1..nc
        cls_idx = label - 1
        label_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"Class-{label}"
        
        # Scale box back to original image resolution
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y

        draw.rectangle([x1, y1, x2, y2], outline="#22c55e", width=3)
        caption = f"{label_name} {score*100:.1f}%"
        draw.text((x1 + 4, max(0, y1 - 15)), caption, fill="#ef4444")
        detected_items.append({"class": label_name, "score": float(score), "box": [float(x1), float(y1), float(x2), float(y2)]})

    out_file = os.path.join(output_dir, os.path.basename(img_path))
    orig_img.save(out_file)
    print(f"[*] Detected {len(detected_items)} objects in {img_path} -> Saved to {out_file}")
    for item in detected_items:
        print(f"    - {item['class']} ({item['score']*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Predict using Faster R-CNN on E-Waste images")
    parser.add_argument("--weights", type=str, default="runs/faster_rcnn/weights/best.pth", help="Path to weights .pth")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="runs/faster_rcnn/predictions", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.weights, device)

    if os.path.isfile(args.source):
        predict_image(model, args.source, class_names, device, args.conf, args.output)
    elif os.path.isdir(args.source):
        valid_exts = (".jpg", ".jpeg", ".png")
        files = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(valid_exts)]
        for f in files[:20]:
            predict_image(model, f, class_names, device, args.conf, args.output)


if __name__ == "__main__":
    main()
