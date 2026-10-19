import os
import sys
import time
import json
import csv
import argparse
from datetime import timedelta
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from dataset_faster_rcnn import EwasteDataset, collate_fn, load_data_yaml


def build_faster_rcnn(num_classes=78, pretrained=True):
    """
    Builds Faster R-CNN with ResNet-50-FPN backbone and Two-Stage RPN.
    num_classes: Total classes including background (class 0 is background).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Replace the Fast R-CNN classification & box regression head for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def plot_training_curves(csv_file, output_path):
    """
    Plots training and validation loss curves and saves to disk.
    """
    if not os.path.exists(csv_file):
        return

    epochs = []
    train_losses = []
    val_losses = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if len(epochs) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="#2563eb", linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label="Validation Loss", color="#dc2626", linewidth=2, marker='s', markersize=4)
    plt.title("Model 3: Faster R-CNN (Two-stage RPN) - Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_one_epoch(model, optimizer, data_loader, device, scaler, grad_accum=4):
    """
    Trains Faster R-CNN for one epoch using mixed precision and gradient accumulation.
    """
    model.train()
    running_total_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0

    optimizer.zero_grad()
    pbar = tqdm(data_loader, desc="  Train", leave=False)

    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        use_cuda = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # Scale loss for gradient accumulation
            loss_scaled = losses / grad_accum

        if scaler is not None and use_cuda:
            scaler.scale(loss_scaled).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(data_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_scaled.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(data_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()

        # Track losses
        l_cls = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
        l_box = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
        l_obj = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
        l_rpn = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
        l_tot = losses.item()

        running_total_loss += l_tot
        running_loss_classifier += l_cls
        running_loss_box_reg += l_box
        running_loss_objectness += l_obj
        running_loss_rpn_box_reg += l_rpn

        pbar.set_postfix({
            "loss": f"{l_tot:.3f}",
            "cls": f"{l_cls:.3f}",
            "box": f"{l_box:.3f}",
            "rpn_obj": f"{l_obj:.3f}",
            "rpn_box": f"{l_rpn:.3f}"
        })

    num_batches = len(data_loader)
    return {
        "total_loss": running_total_loss / max(1, num_batches),
        "loss_classifier": running_loss_classifier / max(1, num_batches),
        "loss_box_reg": running_loss_box_reg / max(1, num_batches),
        "loss_objectness": running_loss_objectness / max(1, num_batches),
        "loss_rpn_box_reg": running_loss_rpn_box_reg / max(1, num_batches)
    }


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """
    Computes validation loss across the validation set.
    Faster R-CNN computes losses only when in train() mode with targets.
    """
    model.train()  # Keeps loss computation active without gradients
    running_val_loss = 0.0
    pbar = tqdm(data_loader, desc="  Validation", leave=False)

    use_cuda = device.type == "cuda"
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        running_val_loss += losses.item()
        pbar.set_postfix({"val_loss": f"{losses.item():.3f}"})

    num_batches = len(data_loader)
    return running_val_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN (Two-stage RPN) for E-Waste Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset definition")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step (default: 4, safe for 6GB GPU)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum = 16)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate (default: 0.0005)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay for optimizer (default: 0.0005)")
    parser.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers (default: 2)")
    parser.add_argument("--img-size", type=int, default=640, help="Image size input resolution (default: 640)")
    parser.add_argument("--output-dir", type=str, default="runs/faster_rcnn", help="Output directory to store model checkpoints & logs")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pth to resume training from")
    parser.add_argument("--device", type=str, default="", help="Device: cuda or cpu (auto-detected if empty)")
    args = parser.parse_args()

    # Create isolated output directory
    weights_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 STARTING TRAINING: Model 3 - Faster R-CNN (Two-Stage RPN)")
    print("=" * 80)

    # 1. Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    if device.type == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 2. Dataset loading
    yaml_info = load_data_yaml(args.data)
    class_names = yaml_info["class_names"]
    num_classes = yaml_info["num_classes"] + 1  # +1 for background class 0
    print(f"[*] Total E-Waste Classes: {yaml_info['num_classes']} (Model Output Classes: {num_classes} incl. background)")

    # Save class mappings for inference
    classes_json = os.path.join(args.output_dir, "classes.json")
    with open(classes_json, "w", encoding="utf-8") as f:
        json.dump({"background": 0, "classes": class_names}, f, indent=2)
    print(f"[*] Saved class mappings to: {classes_json}")

    print(f"[*] Loading training dataset from: {yaml_info['train_path']}")
    train_dataset = EwasteDataset(yaml_info["train_path"], is_train=True, img_size=(args.img_size, args.img_size))
    print(f"[*] Loading validation dataset from: {yaml_info['val_path']}")
    val_dataset = EwasteDataset(yaml_info["val_path"], is_train=False, img_size=(args.img_size, args.img_size))

    print(f"[*] Train images: {len(train_dataset)} | Validation images: {len(val_dataset)}")
    effective_batch = args.batch_size * args.grad_accum
    print(f"[*] Batch size: {args.batch_size} (Grad Accum: {args.grad_accum} -> Effective Batch Size: {effective_batch})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # 3. Model construction
    print("[*] Building Faster R-CNN with ResNet-50-FPN backbone and Two-Stage RPN...")
    model = build_faster_rcnn(num_classes=num_classes, pretrained=True)
    model.to(device)

    # 4. Optimizer & LR Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(args.resume):
        print(f"[*] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[*] Resumed at epoch {start_epoch} (Best val loss so far: {best_val_loss:.4f})")

    # CSV log setup
    csv_log_path = os.path.join(args.output_dir, "training_metrics.csv")
    csv_exists = os.path.exists(csv_log_path)
    csv_file = open(csv_log_path, "a", newline="", encoding="utf-8")
    fieldnames = [
        "epoch", "train_loss", "val_loss", "loss_classifier", 
        "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", 
        "lr", "time_sec"
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()

    plot_img_path = os.path.join(args.output_dir, "loss_history.png")
    best_model_path = os.path.join(weights_dir, "best.pth")
    last_model_path = os.path.join(weights_dir, "last.pth")

    print("\n" + "=" * 80)
    print(f"🚀 Training for {args.epochs} Epochs. Weights saved in: {weights_dir}")
    print("=" * 80 + "\n")

    overall_start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"--- Epoch [{epoch}/{args.epochs}] (LR: {current_lr:.6f}) ---")
            
            # Train step
            train_metrics = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                scaler=scaler,
                grad_accum=args.grad_accum
            )

            # Validation step
            val_loss = evaluate_loss(model=model, data_loader=val_loader, device=device)
            lr_scheduler.step()

            epoch_time = time.time() - epoch_start
            total_elapsed = time.time() - overall_start

            print(f" Epoch {epoch}/{args.epochs} Results:")
            print(f"  • Train Loss : {train_metrics['total_loss']:.4f} "
                  f"[Cls: {train_metrics['loss_classifier']:.4f}, Box: {train_metrics['loss_box_reg']:.4f}, "
                  f"RPN Cls: {train_metrics['loss_objectness']:.4f}, RPN Box: {train_metrics['loss_rpn_box_reg']:.4f}]")
            print(f"  • Val Loss   : {val_loss:.4f}")
            print(f"  • Time       : {epoch_time:.1f}s (Total Elapsed: {str(timedelta(seconds=int(total_elapsed)))})")

            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"  🏆 NEW BEST MODEL! (Val loss: {best_val_loss:.4f}) -> Saving {best_model_path}")

            # Prepare checkpoint dict
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "train_metrics": train_metrics,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "classes": class_names,
                "num_classes": num_classes,
                "img_size": args.img_size
            }

            # Save latest checkpoint
            torch.save(checkpoint_state, last_model_path)
            if is_best:
                torch.save(checkpoint_state, best_model_path)

            # Log to CSV
            csv_writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_metrics['total_loss']:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "loss_classifier": f"{train_metrics['loss_classifier']:.4f}",
                "loss_box_reg": f"{train_metrics['loss_box_reg']:.4f}",
                "loss_objectness": f"{train_metrics['loss_objectness']:.4f}",
                "loss_rpn_box_reg": f"{train_metrics['loss_rpn_box_reg']:.4f}",
                "lr": f"{current_lr:.6f}",
                "time_sec": f"{epoch_time:.1f}"
            })
            csv_file.flush()

            # Update loss curve plot
            plot_training_curves(csv_log_path, plot_img_path)
            print("-" * 80)

    except KeyboardInterrupt:
        print("\n[!] Training paused by user. Latest progress safely preserved in last.pth.")
    finally:
        csv_file.close()

    total_training_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print("✅ Faster R-CNN Training Cycle Finished!")
    print(f"[*] Total Time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"[*] Best Checkpoint : {best_model_path}")
    print(f"[*] Last Checkpoint : {last_model_path}")
    print(f"[*] Training Curve  : {plot_img_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
