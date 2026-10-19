import os
import glob
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random

def load_data_yaml(yaml_path="data.yaml"):
    """
    Parses the data.yaml file to extract dataset paths and class labels.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    class_names = data.get("names", [])
    train_path = data.get("train", "")
    val_path = data.get("val", "")
    test_path = data.get("test", "")
    
    return {
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path
    }


class EwasteDataset(Dataset):
    """
    PyTorch Dataset for E-Waste Object Detection in Faster R-CNN format.
    Converts YOLO format bounding boxes [cx, cy, w, h] (normalized) 
    to Faster R-CNN format [x1, y1, x2, y2] (pixel coordinates).
    Faster R-CNN requires label 0 to be background, so object class labels
    are shifted by +1 (1 to num_classes).
    """
    def __init__(self, images_dir, labels_dir=None, is_train=True, img_size=(640, 640)):
        self.images_dir = images_dir
        self.is_train = is_train
        self.img_size = img_size
        
        # If labels_dir not explicitly provided, infer from images_dir
        # e.g., .../train/images -> .../train/labels
        if labels_dir is None:
            parent = os.path.dirname(os.path.normpath(images_dir))
            labels_dir = os.path.join(parent, "labels")
        self.labels_dir = labels_dir

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_files = []
        if os.path.isdir(self.images_dir):
            for f in sorted(os.listdir(self.images_dir)):
                if f.lower().endswith(valid_extensions):
                    self.image_files.append(os.path.join(self.images_dir, f))
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No valid images found in directory: {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Find corresponding label file
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{file_stem}.txt")

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            cx = float(parts[1])
                            cy = float(parts[2])
                            bw = float(parts[3])
                            bh = float(parts[4])

                            # Convert YOLO normalized cx, cy, w, h to absolute pixel [x1, y1, x2, y2]
                            x1 = (cx - bw / 2.0) * orig_w
                            y1 = (cy - bh / 2.0) * orig_h
                            x2 = (cx + bw / 2.0) * orig_w
                            y2 = (cy + bh / 2.0) * orig_h

                            # Clamp to image boundaries
                            x1 = max(0.0, min(x1, float(orig_w)))
                            y1 = max(0.0, min(y1, float(orig_h)))
                            x2 = max(0.0, min(x2, float(orig_w)))
                            y2 = max(0.0, min(y2, float(orig_h)))

                            # Keep only valid non-degenerate boxes
                            if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                                boxes.append([x1, y1, x2, y2])
                                # Faster R-CNN: 0 is background, so dataset classes are 1..nc
                                labels.append(cls_id + 1)
                        except (ValueError, IndexError):
                            continue

        # Data augmentation during training: Random Horizontal Flip
        if self.is_train and random.random() > 0.5:
            image = F.hflip(image)
            flipped_boxes = []
            for b in boxes:
                # [orig_w - x2, y1, orig_w - x1, y2]
                new_x1 = orig_w - b[2]
                new_x2 = orig_w - b[0]
                flipped_boxes.append([new_x1, b[1], new_x2, b[3]])
            boxes = flipped_boxes

        # Resize image and adjust bounding boxes
        target_w, target_h = self.img_size
        image = F.resize(image, [target_h, target_w])
        
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)

        scaled_boxes = []
        for b in boxes:
            sx1 = b[0] * scale_x
            sy1 = b[1] * scale_y
            sx2 = b[2] * scale_x
            sy2 = b[3] * scale_y
            scaled_boxes.append([sx1, sy1, sx2, sy2])

        # Convert image to Tensor in range [0.0, 1.0]
        img_tensor = F.to_tensor(image)

        if len(scaled_boxes) > 0:
            boxes_tensor = torch.as_tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([orig_h, orig_w], dtype=torch.int64),
            "image_path": img_path
        }

        return img_tensor, target


def collate_fn(batch):
    """
    Collate function for Faster R-CNN DataLoader.
    Combines images and target dicts into tuples since target box counts differ per image.
    """
    return tuple(zip(*batch))
