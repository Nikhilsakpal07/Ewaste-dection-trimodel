import os
import cv2
import numpy as np
import threading
import time
import base64
import json
import socket
import uuid
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

# Try importing google.genai
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

app = Flask(__name__)
torch.set_num_threads(2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 77 E-Waste Classes & Domain Metadata
# ---------------------------------------------------------------------------
cls_name = [
    'Air-Conditioner', 'Bar-Phone', 'Battery', 'Blood-Pressure-Monitor', 'Boiler', 
    'CRT-Monitor', 'CRT-TV', 'Calculator', 'Camera', 'Ceiling-Fan', 
    'Christmas-Lights', 'Clothes-Iron', 'Coffee-Machine', 'Compact-Fluorescent-Lamps', 
    'Computer-Keyboard', 'Computer-Mouse', 'Cooled-Dispenser', 'Cooling-Display', 
    'Dehumidifier', 'Desktop-PC', 'Digital-Oscilloscope', 'Dishwasher', 'Drone', 
    'Electric-Bicycle', 'Electric-Guitar', 'Electrocardiograph-Machine', 'Electronic-Keyboard', 
    'Exhaust-Fan', 'Flashlight', 'Flat-Panel-Monitor', 'Flat-Panel-TV', 'Floor-Fan', 
    'Freezer', 'Glucose-Meter', 'HDD', 'Hair-Dryer', 'Headphone', 'LED-Bulb', 
    'Laptop', 'Microwave', 'Music-Player', 'Neon-Sign', 'Network-Switch', 
    'Non-Cooled-Dispenser', 'Oven', 'PCB', 'Patient-Monitoring-System', 
    'Photovoltaic-Panel', 'PlayStation-5', 'Power-Adapter', 'Printer', 'Projector', 
    'Pulse-Oximeter', 'Range-Hood', 'Refrigerator', 'Rotary-Mower', 'Router', 
    'SSD', 'Server', 'Smart-Watch', 'Smartphone', 'Smoke-Detector', 'Soldering-Iron', 
    'Speaker', 'Stove', 'Straight-Tube-Fluorescent-Lamp', 'Street-Lamp', 
    'TV-Remote-Control', 'Table-Lamp', 'Tablet', 'Telephone-Set', 'Toaster', 
    'Tumble-Dryer', 'USB-Flash-Drive', 'Vacuum-Cleaner', 'Washing-Machine', 'Xbox-Series-X'
]

value_mapping = {
    'Air-Conditioner': 1200, 'Bar-Phone': 100, 'Battery': 0, 'Blood-Pressure-Monitor': 50, 
    'Boiler': 800, 'CRT-Monitor': 0, 'CRT-TV': 0, 'Calculator': 30, 'Camera': 80, 
    'Ceiling-Fan': 150, 'Christmas-Lights': 20, 'Clothes-Iron': 50, 'Coffee-Machine': 100, 
    'Compact-Fluorescent-Lamps': 0, 'Computer-Keyboard': 30, 'Computer-Mouse': 20, 
    'Cooled-Dispenser': 500, 'Cooling-Display': 1500, 'Dehumidifier': 300, 'Desktop-PC': 850, 
    'Digital-Oscilloscope': 1000, 'Dishwasher': 800, 'Drone': 250, 'Electric-Bicycle': 2000, 
    'Electric-Guitar': 300, 'Electrocardiograph-Machine': 2000, 'Electronic-Keyboard': 400, 
    'Exhaust-Fan': 100, 'Flashlight': 20, 'Flat-Panel-Monitor': 150, 'Flat-Panel-TV': 500, 
    'Floor-Fan': 80, 'Freezer': 1000, 'Glucose-Meter': 30, 'HDD': 60, 'Hair-Dryer': 40, 
    'Headphone': 30, 'LED-Bulb': 0, 'Laptop': 650, 'Microwave': 200, 'Music-Player': 50, 
    'Neon-Sign': 100, 'Network-Switch': 100, 'Non-Cooled-Dispenser': 300, 'Oven': 600, 
    'PCB': 350, 'Patient-Monitoring-System': 3000, 'Photovoltaic-Panel': 800, 'PlayStation-5': 400, 
    'Power-Adapter': 30, 'Printer': 150, 'Projector': 300, 'Pulse-Oximeter': 20, 'Range-Hood': 400, 
    'Refrigerator': 1500, 'Rotary-Mower': 600, 'Router': 40, 'SSD': 50, 'Server': 1000, 
    'Smart-Watch': 100, 'Smartphone': 200, 'Smoke-Detector': 0, 'Soldering-Iron': 30, 
    'Speaker': 100, 'Stove': 500, 'Straight-Tube-Fluorescent-Lamp': 0, 'Street-Lamp': 300, 
    'TV-Remote-Control': 10, 'Table-Lamp': 40, 'Tablet': 150, 'Telephone-Set': 50, 
    'Toaster': 50, 'Tumble-Dryer': 800, 'USB-Flash-Drive': 10, 'Vacuum-Cleaner': 200, 
    'Washing-Machine': 1200, 'Xbox-Series-X': 400
}

hazard_mapping = {
    'Battery': 'HAZMAT WARNING: Fire & explosion risk. Contains Lithium, Lead, or Acid. Do NOT crush or puncture.',
    'CRT-Monitor': 'HAZMAT WARNING: High vacuum implosion hazard. Contains up to 4-8 lbs of toxic Lead and Phosphor glass.',
    'CRT-TV': 'HAZMAT WARNING: High vacuum implosion hazard. Contains toxic Lead glass and Barium.',
    'Compact-Fluorescent-Lamps': 'HAZMAT WARNING: Contains volatile Mercury vapor. Extremely hazardous if broken.',
    'Straight-Tube-Fluorescent-Lamp': 'HAZMAT WARNING: Contains volatile Mercury vapor. Requires hazardous bulb recycler.',
    'Smoke-Detector': 'HAZMAT WARNING: Contains radioactive Americium-241 ionization sources. Requires radioactive disposal protocol.',
    'Air-Conditioner': 'HAZMAT WARNING: Contains pressurized Ozone-depleting Refrigerants (CFC/HFC). Requires licensed evacuation.',
    'Refrigerator': 'HAZMAT WARNING: Contains harmful greenhouse refrigerants and compressor oils.',
    'Freezer': 'HAZMAT WARNING: Contains pressurized refrigerants (Freon/R-134a). Do not vent to atmosphere.',
    'Dehumidifier': 'HAZMAT WARNING: Contains closed-loop refrigerant circuits.'
}

# ---------------------------------------------------------------------------
# Individual Model Engines (Distinct Code & Weights - Never Mixed)
# ---------------------------------------------------------------------------

class YOLOv8Detector:
    """Model 2: YOLOv8 Anchor-Free Production Engine"""
    def __init__(self):
        self.model = None
        self.error = None
        self.lock = threading.Lock()
        self.weights_path = os.path.join(BASE_DIR, "runs", "detect", "train-8", "weights", "best.pt")

    def _ensure_loaded(self):
        if self.model is None and self.error is None:
            with self.lock:
                if self.model is None:
                    try:
                        if not os.path.exists(self.weights_path):
                            raise FileNotFoundError(f"YOLOv8 weights not found at {self.weights_path}")
                        self.model = YOLO(self.weights_path)
                    except Exception as e:
                        self.error = str(e)
                        print(f"[YOLOv8] Error loading model: {e}")

    def predict(self, frame, conf=0.20):
        self._ensure_loaded()
        if self.error:
            return {"error": self.error, "model_id": "yolov8", "name": "YOLOv8 SOTA"}

        t_start = time.perf_counter()
        results = self.model.predict(source=frame, save=False, verbose=False, conf=conf, iou=0.45)
        latency = (time.perf_counter() - t_start) * 1000

        result = results[0]
        plotted_frame = result.plot()
        boxes = result.boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_idx = int(b.cls[0].item())
                score = float(b.conf[0].item())
                cname = cls_name[cls_idx] if cls_idx < len(cls_name) else f"Class-{cls_idx}"
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                detections.append({
                    "class": cname,
                    "confidence": int(score * 100),
                    "score": round(score, 3),
                    "box": xyxy,
                    "value": f"₹{value_mapping.get(cname, 20)}",
                    "hazard": cname in hazard_mapping,
                    "hazard_text": hazard_mapping.get(cname, "")
                })

        detections.sort(key=lambda d: d["score"], reverse=True)
        top = detections[0] if detections else None
        total_val = sum([value_mapping.get(d["class"], 20) for d in detections])
        hazards = [d for d in detections if d["hazard"]]

        return {
            "model_id": "yolov8",
            "model_name": "YOLOv8n Deep Vision",
            "architecture": "Single-Stage Anchor-Free (C2f Backbone + Decoupled Head)",
            "latency_ms": round(latency, 1),
            "fps": round(1000.0 / max(latency, 1.0), 1),
            "count": len(detections),
            "top_detection": top,
            "detections": detections,
            "total_value": f"₹{total_val}",
            "has_hazard": len(hazards) > 0,
            "hazard_text": hazards[0]["hazard_text"] if hazards else "",
            "annotated_frame": plotted_frame
        }


class YOLOv5Detector:
    """Model 1: YOLOv5 Anchor-Based Baseline Engine"""
    def __init__(self):
        self.model = None
        self.error = None
        self.lock = threading.Lock()
        self.weights_path = os.path.join(BASE_DIR, "runs", "yolov5", "train", "weights", "best.pt")

    def _ensure_loaded(self):
        if self.model is None and self.error is None:
            with self.lock:
                if self.model is None:
                    try:
                        if not os.path.exists(self.weights_path):
                            raise FileNotFoundError(f"YOLOv5 weights not found at {self.weights_path}")
                        self.model = YOLO(self.weights_path)
                    except Exception as e:
                        self.error = str(e)
                        print(f"[YOLOv5] Error loading model: {e}")

    def predict(self, frame, conf=0.20):
        self._ensure_loaded()
        if self.error:
            return {"error": self.error, "model_id": "yolov5", "name": "YOLOv5 Baseline"}

        t_start = time.perf_counter()
        results = self.model.predict(source=frame, save=False, verbose=False, conf=conf, iou=0.45)
        latency = (time.perf_counter() - t_start) * 1000

        result = results[0]
        plotted_frame = result.plot()
        boxes = result.boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_idx = int(b.cls[0].item())
                score = float(b.conf[0].item())
                cname = cls_name[cls_idx] if cls_idx < len(cls_name) else f"Class-{cls_idx}"
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                detections.append({
                    "class": cname,
                    "confidence": int(score * 100),
                    "score": round(score, 3),
                    "box": xyxy,
                    "value": f"₹{value_mapping.get(cname, 20)}",
                    "hazard": cname in hazard_mapping,
                    "hazard_text": hazard_mapping.get(cname, "")
                })

        detections.sort(key=lambda d: d["score"], reverse=True)
        top = detections[0] if detections else None
        total_val = sum([value_mapping.get(d["class"], 20) for d in detections])
        hazards = [d for d in detections if d["hazard"]]

        return {
            "model_id": "yolov5",
            "model_name": "YOLOv5s Baseline",
            "architecture": "Single-Stage Anchor-Based (C3 Backbone + Coupled Head)",
            "latency_ms": round(latency, 1),
            "fps": round(1000.0 / max(latency, 1.0), 1),
            "count": len(detections),
            "top_detection": top,
            "detections": detections,
            "total_value": f"₹{total_val}",
            "has_hazard": len(hazards) > 0,
            "hazard_text": hazards[0]["hazard_text"] if hazards else "",
            "annotated_frame": plotted_frame
        }


class FasterRCNNDetector:
    """Model 3: Faster R-CNN ResNet-50-FPN Two-Stage Engine"""
    def __init__(self):
        self.model = None
        self.error = None
        self.class_names = cls_name
        self.device = torch.device("cpu")
        self.lock = threading.Lock()
        self.weights_path = os.path.join(BASE_DIR, "runs", "faster_rcnn", "weights", "best.pth")

    def _ensure_loaded(self):
        if self.model is None and self.error is None:
            with self.lock:
                if self.model is None:
                    try:
                        if not os.path.exists(self.weights_path):
                            raise FileNotFoundError(f"Faster R-CNN weights not found at {self.weights_path}")
                        
                        ckpt = torch.load(self.weights_path, map_location=self.device)
                        num_classes = ckpt.get("num_classes", 78)
                        if "classes" in ckpt and len(ckpt["classes"]) > 0:
                            self.class_names = ckpt["classes"]

                        m = fasterrcnn_resnet50_fpn(weights=None)
                        in_features = m.roi_heads.box_predictor.cls_score.in_features
                        m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                        m.load_state_dict(ckpt["model_state_dict"])
                        m.to(self.device)
                        m.eval()
                        self.model = m
                    except Exception as e:
                        self.error = str(e)
                        print(f"[Faster R-CNN] Error loading model: {e}")

    def predict(self, frame, conf=0.20):
        self._ensure_loaded()
        if self.error:
            return {"error": self.error, "model_id": "faster_rcnn", "name": "Faster R-CNN"}

        t_start = time.perf_counter()
        orig_h, orig_w, _ = frame.shape
        rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resized = TF.resize(rgb_pil, [640, 640])
        tensor = TF.to_tensor(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)[0]

        latency = (time.perf_counter() - t_start) * 1000

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()

        sx = orig_w / 640.0
        sy = orig_h / 640.0

        detections = []
        annotated_frame = frame.copy()

        for box, label, score in zip(boxes, labels, scores):
            if score < conf:
                continue
            
            # Label 0 is background, classes are 1..77
            cls_idx = label - 1
            if 0 <= cls_idx < len(self.class_names):
                cname = self.class_names[cls_idx]
            else:
                cname = f"Class-{label}"

            x1 = int(box[0] * sx)
            y1 = int(box[1] * sy)
            x2 = int(box[2] * sx)
            y2 = int(box[3] * sy)

            is_hazard = cname in hazard_mapping
            hazard_txt = hazard_mapping.get(cname, "")
            score_pct = int(score * 100)

            detections.append({
                "class": cname,
                "confidence": score_pct,
                "score": round(float(score), 3),
                "box": [x1, y1, x2, y2],
                "value": f"₹{value_mapping.get(cname, 20)}",
                "hazard": is_hazard,
                "hazard_text": hazard_txt
            })

            # Draw distinct Faster R-CNN visual box (Violet / Neon Purple theme)
            box_color = (255, 60, 100) if is_hazard else (255, 140, 180)  # BGR
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            label_caption = f"{cname} {score_pct}%"
            (tw, th), _ = cv2.getTextSize(label_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), box_color, -1)
            cv2.putText(annotated_frame, label_caption, (x1 + 2, max(th + 2, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        detections.sort(key=lambda d: d["score"], reverse=True)
        top = detections[0] if detections else None
        total_val = sum([value_mapping.get(d["class"], 20) for d in detections])
        hazards = [d for d in detections if d["hazard"]]

        return {
            "model_id": "faster_rcnn",
            "model_name": "Faster R-CNN ResNet-50",
            "architecture": "Two-Stage Region Proposal (ResNet-50-FPN + RoIAlign)",
            "latency_ms": round(latency, 1),
            "fps": round(1000.0 / max(latency, 1.0), 1),
            "count": len(detections),
            "top_detection": top,
            "detections": detections,
            "total_value": f"₹{total_val}",
            "has_hazard": len(hazards) > 0,
            "hazard_text": hazards[0]["hazard_text"] if hazards else "",
            "annotated_frame": annotated_frame
        }


# Initialize model detectors
detector_v8 = YOLOv8Detector()
detector_v5 = YOLOv5Detector()
detector_frcnn = FasterRCNNDetector()

# ---------------------------------------------------------------------------
# Gemini LLM Environmental Report Generator (Report generation only)
# ---------------------------------------------------------------------------

def generate_gemini_report(detections_summary):
    """
    Generates a comprehensive E-Waste Inspection and Environmental Report.
    NOTE: Gemini LLM is used EXCLUSIVELY for generating this report.
    Image detection is performed entirely by YOLOv8, YOLOv5, and Faster R-CNN.
    """
    items_detected = detections_summary.get("detected_classes", [])
    if not items_detected:
        return (
            "### 📋 E-Waste Inspection Report\n\n"
            "**Status:** No electronic waste components were identified in the scanned frame.\n\n"
            "- Ensure the target hardware is well-illuminated and clearly framed.\n"
            "- Supported categories include PCBs, Batteries, Displays, Computing Equipment, and Household Appliances."
        )

    unique_items = list(set(items_detected))
    items_str = ", ".join(unique_items)
    hazards_found = detections_summary.get("hazards", [])

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if HAS_GENAI and api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
You are an expert Environmental Engineer and certified Hazardous E-Waste Inspector.
An automated computer vision audit utilizing YOLOv8, YOLOv5, and Faster R-CNN has detected the following electronic hardware components:
Detected Items: {items_str}
Detected Hazards: {', '.join(hazards_found) if hazards_found else 'None flagged as high-risk'}

Generate a professional, structured Environmental Impact & Recycling Audit Report in Markdown format with the following exact sections:
1. ## 🔍 Executive Summary & Hardware Breakdown
   Brief technical overview of what was identified and its lifecycle status.
2. ## ☣️ Ecological Hazard & Toxicity Assessment
   Detail the heavy metals (Lead, Cadmium, Mercury, Brominated Flame Retardants, Lithium) present in these items and their risks to soil/groundwater.
3. ## 💎 Urban Mining & Precious Metal Recovery Potential
   Quantify recoverable materials (Gold, Silver, Copper, Palladium, Rare Earth Elements) and explain why landfilling them is economically wasteful.
4. ## 🛠️ Step-by-Step Certified Recycling Roadmap
   Actionable protocols for safe disassembly, battery neutralization, component separation, and certified drop-off.
5. ## 📜 Regulatory Compliance Directives
   Reference applicable standards (RoHS, WEEE Directives, Basel Convention, and Local E-Waste Management Rules).

Keep the report punchy, highly informative, and scientifically accurate.
"""
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[Gemini API Error] {e}")

    # Fallback High-Quality Technical Report Generator
    report_lines = [
        "## 🔍 Executive Summary & Hardware Breakdown",
        f"Automated multi-model computer vision successfully cataloged **{len(unique_items)} hardware categories**: **{items_str}**.",
        "These assemblies contain dense printed circuitry, discrete silicon semiconductors, and structural chassis components.",
        "",
        "## ☣️ Ecological Hazard & Toxicity Assessment"
    ]

    if hazards_found:
        report_lines.append(f"⚠️ **CRITICAL HAZARDS DETECTED:** {len(hazards_found)} components present active contamination hazards:")
        for h in hazards_found:
            report_lines.append(f"- **{h}**")
        report_lines.append("Improper landfill disposal allows leachate to contaminate local aquifers with toxic lead and heavy metals.")
    else:
        report_lines.append("The identified items contain low immediately volatile hazards, but printed circuit boards and power stages contain brominated flame retardants (BFRs) that release dioxins if incinerated.")

    report_lines.extend([
        "",
        "## 💎 Urban Mining & Precious Metal Recovery Potential",
        "Electronic waste holds significantly higher concentrations of precious metals than virgin mined ore:",
        f"- **Gold (Au) & Silver (Ag):** Found on connector pins, bonding wires, and edge fingers of {unique_items[0] if unique_items else 'PCBs'}.",
        "- **Copper (Cu):** High yield recoverable from multi-layer PCB ground planes, transformer coils, and internal harnesses.",
        "- **Palladium & Platinum Group Metals:** Present in multi-layer ceramic capacitors (MLCCs).",
        "",
        "## 🛠️ Step-by-Step Certified Recycling Roadmap",
        "1. **Discharge & Isolation:** Depower and safely isolate any internal capacitors or lithium cells.",
        "2. **Chassis Separation:** Segregate external ABS plastics and aluminum heat sinks from populated PCBs.",
        "3. **Authorized E-Waste Drop-Off:** Hand over assemblies exclusively to R2 (Responsible Recycling) or e-Stewards certified facilities.",
        "",
        "## 📜 Regulatory Compliance Directives",
        "- **RoHS Compliant Handling:** Mandates containment of restricted substances (Pb, Cd, Hg, Cr-VI).",
        "- **WEEE Directive 2012/19/EU:** Obligates 85% material recovery and minimum 80% recycling rate by mass.",
        "- **Basel Convention:** Prohibits transboundary movement of hazardous e-waste to developing nations."
    ])

    return "\n".join(report_lines)

# ---------------------------------------------------------------------------
# Inventory Management
# ---------------------------------------------------------------------------

INVENTORY_FILE = os.path.join(BASE_DIR, "inventory.json")

def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        try:
            with open(INVENTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_inventory_to_disk(records):
    try:
        with open(INVENTORY_FILE, 'w') as f:
            json.dump(records, f, indent=4)
    except Exception as e:
        print(f"Error saving inventory: {e}")

inventory_history = load_inventory()

def save_item_to_inventory(top_class, value_str, is_hazard, img_array, model_summary=""):
    if top_class and top_class != 'None':
        filename = f"scan_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        upload_dir = os.path.join(BASE_DIR, "static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        cv2.imwrite(filepath, img_array)

        record = {
            "id": filename,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": f"/static/uploads/{filename}",
            "info": {
                "class": top_class,
                "value": value_str,
                "hazard": is_hazard,
                "summary": model_summary
            }
        }
        inventory_history.insert(0, record)
        save_inventory_to_disk(inventory_history)

# Global video streaming state
camera_lock = threading.Lock()
cap = None
is_streaming = False
current_stream_model = "yolov8"
mobile_sync_data = None

# ---------------------------------------------------------------------------
# Multi-Model Synchronous Inference Core
# ---------------------------------------------------------------------------

def run_all_three_models(frame, conf=0.20):
    """
    Executes YOLOv8, YOLOv5, and Faster R-CNN on the EXACT same input frame.
    Each model runs with its own dedicated weights and output pipeline.
    """
    # 1. Run YOLOv8
    res_v8 = detector_v8.predict(frame, conf=conf)
    
    # 2. Run YOLOv5
    res_v5 = detector_v5.predict(frame, conf=conf)
    
    # 3. Run Faster R-CNN
    res_frcnn = detector_frcnn.predict(frame, conf=conf)

    # Encode images to JPEG Base64
    def to_b64(img):
        if img is None:
            return ""
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    img_v8_b64 = to_b64(res_v8.get("annotated_frame"))
    img_v5_b64 = to_b64(res_v5.get("annotated_frame"))
    img_frcnn_b64 = to_b64(res_frcnn.get("annotated_frame"))

    # Cleanup raw numpy frames from json output
    res_v8_clean = {k: v for k, v in res_v8.items() if k != "annotated_frame"}
    res_v5_clean = {k: v for k, v in res_v5.items() if k != "annotated_frame"}
    res_frcnn_clean = {k: v for k, v in res_frcnn.items() if k != "annotated_frame"}

    res_v8_clean["image"] = img_v8_b64
    res_v5_clean["image"] = img_v5_b64
    res_frcnn_clean["image"] = img_frcnn_b64

    # Build Comparative Consensus
    all_classes = set()
    for m in [res_v8_clean, res_v5_clean, res_frcnn_clean]:
        for d in m.get("detections", []):
            all_classes.add(d["class"])

    all_hazards = []
    for m in [res_v8_clean, res_v5_clean, res_frcnn_clean]:
        if m.get("hazard_text"):
            all_hazards.append(f"{m['model_name']}: {m['hazard_text']}")

    # Determine fastest & highest confidence
    models_list = [res_v8_clean, res_v5_clean, res_frcnn_clean]
    valid_models = [m for m in models_list if "error" not in m]
    
    fastest = min(valid_models, key=lambda m: m["latency_ms"]) if valid_models else res_v8_clean
    most_sensitive = max(valid_models, key=lambda m: m["count"]) if valid_models else res_v8_clean

    # Pick overall representative detection for inventory
    top_candidates = [m["top_detection"] for m in valid_models if m.get("top_detection")]
    consensus_top = max(top_candidates, key=lambda d: d["score"]) if top_candidates else None

    # Generate Gemini Report (Prompt receives text detection summary ONLY)
    report_summary = {
        "detected_classes": list(all_classes),
        "hazards": list(set(all_hazards))
    }
    ai_report = generate_gemini_report(report_summary)

    # Save representative item to inventory
    if consensus_top:
        save_item_to_inventory(
            consensus_top["class"],
            consensus_top["value"],
            consensus_top["hazard"],
            res_v8.get("annotated_frame", frame),
            f"Audited by 3 Models: YOLOv8 ({res_v8_clean['count']}), YOLOv5 ({res_v5_clean['count']}), Faster R-CNN ({res_frcnn_clean['count']})"
        )

    return {
        "models": {
            "yolov8": res_v8_clean,
            "yolov5": res_v5_clean,
            "faster_rcnn": res_frcnn_clean
        },
        "comparison": {
            "fastest_model": fastest.get("model_name", "YOLOv8"),
            "fastest_latency": fastest.get("latency_ms", 0),
            "most_sensitive": most_sensitive.get("model_name", "YOLOv5"),
            "max_objects_found": most_sensitive.get("count", 0),
            "total_distinct_classes": len(all_classes),
            "classes_detected": list(all_classes),
            "consensus_top": consensus_top
        },
        "report": ai_report
    }

# ---------------------------------------------------------------------------
# Streaming Generator
# ---------------------------------------------------------------------------

def generate_video_stream():
    global cap, is_streaming, current_stream_model
    
    with camera_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
    
    is_streaming = True
    
    while is_streaming:
        with camera_lock:
            if cap is None or not cap.isOpened():
                break
            success, frame = cap.read()
            
        if not success:
            break
        
        # In live streaming, run selected model (default YOLOv8 for 30+ FPS)
        if current_stream_model == "yolov5":
            res = detector_v5.predict(frame, conf=0.25)
        elif current_stream_model == "faster_rcnn":
            res = detector_frcnn.predict(frame, conf=0.25)
        else:
            res = detector_v8.predict(frame, conf=0.25)
            
        display_frame = res.get("annotated_frame", frame)
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', local_ip=get_local_ip())

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/video_feed')
def video_feed():
    global current_stream_model
    m = request.args.get('model', 'yolov8').lower()
    if m in ['yolov8', 'yolov5', 'faster_rcnn']:
        current_stream_model = m
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_stream_model', methods=['POST'])
def set_stream_model():
    global current_stream_model
    data = request.get_json() or {}
    m = data.get('model', 'yolov8').lower()
    if m in ['yolov8', 'yolov5', 'faster_rcnn']:
        current_stream_model = m
        return jsonify({"status": "ok", "stream_model": current_stream_model})
    return jsonify({"error": "Invalid model specified"}), 400

@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    global is_streaming, cap
    is_streaming = False
    with camera_lock:
        if cap is not None:
            cap.release()
            cap = None
    return jsonify({"status": "stopped"})

@app.route('/api/analyze_multi_models', methods=['POST'])
def analyze_multi_models():
    """
    Main Endpoint: Runs YOLOv8, YOLOv5, and Faster R-CNN simultaneously on the uploaded frame.
    """
    try:
        if 'image' in request.files:
            file = request.files['image']
            npimg = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        elif request.is_json and 'image_base64' in request.json:
            b64_str = request.json['image_base64'].split(',')[-1]
            img_bytes = base64.b64decode(b64_str)
            npimg = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "No image data provided in request"}), 400

        if frame is None:
            return jsonify({"error": "Failed to decode image matrix"}), 400

        conf = float(request.form.get('conf', 0.20)) if request.form else 0.20
        payload = run_all_three_models(frame, conf=conf)
        return jsonify(payload)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/analyze_image', methods=['POST'])
def analyze_image():
    """Legacy compatibility endpoint pointing to the multi-model pipeline"""
    return analyze_multi_models()

@app.route('/api/analyze_sample', methods=['POST'])
def analyze_sample():
    """Runs simultaneous 3-model scan on a bundled sample image"""
    try:
        data = request.get_json() or {}
        sample_name = data.get('sample_name', 'pcb')
        sample_file = os.path.join(BASE_DIR, 'static', 'samples', f'{sample_name}.jpg')
        
        if not os.path.exists(sample_file):
            return jsonify({"error": f"Sample file not found: {sample_name}.jpg"}), 404
            
        frame = cv2.imread(sample_file)
        if frame is None:
            return jsonify({"error": "Could not read sample file"}), 500

        conf = float(data.get('conf', 0.20))
        payload = run_all_three_models(frame, conf=conf)
        return jsonify(payload)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report_endpoint():
    """Dedicated endpoint to re-generate the Gemini Environmental Report"""
    try:
        data = request.get_json() or {}
        report = generate_gemini_report(data)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sample_images', methods=['GET'])
def get_sample_images():
    samples_dir = os.path.join(BASE_DIR, 'static', 'samples')
    samples = []
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.endswith('.jpg'):
                name = os.path.splitext(f)[0]
                samples.append({
                    "id": name,
                    "title": name.replace('_', ' ').capitalize(),
                    "url": f"/static/samples/{f}"
                })
    return jsonify(samples)

@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    return jsonify(inventory_history)

@app.route('/api/desktop_sync', methods=['GET'])
def desktop_sync():
    global mobile_sync_data
    if mobile_sync_data:
        data = mobile_sync_data
        mobile_sync_data = None
        return jsonify({"has_new": True, "data": data})
    return jsonify({"has_new": False})

@app.route('/api/mobile_upload', methods=['POST'])
def mobile_upload():
    global mobile_sync_data
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    payload = run_all_three_models(frame, conf=0.20)
    top = payload["comparison"].get("consensus_top") or {
        "class": "None", "value": "₹0", "hazard": False, "hazard_text": ""
    }
    payload["info"] = {
        "class": top.get("class", "None"),
        "confidence": top.get("confidence", 0),
        "value": top.get("value", "₹0"),
        "hazard": top.get("hazard", False),
        "hazard_text": top.get("hazard_text", ""),
        "impact": payload.get("report", "")
    }
    payload["image"] = payload["models"]["yolov8"].get("image", "")
    mobile_sync_data = payload
    return jsonify({"status": "success", "data": payload})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
