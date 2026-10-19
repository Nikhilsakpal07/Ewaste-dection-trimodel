import os
import time
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- Page Configuration ---
st.set_page_config(page_title="ReciBot - Tri-Model E-Waste AI Suite", page_icon="♻️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    'Battery': 'HAZMAT WARNING: Fire & explosion risk. Contains heavy metals/acid. Do NOT crush or puncture.',
    'CRT-Monitor': 'HAZMAT WARNING: Contains high levels of toxic Lead and Phosphor glass.',
    'CRT-TV': 'HAZMAT WARNING: Contains high levels of toxic Lead and Phosphor glass.',
    'Compact-Fluorescent-Lamps': 'HAZMAT WARNING: Contains Mercury vapor. Extremely toxic if broken.',
    'Straight-Tube-Fluorescent-Lamp': 'HAZMAT WARNING: Contains Mercury vapor. Hazardous.',
    'Smoke-Detector': 'HAZMAT WARNING: Contains radioactive Americium-241. Specialized disposal required.',
    'Air-Conditioner': 'HAZMAT WARNING: Contains harmful refrigerants (CFCs/HFCs).',
    'Refrigerator': 'HAZMAT WARNING: Contains ozone-depleting refrigerants.',
    'Freezer': 'HAZMAT WARNING: Contains pressurized refrigerants.',
    'Dehumidifier': 'HAZMAT WARNING: Contains harmful greenhouse refrigerants.'
}

# --- Dedicated Model Loaders (Cached in Memory) ---

@st.cache_resource
def load_yolov8():
    w_path = os.path.join(BASE_DIR, "runs", "detect", "train-8", "weights", "best.pt")
    if os.path.exists(w_path):
        return YOLO(w_path)
    return None

@st.cache_resource
def load_yolov5():
    w_path = os.path.join(BASE_DIR, "runs", "yolov5", "train", "weights", "best.pt")
    if os.path.exists(w_path):
        return YOLO(w_path)
    return None

@st.cache_resource
def load_faster_rcnn():
    w_path = os.path.join(BASE_DIR, "runs", "faster_rcnn", "weights", "best.pth")
    if not os.path.exists(w_path):
        return None, []
    
    ckpt = torch.load(w_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 78)
    class_names = ckpt.get("classes", cls_name)
    
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names

m_v8 = load_yolov8()
m_v5 = load_yolov5()
m_frcnn, frcnn_classes = load_faster_rcnn()

# --- Dedicated Model Inference Functions ---

def run_yolov8_inference(frame, conf=0.20):
    if m_v8 is None:
        return frame, {"error": "YOLOv8 weights not found"}
    t0 = time.perf_counter()
    res = m_v8.predict(source=frame, save=False, verbose=False, conf=conf, iou=0.45)[0]
    latency = (time.perf_counter() - t0) * 1000
    plotted = res.plot()
    boxes = res.boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            idx = int(b.cls[0].item())
            score = float(b.conf[0].item())
            cname = cls_name[idx] if idx < len(cls_name) else f"Class-{idx}"
            detections.append({
                "class": cname,
                "confidence": int(score * 100),
                "score": score,
                "value": f"₹{value_mapping.get(cname, 20)}",
                "hazard": cname in hazard_mapping,
                "hazard_text": hazard_mapping.get(cname, "")
            })
    detections.sort(key=lambda d: d["score"], reverse=True)
    return plotted, {
        "model": "YOLOv8n SOTA",
        "architecture": "Anchor-Free C2f",
        "latency_ms": round(latency, 1),
        "fps": round(1000.0 / max(latency, 1.0), 1),
        "count": len(detections),
        "top": detections[0] if detections else None,
        "detections": detections,
        "has_hazard": any(d["hazard"] for d in detections),
        "total_val": sum(value_mapping.get(d["class"], 20) for d in detections)
    }

def run_yolov5_inference(frame, conf=0.20):
    if m_v5 is None:
        return frame, {"error": "YOLOv5 weights not found"}
    t0 = time.perf_counter()
    res = m_v5.predict(source=frame, save=False, verbose=False, conf=conf, iou=0.45)[0]
    latency = (time.perf_counter() - t0) * 1000
    plotted = res.plot()
    boxes = res.boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            idx = int(b.cls[0].item())
            score = float(b.conf[0].item())
            cname = cls_name[idx] if idx < len(cls_name) else f"Class-{idx}"
            detections.append({
                "class": cname,
                "confidence": int(score * 100),
                "score": score,
                "value": f"₹{value_mapping.get(cname, 20)}",
                "hazard": cname in hazard_mapping,
                "hazard_text": hazard_mapping.get(cname, "")
            })
    detections.sort(key=lambda d: d["score"], reverse=True)
    return plotted, {
        "model": "YOLOv5s Baseline",
        "architecture": "Anchor-Based C3",
        "latency_ms": round(latency, 1),
        "fps": round(1000.0 / max(latency, 1.0), 1),
        "count": len(detections),
        "top": detections[0] if detections else None,
        "detections": detections,
        "has_hazard": any(d["hazard"] for d in detections),
        "total_val": sum(value_mapping.get(d["class"], 20) for d in detections)
    }

def run_faster_rcnn_inference(frame, conf=0.20):
    if m_frcnn is None:
        return frame, {"error": "Faster R-CNN weights not found"}
    t0 = time.perf_counter()
    orig_h, orig_w, _ = frame.shape
    rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = TF.to_tensor(TF.resize(rgb_pil, [640, 640])).unsqueeze(0)

    with torch.no_grad():
        preds = m_frcnn(tensor)[0]
    latency = (time.perf_counter() - t0) * 1000

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    sx = orig_w / 640.0
    sy = orig_h / 640.0
    annotated = frame.copy()
    detections = []

    for box, label, score in zip(boxes, labels, scores):
        if score < conf:
            continue
        cls_idx = label - 1
        cname = frcnn_classes[cls_idx] if 0 <= cls_idx < len(frcnn_classes) else f"Class-{label}"
        x1, y1 = int(box[0] * sx), int(box[1] * sy)
        x2, y2 = int(box[2] * sx), int(box[3] * sy)
        is_haz = cname in hazard_mapping

        detections.append({
            "class": cname,
            "confidence": int(score * 100),
            "score": float(score),
            "value": f"₹{value_mapping.get(cname, 20)}",
            "hazard": is_haz,
            "hazard_text": hazard_mapping.get(cname, "")
        })

        color = (255, 60, 100) if is_haz else (220, 140, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{cname} {score*100:.0f}%"
        cv2.putText(annotated, caption, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    detections.sort(key=lambda d: d["score"], reverse=True)
    return annotated, {
        "model": "Faster R-CNN",
        "architecture": "Two-Stage ResNet-50-FPN",
        "latency_ms": round(latency, 1),
        "count": len(detections),
        "top": detections[0] if detections else None,
        "detections": detections,
        "has_hazard": any(d["hazard"] for d in detections),
        "total_val": sum(value_mapping.get(d["class"], 20) for d in detections)
    }

# --- Gemini LLM Report Generator (Strictly for Environmental Report, Never Image Classification) ---
def generate_gemini_report_text(detected_classes, hazards):
    if not detected_classes:
        return "No electronic waste detected in the input image."
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    items_str = ", ".join(list(set(detected_classes)))

    if HAS_GENAI and api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
You are an expert Environmental Engineer and certified E-Waste Auditor.
Computer vision object detectors (YOLOv8, YOLOv5, and Faster R-CNN) identified the following hardware items:
Detected Hardware: {items_str}
Hazards: {', '.join(hazards) if hazards else 'None high-risk flagged'}

Generate a professional technical E-Waste Audit & Recycling Directive report in Markdown covering:
1. Executive Summary & Hardware Classification
2. Environmental Toxicity & Heavy Metal Leaching Risks (Lead, Mercury, Cadmium, Lithium)
3. Urban Mining Recovery Yields (Gold, Silver, Copper, Platinum Group Metals)
4. Certified Recycling Roadmap & Safe Handling Directives (RoHS, WEEE Compliance)
"""
            res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            if res and res.text:
                return res.text
        except Exception as e:
            pass

    return f"""
### 🔍 Executive Summary & Hardware Breakdown
Multi-model vision cataloged **{items_str}**. These electronic assemblies contain printed circuits, passive components, and structural housings.

### ☣️ Ecological Hazard & Toxicity Assessment
- **Heavy Metals:** Circuit boards and power stages contain solder traces (Lead/Tin alloys) and brominated flame retardants (BFRs).
- **Leachate Danger:** Unlined landfilling allows acid rain to dissolve heavy metals into municipal aquifers.

### 💎 Urban Mining & Precious Metal Recovery Potential
- **Gold & Copper Yield:** Printed circuit boards hold up to 40-800x the gold concentration of raw ore.
- **Ferrous & Non-Ferrous Metals:** High recoverable quantities of copper cabling and aluminum heat-spreaders.

### 🛠️ Certified Recycling Roadmap
1. Disconnect any battery sources and isolate terminals.
2. Segregate external ABS plastics from populated printed circuit boards.
3. Deliver components exclusively to R2 or e-Stewards certified recycling facilities.
"""

# --- Streamlit UI Layout ---
st.title("♻️ ReciBot: Tri-Model E-Waste AI Dashboard")
st.markdown("**Simultaneous Multi-Model Vision Suite**: Executing **YOLOv8**, **YOLOv5**, and **Faster R-CNN** simultaneously with dedicated weights. Google Gemini LLM generates comprehensive environmental audit reports.")

col_in1, col_in2, col_in3 = st.columns([1, 1, 1])

with col_in1:
    st.subheader("📁 Preset Sample")
    sample_choice = st.selectbox("Select sample hardware:", ["PCB / Motherboard", "Lithium Battery", "Smartphone", "Laptop", "Refrigerator"])

with col_in2:
    st.subheader("📤 File Upload")
    uploaded_file = st.file_uploader("Upload e-waste photo", type=["jpg", "jpeg", "png"])

with col_in3:
    st.subheader("📷 Camera Input")
    camera_file = st.camera_input("Take camera snapshot")

target_img = None
if uploaded_file is not None:
    target_img = Image.open(uploaded_file)
elif camera_file is not None:
    target_img = Image.open(camera_file)
elif sample_choice:
    sample_map = {
        "PCB / Motherboard": "pcb.jpg",
        "Lithium Battery": "battery.jpg",
        "Smartphone": "smartphone.jpg",
        "Laptop": "laptop.jpg",
        "Refrigerator": "refrigerator.jpg"
    }
    s_path = os.path.join(BASE_DIR, "static", "samples", sample_map[sample_choice])
    if os.path.exists(s_path):
        target_img = Image.open(s_path)

if target_img is not None:
    frame_rgb = np.array(target_img.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    st.markdown("---")
    st.subheader("⚡ Simultaneous Multi-Model Dashboard")

    with st.spinner("Executing YOLOv8, YOLOv5, and Faster R-CNN simultaneously..."):
        # Run all 3 models independently
        v8_frame, v8_info = run_yolov8_inference(frame_bgr, conf=0.15)
        v5_frame, v5_info = run_yolov5_inference(frame_bgr, conf=0.15)
        frcnn_frame, frcnn_info = run_faster_rcnn_inference(frame_bgr, conf=0.15)

    # 3 Simultaneous Model Columns
    col1, col2, col3 = st.columns(3)

    # Section 1: YOLOv8
    with col1:
        st.markdown("### 🟢 Model 1: YOLOv8 SOTA")
        st.caption("Anchor-Free C2f Backbone // Decoupled Head")
        st.image(cv2.cvtColor(v8_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Latency", f"{v8_info.get('latency_ms', 0)} ms", f"{v8_info.get('fps', 0)} FPS")
        st.metric("Objects Detected", v8_info.get("count", 0))
        if v8_info.get("top"):
            st.success(f"**Top:** {v8_info['top']['class']} ({v8_info['top']['confidence']}%)")
            st.info(f"**Scrap Value:** ₹{v8_info.get('total_val', 0)}")
            if v8_info.get("has_hazard"):
                st.error("⚠️ Hazardous Material Detected!")
        else:
            st.warning("No objects detected above threshold.")

    # Section 2: YOLOv5
    with col2:
        st.markdown("### 🔵 Model 2: YOLOv5 Baseline")
        st.caption("Anchor-Based C3 Backbone // Coupled Head")
        st.image(cv2.cvtColor(v5_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Latency", f"{v5_info.get('latency_ms', 0)} ms", f"{v5_info.get('fps', 0)} FPS")
        st.metric("Objects Detected", v5_info.get("count", 0))
        if v5_info.get("top"):
            st.success(f"**Top:** {v5_info['top']['class']} ({v5_info['top']['confidence']}%)")
            st.info(f"**Scrap Value:** ₹{v5_info.get('total_val', 0)}")
            if v5_info.get("has_hazard"):
                st.error("⚠️ Hazardous Material Detected!")
        else:
            st.warning("No objects detected above threshold.")

    # Section 3: Faster R-CNN
    with col3:
        st.markdown("### 🟣 Model 3: Faster R-CNN")
        st.caption("Two-Stage ResNet-50-FPN // RPN + RoIAlign")
        st.image(cv2.cvtColor(frcnn_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Latency", f"{frcnn_info.get('latency_ms', 0)} ms", "Two-Stage")
        st.metric("Objects Detected", frcnn_info.get("count", 0))
        if frcnn_info.get("top"):
            st.success(f"**Top:** {frcnn_info['top']['class']} ({frcnn_info['top']['confidence']}%)")
            st.info(f"**Scrap Value:** ₹{frcnn_info.get('total_val', 0)}")
            if frcnn_info.get("has_hazard"):
                st.error("⚠️ Hazardous Material Detected!")
        else:
            st.warning("No objects detected above threshold.")

    # Side-by-Side Comparison Table
    st.markdown("---")
    st.subheader("📊 Architectural Benchmark Comparison Matrix")
    benchmark_data = {
        "Metric": ["Architecture", "Inference Speed (Latency)", "Total Objects Found", "Primary Prediction", "Scrap Recovery Value", "Hazard Status"],
        "YOLOv8 SOTA": [
            "Anchor-Free C2f",
            f"{v8_info.get('latency_ms', 0)} ms",
            v8_info.get("count", 0),
            f"{v8_info['top']['class']} ({v8_info['top']['confidence']}%)" if v8_info.get("top") else "None",
            f"₹{v8_info.get('total_val', 0)}",
            "⚠️ HAZARD" if v8_info.get("has_hazard") else "Secure"
        ],
        "YOLOv5 Baseline": [
            "Anchor-Based C3",
            f"{v8_info.get('latency_ms', 0)} ms",
            v5_info.get("count", 0),
            f"{v5_info['top']['class']} ({v5_info['top']['confidence']}%)" if v5_info.get("top") else "None",
            f"₹{v5_info.get('total_val', 0)}",
            "⚠️ HAZARD" if v5_info.get("has_hazard") else "Secure"
        ],
        "Faster R-CNN": [
            "ResNet-50-FPN + RPN",
            f"{frcnn_info.get('latency_ms', 0)} ms",
            frcnn_info.get("count", 0),
            f"{frcnn_info['top']['class']} ({frcnn_info['top']['confidence']}%)" if frcnn_info.get("top") else "None",
            f"₹{frcnn_info.get('total_val', 0)}",
            "⚠️ HAZARD" if frcnn_info.get("has_hazard") else "Secure"
        ]
    }
    st.table(benchmark_data)

    # Gemini LLM Environmental Report
    st.markdown("---")
    st.subheader("🧠 Google Gemini LLM Environmental & Recycling Audit")
    st.caption("Visual detection was performed exclusively by YOLOv8, YOLOv5, and Faster R-CNN. Gemini LLM generates this material analysis report based on the detected hardware findings.")

    all_found_classes = set()
    for m in [v8_info, v5_info, frcnn_info]:
        for d in m.get("detections", []):
            all_found_classes.add(d["class"])
    
    hazards_list = []
    for m in [v8_info, v5_info, frcnn_info]:
        if m.get("has_hazard") and m.get("top") and m["top"].get("hazard_text"):
            hazards_list.append(m["top"]["hazard_text"])

    report_text = generate_gemini_report_text(list(all_found_classes), list(set(hazards_list)))
    st.markdown(report_text)
