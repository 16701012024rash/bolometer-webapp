"""
Bolometer Pipeline - Flask Web App
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def to_thermal(img_gray):
    return img_gray.astype(np.float32) / 255.0

def calibrate(frame):
    gain = 1.0 + np.random.randn(*frame.shape) * 0.01
    offset = np.random.randn(*frame.shape) * 0.005
    calibrated = gain * frame + offset
    return np.clip(calibrated, 0, 1).astype(np.float32)

def deterministic_process(frame):
    u8 = (frame * 255).clip(0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(u8, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    return enhanced.astype(np.float32) / 255.0

def ai_enhance(frame):
    H, W = frame.shape
    u8 = (frame * 255).clip(0, 255).astype(np.uint8)
    upscaled = cv2.resize(u8, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 1.0)
    sharpened = cv2.addWeighted(upscaled, 1.3, blurred, -0.3, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    result = clahe.apply(sharpened)
    return result.astype(np.float32) / 255.0

def detect_objects(frame):
    u8 = (frame * 255).clip(0, 255).astype(np.uint8)
    H, W = u8.shape
    detections = []
    all_bboxes = []
    all_confs = []

    mean_val = float(u8.mean())

    for thresh_pct in [1.20, 1.30]:
        thresh_val = min(int(mean_val * thresh_pct), 254)
        _, binary = cv2.threshold(u8, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w > W * 0.85 or h > H * 0.85:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            aspect = w / (h + 1e-6)

            if solidity < 0.3 or aspect > 10:
                continue

            roi = frame[y:y+h, x:x+w]
            thermal_contrast = (roi.mean() - frame.mean()) / (frame.mean() + 1e-6)
            conf = float(np.clip(solidity * 0.5 + thermal_contrast * 0.5, 0, 1))

            if conf < 0.35:
                continue

            if aspect < 0.85 and area > 2000:
                label = "Person"
                color = (0, 255, 128)
            elif aspect > 1.5 and area > 5000:
                label = "Vehicle"
                color = (0, 128, 255)
            elif 1000 < area < 4000:
                label = "Animal"
                color = (255, 200, 0)
            else:
                label = "Object"
                color = (200, 0, 255)

            all_bboxes.append([x, y, w, h])
            all_confs.append(conf)
            detections.append({
                "bbox": [x, y, w, h],
                "label": label,
                "confidence": round(conf, 2),
                "color": color
            })

    if all_bboxes:
        indices = cv2.dnn.NMSBoxes(all_bboxes, all_confs, 0.35, 0.3)
        if len(indices) > 0:
            keep = set(indices.flatten())
            detections = [d for i, d in enumerate(detections) if i in keep]

    return detections

def apply_colormap(frame_norm):
    u8 = (frame_norm * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

def draw_detections(colored_frame, detections):
    out = colored_frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        color = det["color"]
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        cv2.putText(out, label + " " + str(conf), (x, max(y-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def frame_to_base64(frame_bgr):
    _, buffer = cv2.imencode('.png', frame_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def run_pipeline(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Could not read image"

    h, w = img.shape[:2]
    max_dim = 400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raw = to_thermal(gray)
    raw_colored = apply_colormap(raw)

    calibrated = calibrate(raw)
    cal_colored = apply_colormap(calibrated)

    processed = deterministic_process(calibrated)
    proc_colored = apply_colormap(processed)

    enhanced = ai_enhance(processed)
    enh_colored = apply_colormap(enhanced)

    detections = detect_objects(enhanced)
    det_colored = draw_detections(enh_colored.copy(), detections)

    target_h = 200
    def resize_to_h(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*scale), h))

    stages = [raw_colored, cal_colored, proc_colored, enh_colored, det_colored]
    stages = [resize_to_h(s, target_h) for s in stages]

    separator = np.ones((target_h, 3, 3), dtype=np.uint8) * 50
    composite_parts = []
    for i, s in enumerate(stages):
        composite_parts.append(s)
        if i < len(stages) - 1:
            composite_parts.append(separator)
    composite = np.hstack(composite_parts)

    label_bar = np.zeros((28, composite.shape[1], 3), dtype=np.uint8)
    labels = ["RAW", "CALIBRATED", "PROCESSED", "AI ENHANCED", "DETECTIONS"]
    panel_w = stages[0].shape[1]
    for i, label in enumerate(labels):
        x = i * (panel_w + 3) + panel_w // 2 - len(label) * 4
        cv2.putText(label_bar, label, (x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 212, 170), 1, cv2.LINE_AA)

    final = np.vstack([label_bar, composite])

    result = {
        "composite": frame_to_base64(final),
        "stages": {
            "raw": frame_to_base64(raw_colored),
            "calibrated": frame_to_base64(cal_colored),
            "processed": frame_to_base64(proc_colored),
            "enhanced": frame_to_base64(enh_colored),
            "detections": frame_to_base64(det_colored),
        },
        "detections": [
            {"label": d["label"], "confidence": d["confidence"]}
            for d in detections
        ],
        "count": len(detections)
    }
    return result, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    image_bytes = file.read()
    result, error = run_pipeline(image_bytes)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)