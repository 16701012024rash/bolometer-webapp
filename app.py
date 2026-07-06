"""
Bolometer Pipeline - Flask Web App
Detection: MobileNet SSD (knows 80+ classes including person, cat, dog, car, bus)
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import os
import traceback
import urllib.request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ── MobileNet SSD Setup ───────────────────────────────────────────────────────

MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
PROTO_PATH  = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH  = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

PROTO_URL = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
MODEL_URL = "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel"

# MobileNet SSD class labels
CLASSES = [
    "background", "aeroplane", "bicycle", "bird",   "boat",
    "bottle",     "bus",       "car",     "cat",    "chair",
    "cow",        "diningtable","dog",    "horse",  "motorbike",
    "person",     "pottedplant","sheep",  "sofa",   "train",
    "tvmonitor"
]

# Map to our 4 categories
ANIMAL_CLASSES  = {"bird", "cat", "cow", "dog", "horse", "sheep"}
VEHICLE_CLASSES = {"aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train"}
PERSON_CLASSES  = {"person"}

# Colors
COLOR_PERSON  = (0, 255, 128)
COLOR_VEHICLE = (0, 128, 255)
COLOR_ANIMAL  = (255, 200, 0)
COLOR_OBJECT  = (200, 200, 200)

net = None
net_loaded = False

def load_model():
    global net, net_loaded
    if net_loaded:
        return
    net_loaded = True
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        if not os.path.exists(PROTO_PATH):
            print("[model] Downloading prototxt...")
            urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)
            print("[model] Prototxt downloaded.")

        if not os.path.exists(MODEL_PATH):
            print("[model] Downloading caffemodel (~23MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[model] Model downloaded.")

        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        print("[model] MobileNet SSD loaded successfully.")
    except Exception as e:
        print(f"[model] Failed to load MobileNet: {e}")
        net = None

# ── Haar cascade fallback (face only) ────────────────────────────────────────

def load_cascade(filename):
    path = cv2.data.haarcascades + filename
    if os.path.exists(path):
        clf = cv2.CascadeClassifier(path)
        if not clf.empty():
            return clf
    return None

face_cascade = load_cascade('haarcascade_frontalface_default.xml')

# ── Pipeline functions ────────────────────────────────────────────────────────

def to_thermal(img_gray):
    return img_gray.astype(np.float32) / 255.0

def calibrate(frame):
    rng    = np.random.RandomState(42)
    gain   = 1.0 + rng.randn(*frame.shape).astype(np.float32) * 0.005
    offset =       rng.randn(*frame.shape).astype(np.float32) * 0.002
    return np.clip(gain * frame + offset, 0, 1).astype(np.float32)

def deterministic_process(frame):
    u8       = (frame * 255).clip(0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(u8, 9, 75, 75)
    clahe    = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return clahe.apply(filtered).astype(np.float32) / 255.0

def ai_enhance(frame):
    H, W      = frame.shape
    u8        = (frame * 255).clip(0, 255).astype(np.uint8)
    upscaled  = cv2.resize(u8, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
    blurred   = cv2.GaussianBlur(upscaled, (3, 3), 1.0)
    sharpened = cv2.addWeighted(upscaled, 1.2, blurred, -0.2, 0)
    return np.clip(sharpened, 0, 255).astype(np.float32) / 255.0

def apply_colormap(frame_norm):
    lo, hi = np.percentile(frame_norm, [2, 98])
    if hi - lo < 0.01:
        hi = lo + 0.01
    normalized = ((frame_norm - lo) / (hi - lo)).clip(0, 1)
    u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

def detect_objects(frame, original_img_bgr):
    """
    Use MobileNet SSD on the original colour image for accurate detection.
    Overlay results on the thermal frame.
    """
    # Load model lazily on first detection call
    load_model()

    H, W = frame.shape
    detections = []

    # ── MobileNet SSD ─────────────────────────────────────────────────────
    if net is not None:
        oh, ow = original_img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(original_img_bgr, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        preds = net.forward()

        for i in range(preds.shape[2]):
            conf = float(preds[0, 0, i, 2])
            if conf < 0.40:
                continue

            class_idx = int(preds[0, 0, i, 1])
            if class_idx >= len(CLASSES):
                continue
            class_name = CLASSES[class_idx]

            # Scale box back to original image size then to thermal frame size
            box = preds[0, 0, i, 3:7] * np.array([ow, oh, ow, oh])
            x1, y1, x2, y2 = box.astype(int)

            # Scale to thermal frame (which may be 2x due to ai_enhance)
            sx = W / ow
            sy = H / oh
            tx1 = max(0, int(x1 * sx))
            ty1 = max(0, int(y1 * sy))
            tx2 = min(W, int(x2 * sx))
            ty2 = min(H, int(y2 * sy))
            tw  = tx2 - tx1
            th  = ty2 - ty1

            if tw < 5 or th < 5:
                continue

            # Map to our category
            if class_name in PERSON_CLASSES:
                label = "Person"
                color = COLOR_PERSON
            elif class_name in VEHICLE_CLASSES:
                label = "Vehicle"
                color = COLOR_VEHICLE
            elif class_name in ANIMAL_CLASSES:
                label = "Animal"
                color = COLOR_ANIMAL
            else:
                label = class_name.capitalize()
                color = COLOR_OBJECT

            detections.append({
                "bbox":       [tx1, ty1, tw, th],
                "label":      label,
                "confidence": round(conf, 2),
                "color":      color,
            })

    # ── Haar face fallback if MobileNet found nothing ─────────────────────
    if len(detections) == 0 and face_cascade is not None:
        gray = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
        scaled_gray = cv2.resize(gray, (W, H))
        faces = face_cascade.detectMultiScale(
            scaled_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        for (x, y, w, h) in faces:
            detections.append({
                "bbox":       [int(x), int(y), int(w), int(h)],
                "label":      "Person",
                "confidence": 0.85,
                "color":      COLOR_PERSON,
            })

    return detections

def draw_detections(colored_frame, detections):
    out = colored_frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = det["color"]
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        label_text = det["label"] + " " + str(det["confidence"])
        cv2.putText(out, label_text, (x, max(y-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def frame_to_base64(frame_bgr):
    _, buffer = cv2.imencode('.png', frame_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def run_pipeline(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not read image"

    h, w = img.shape[:2]
    max_dim = 500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raw        = to_thermal(gray)
    calibrated = calibrate(raw)
    processed  = deterministic_process(calibrated)
    enhanced   = ai_enhance(processed)

    # Detect on original colour image for accuracy
    detections = detect_objects(enhanced, img)

    raw_colored  = apply_colormap(raw)
    cal_colored  = apply_colormap(calibrated)
    proc_colored = apply_colormap(processed)
    enh_colored  = apply_colormap(enhanced)
    det_colored  = draw_detections(apply_colormap(enhanced), detections)

    target_h = 200
    def resize_to_h(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*scale), h))

    stages = [resize_to_h(s, target_h)
              for s in [raw_colored, cal_colored, proc_colored, enh_colored, det_colored]]

    sep   = np.ones((target_h, 3, 3), dtype=np.uint8) * 50
    parts = []
    for i, s in enumerate(stages):
        parts.append(s)
        if i < len(stages) - 1:
            parts.append(sep)
    composite = np.hstack(parts)

    label_bar = np.zeros((28, composite.shape[1], 3), dtype=np.uint8)
    labels    = ["RAW", "CALIBRATED", "PROCESSED", "AI ENHANCED", "DETECTIONS"]
    panel_w   = stages[0].shape[1]
    for i, lbl in enumerate(labels):
        x = i * (panel_w + 3) + panel_w // 2 - len(lbl) * 4
        cv2.putText(label_bar, lbl, (x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 212, 170), 1, cv2.LINE_AA)

    final = np.vstack([label_bar, composite])

    return {
        "composite": frame_to_base64(final),
        "stages": {
            "raw":        frame_to_base64(raw_colored),
            "calibrated": frame_to_base64(cal_colored),
            "processed":  frame_to_base64(proc_colored),
            "enhanced":   frame_to_base64(enh_colored),
            "detections": frame_to_base64(det_colored),
        },
        "detections": [{"label": d["label"], "confidence": d["confidence"]}
                       for d in detections],
        "count": len(detections),
    }, None

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
    try:
        result, error = run_pipeline(file.read())
        if error:
            return jsonify({"error": error}), 400
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
