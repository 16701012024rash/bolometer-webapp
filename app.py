"""
Bolometer Pipeline - Flask Web App
Upgraded with Haar Cascade face + body detection
Fixed: proper colormap normalization for all image types
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import os
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

LOCAL_CASCADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades")

def load_cascade(filename):
    candidates = [
        os.path.join(LOCAL_CASCADE_DIR, filename),
        cv2.data.haarcascades + filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                return clf
    print(f"[cascade] not found: {filename}")
    return None

face_cascade       = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade        = load_cascade('haarcascade_eye.xml')
body_cascade       = load_cascade('haarcascade_fullbody.xml')
upper_body_cascade = load_cascade('haarcascade_upperbody.xml')
car_cascade        = load_cascade('haarcascade_car.xml')


def to_thermal(img_gray):
    return img_gray.astype(np.float32) / 255.0


def calibrate(frame):
    rng = np.random.RandomState(42)
    gain   = 1.0 + rng.randn(*frame.shape).astype(np.float32) * 0.005
    offset =       rng.randn(*frame.shape).astype(np.float32) * 0.002
    return np.clip(gain * frame + offset, 0, 1).astype(np.float32)


def deterministic_process(frame):
    u8 = (frame * 255).clip(0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(u8, 9, 75, 75)
    clahe    = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    return enhanced.astype(np.float32) / 255.0


def ai_enhance(frame):
    H, W = frame.shape
    u8       = (frame * 255).clip(0, 255).astype(np.uint8)
    upscaled = cv2.resize(u8, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
    blurred  = cv2.GaussianBlur(upscaled, (3, 3), 1.0)
    sharpened = cv2.addWeighted(upscaled, 1.2, blurred, -0.2, 0)
    return np.clip(sharpened, 0, 255).astype(np.float32) / 255.0


def apply_colormap(frame_norm):
    """Percentile normalisation so both dark AND bright images render correctly."""
    lo, hi = np.percentile(frame_norm, [2, 98])
    if hi - lo < 0.01:
        hi = lo + 0.01
    normalized = ((frame_norm - lo) / (hi - lo)).clip(0, 1)
    u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def detect_objects(frame, original_gray):
    H, W = frame.shape
    detections      = []
    detected_regions = []

    scaled_gray = cv2.resize(original_gray, (W, H))

    def overlaps(x, y, w, h):
        for (fx, fy, fw, fh) in detected_regions:
            if abs(x - fx) < fw and abs(y - fy) < fh:
                return True
        return False

    # 1. Face
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(
            scaled_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            detections.append({"bbox": [int(x), int(y), int(w), int(h)],
                                "label": "Person", "confidence": 0.92,
                                "color": (0, 255, 128)})
            detected_regions.append((x, y, w, h))

    # 2. Upper body
    if upper_body_cascade is not None:
        for (x, y, w, h) in upper_body_cascade.detectMultiScale(
                scaled_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)):
            if not overlaps(x, y, w, h):
                detections.append({"bbox": [int(x), int(y), int(w), int(h)],
                                    "label": "Person", "confidence": 0.82,
                                    "color": (0, 255, 128)})
                detected_regions.append((x, y, w, h))

    # 3. Full body
    if body_cascade is not None:
        for (x, y, w, h) in body_cascade.detectMultiScale(
                scaled_gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 60)):
            if not overlaps(x, y, w, h):
                detections.append({"bbox": [int(x), int(y), int(w), int(h)],
                                    "label": "Person", "confidence": 0.78,
                                    "color": (0, 255, 128)})
                detected_regions.append((x, y, w, h))

    # 4. Car
    if car_cascade is not None:
        for (x, y, w, h) in car_cascade.detectMultiScale(
                scaled_gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)):
            detections.append({"bbox": [int(x), int(y), int(w), int(h)],
                                "label": "Vehicle", "confidence": 0.80,
                                "color": (0, 128, 255)})

    # 5. Thermal contour fallback
    if len(detections) == 0:
        u8 = (frame * 255).clip(0, 255).astype(np.uint8)
        mean_val  = float(u8.mean())
        thresh_val = min(int(mean_val * 1.25), 254)
        _, binary  = cv2.threshold(u8, thresh_val, 255, cv2.THRESH_BINARY)
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary     = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w > W * 0.8 or h > H * 0.8:
                continue
            detections.append({"bbox": [int(x), int(y), int(w), int(h)],
                                "label": "Object", "confidence": 0.50,
                                "color": (200, 200, 0)})

    return detections


def draw_detections(colored_frame, detections):
    out = colored_frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = det["color"]
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        cv2.putText(out, det["label"] + " " + str(det["confidence"]),
                    (x, max(y-6, 12)),
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
    detections = detect_objects(enhanced, gray)

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

    sep = np.ones((target_h, 3, 3), dtype=np.uint8) * 50
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