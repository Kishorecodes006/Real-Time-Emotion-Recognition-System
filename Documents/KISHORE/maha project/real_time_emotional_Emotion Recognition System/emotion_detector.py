"""
Real-Time Emotion Recognition System
=====================================
College Project - Deep Learning with OpenCV & DeepFace
Author: [Your Name]
Date: 2026
"""

import cv2
import numpy as np
import time
import json
from deepface import DeepFace
from collections import deque

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
ANALYZE_EVERY_N_FRAMES = 5   # Analyze every 5th frame for performance
SMOOTHING_WINDOW = 10        # Rolling average over last N readings

# Basic → Complex emotion mapping (confidence-based)
COMPLEX_EMOTION_MAP = {
    ("happy", "surprise"): ("JOY",        (0, 255, 150)),
    ("happy", "neutral"):  ("CONTENTMENT",(0, 200, 255)),
    ("sad",   "angry"):    ("FRUSTRATION",(255, 80,  80)),
    ("sad",   "neutral"):  ("MELANCHOLY", (120, 80, 255)),
    ("angry", "disgust"):  ("CONTEMPT",   (255, 60, 120)),
    ("angry", "sad"):      ("RESENTMENT", (200, 60,  60)),
    ("fear",  "surprise"): ("AWE",        (255, 200,  0)),
    ("fear",  "sad"):      ("DESPAIR",    (100, 80, 200)),
    ("neutral","happy"):   ("CALM",       (0, 230, 200)),
}

EMOTION_COLORS = {
    "happy":    (0,   255, 150),
    "sad":      (100, 100, 255),
    "angry":    (255,  60,  60),
    "fear":     (200, 100, 255),
    "disgust":  (100, 200,  60),
    "surprise": (255, 200,   0),
    "neutral":  (180, 180, 180),
}

# ─── Smoothing Buffer ──────────────────────────────────────────────────────────
class EmotionSmoother:
    def __init__(self, window=SMOOTHING_WINDOW):
        self.buffers = {e: deque(maxlen=window) for e in EMOTION_COLORS}

    def update(self, emotion_scores: dict):
        for emotion, score in emotion_scores.items():
            key = emotion.lower()
            if key in self.buffers:
                self.buffers[key].append(score)

    def get_smoothed(self) -> dict:
        return {
            e: (sum(buf) / len(buf) if buf else 0.0)
            for e, buf in self.buffers.items()
        }

# ─── Complex Emotion Resolver ─────────────────────────────────────────────────
def resolve_complex_emotion(smoothed: dict):
    sorted_emotions = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
    if not sorted_emotions:
        return "UNKNOWN", 0, (180, 180, 180)

    primary   = sorted_emotions[0][0]
    secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else ""
    confidence = sorted_emotions[0][1]

    for (e1, e2), (label, color) in COMPLEX_EMOTION_MAP.items():
        if primary == e1 and secondary == e2:
            return label, confidence, color

    # Fallback to dominant emotion
    label = primary.upper()
    color = EMOTION_COLORS.get(primary, (180, 180, 180))
    return label, confidence, color

# ─── Drawing Utilities ────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=12, thickness=2):
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0, 0, 90, color, thickness)

def draw_emotion_bar(img, label, value, x, y, width=200, max_val=100):
    bar_h  = 18
    filled = int(width * value / max_val)
    color  = EMOTION_COLORS.get(label.lower(), (180, 180, 180))

    # Background bar
    cv2.rectangle(img, (x, y), (x + width, y + bar_h), (40, 40, 60), -1)
    # Filled portion
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + bar_h), color, -1)
    # Label
    cv2.putText(img, f"{label.capitalize()}: {value:.0f}%",
                (x - 120, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

def overlay_hud(frame, smoothed, complex_emotion, complex_conf, complex_color,
                faces, fps, processing_ms):
    h, w = frame.shape[:2]

    # ── Dark side panel ──────────────────────────────────────────────────────
    panel_x = w - 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (8, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(frame, "EMOTION ANALYSIS", (panel_x + 15, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1)
    cv2.line(frame, (panel_x + 10, 38), (w - 10, 38), (0, 180, 200), 1)

    # ── Emotion bars ──────────────────────────────────────────────────────────
    sorted_emotions = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
    for i, (emotion, score) in enumerate(sorted_emotions[:6]):
        y_pos = 60 + i * 30
        draw_emotion_bar(frame, emotion, score,
                         x=panel_x + 130, y=y_pos, width=120)

    # ── Complex emotion box ───────────────────────────────────────────────────
    box_y = 250
    draw_rounded_rect(frame, panel_x + 10, box_y, w - 10, box_y + 70,
                      complex_color, radius=8, thickness=2)
    cv2.putText(frame, complex_emotion,
                (panel_x + 20, box_y + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, complex_color, 2)
    cv2.putText(frame, f"Confidence: {complex_conf:.0f}%",
                (panel_x + 20, box_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = [
        f"Faces Detected : {len(faces)}",
        f"Processing Time: {processing_ms:.0f}ms",
        f"FPS            : {fps:.1f}",
        f"Model          : DeepFace VGG",
    ]
    for i, stat in enumerate(stats):
        cv2.putText(frame, stat, (panel_x + 12, 345 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 180), 1)

    # ── Face bounding boxes ───────────────────────────────────────────────────
    for face in faces:
        x, y, fw, fh = face.get("region", {}).get("x", 0), \
                        face.get("region", {}).get("y", 0), \
                        face.get("region", {}).get("w", 0), \
                        face.get("region", {}).get("h", 0)
        if fw and fh:
            draw_rounded_rect(frame, x, y, x + fw, y + fh,
                              complex_color, radius=10, thickness=2)
            dominant = face.get("dominant_emotion", "").upper()
            cv2.putText(frame, dominant, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, complex_color, 2)

    return frame

# ─── Main Loop ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    smoother       = EmotionSmoother()
    frame_count    = 0
    fps            = 0.0
    processing_ms  = 0.0
    last_time      = time.time()
    cached_faces   = []
    complex_emotion = "DETECTING…"
    complex_conf    = 0.0
    complex_color   = (180, 180, 180)

    print("[INFO] Starting Real-Time Emotion Recognition…")
    print("[INFO] Press 'q' to quit | 's' to save screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not accessible.")
            break

        frame_count += 1

        # ── FPS calculation ────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / (now - last_time + 1e-9)
        last_time = now

        # ── Emotion analysis (every N frames) ─────────────────────────────────
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            t0 = time.time()
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                cached_faces = results if isinstance(results, list) else [results]

                # Aggregate scores across all faces
                combined = {e: 0.0 for e in EMOTION_COLORS}
                for face in cached_faces:
                    for emotion, score in face.get("emotion", {}).items():
                        key = emotion.lower()
                        if key in combined:
                            combined[key] += score

                smoother.update(combined)
                smoothed = smoother.get_smoothed()
                complex_emotion, complex_conf, complex_color = \
                    resolve_complex_emotion(smoothed)

            except Exception as e:
                print(f"[WARN] Analysis error: {e}")

            processing_ms = (time.time() - t0) * 1000

        # ── Draw HUD ───────────────────────────────────────────────────────────
        smoothed = smoother.get_smoothed()
        frame = overlay_hud(frame, smoothed, complex_emotion, complex_conf,
                            complex_color, cached_faces, fps, processing_ms)

        # ── Window title bar ───────────────────────────────────────────────────
        cv2.putText(frame, "Real-Time Emotion Recognition System",
                    (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 255), 1)
        cv2.putText(frame, "Press Q to quit | S to screenshot",
                    (10, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imshow("Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

if __name__ == "__main__":
    main()