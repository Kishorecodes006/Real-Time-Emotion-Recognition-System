"""
demo_mode.py — Run without a real webcam (uses simulated data)
Useful for testing the UI and logic before connecting a camera.
"""

import cv2
import numpy as np
import time
import math
import random
from collections import deque

FRAME_WIDTH  = 900
FRAME_HEIGHT = 520

EMOTION_COLORS = {
    "happy":    (0,   255, 150),
    "sad":      (100, 100, 255),
    "angry":    (255,  60,  60),
    "fear":     (200, 100, 255),
    "disgust":  (100, 200,  60),
    "surprise": (255, 200,   0),
    "neutral":  (180, 180, 180),
}

COMPLEX_MAP = {
    ("happy", "surprise"): ("JOY",         (0, 255, 150)),
    ("happy", "neutral"):  ("CONTENTMENT", (0, 200, 255)),
    ("sad",   "angry"):    ("FRUSTRATION", (255, 80, 80)),
    ("neutral","happy"):   ("CALM",        (0, 230, 200)),
}

def simulate_emotions(t):
    """Generate smoothly oscillating fake emotion scores."""
    base = {
        "happy":    max(0, 60 + 30 * math.sin(t * 0.4)),
        "surprise": max(0, 12 + 10 * math.sin(t * 0.7 + 1)),
        "neutral":  max(0, 10 + 8  * math.sin(t * 0.3 + 2)),
        "sad":      max(0,  8 + 5  * math.sin(t * 0.5 + 3)),
        "angry":    max(0,  5 + 4  * math.sin(t * 0.6 + 4)),
        "fear":     max(0,  3 + 2  * math.sin(t * 0.9 + 5)),
        "disgust":  max(0,  2 + 1  * math.sin(t * 1.1 + 6)),
    }
    total = sum(base.values()) or 1
    return {k: (v / total) * 100 for k, v in base.items()}

def draw_face(frame, cx, cy, t):
    """Draw an animated cartoon face."""
    r = 90
    # Face
    cv2.circle(frame, (cx, cy), r, (220, 185, 130), -1)
    cv2.circle(frame, (cx, cy), r, (0, 200, 180),  2)
    # Eyes
    blink = abs(math.sin(t * 2)) > 0.95
    eye_h = 2 if blink else 10
    cv2.ellipse(frame, (cx - 28, cy - 22), (10, eye_h), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(frame, (cx + 28, cy - 22), (10, eye_h), 0, 0, 360, (40, 40, 40), -1)
    # Smile
    smile = int(20 + 15 * abs(math.sin(t * 0.4)))
    cv2.ellipse(frame, (cx, cy + 25), (smile, 18), 0, 0, 180, (40, 40, 40), 2)

def draw_bar(frame, label, value, x, y, w=160):
    filled = int(w * value / 100)
    color  = EMOTION_COLORS.get(label, (180,180,180))
    cv2.rectangle(frame, (x, y), (x+w, y+16), (30,30,50), -1)
    if filled:
        cv2.rectangle(frame, (x, y), (x+filled, y+16), color, -1)
    cv2.putText(frame, f"{label.capitalize():<9} {value:5.1f}%",
                (x-150, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

def main():
    print("[DEMO] Running in simulation mode. Press Q to quit.")
    start = time.time()

    while True:
        t = time.time() - start
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        # Background gradient
        for row in range(FRAME_HEIGHT):
            alpha = row / FRAME_HEIGHT
            frame[row, :] = [int(8 + alpha*6), int(10 + alpha*8), int(20 + alpha*15)]

        emotions = simulate_emotions(t)
        sorted_e = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        primary  = sorted_e[0][0]
        secondary= sorted_e[1][0]
        conf     = sorted_e[0][1]

        complex_label = primary.upper()
        complex_color = EMOTION_COLORS.get(primary, (180,180,180))
        for (e1, e2), (lbl, col) in COMPLEX_MAP.items():
            if primary == e1 and secondary == e2:
                complex_label, complex_color = lbl, col

        # Face region
        face_cx, face_cy = 230, 260
        draw_rounded = lambda x1,y1,x2,y2,c,r=10: (
            cv2.rectangle(frame,(x1+r,y1),(x2-r,y2),c,2),
            cv2.rectangle(frame,(x1,y1+r),(x2,y2-r),c,2))

        cv2.rectangle(frame, (60,80),(400,440),(20,25,40),-1)
        draw_face(frame, face_cx, face_cy, t)

        # Green face box
        cv2.rectangle(frame,(100,120),(360,400),complex_color,2)
        cv2.putText(frame, f"{primary.upper()} ({conf:.0f}%)",
                    (105,115),cv2.FONT_HERSHEY_SIMPLEX,0.55,complex_color,2)
        cv2.putText(frame, f"Complex: {complex_label}",
                    (130,420),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,80,180),1)

        # Panel
        px = 430
        cv2.rectangle(frame,(px,0),(FRAME_WIDTH,FRAME_HEIGHT),(10,12,24),-1)
        cv2.putText(frame,"Emotion Analysis",(px+20,35),
                    cv2.FONT_HERSHEY_DUPLEX,0.65,(0,220,255),1)
        cv2.line(frame,(px+10,44),(FRAME_WIDTH-10,44),(0,160,180),1)
        cv2.putText(frame,"Basic Emotions:",(px+15,65),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)

        for i,(e,v) in enumerate(sorted_e[:6]):
            draw_bar(frame, e, v, x=px+170, y=76+i*28, w=140)

        # Complex box
        by = 270
        cv2.rectangle(frame,(px+10,by),(FRAME_WIDTH-10,by+70),complex_color,2)
        cv2.putText(frame,complex_label,(px+25,by+30),
                    cv2.FONT_HERSHEY_DUPLEX,0.8,complex_color,2)
        cv2.putText(frame,f"Confidence: {conf:.0f}%",(px+25,by+58),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)

        stats = [
            f"Face Detected : YES",
            f"Processing    : {random.randint(38,52)}ms",
            f"FPS           : {random.randint(20,25)}",
            f"Model         : DeepFace VGG-Face",
        ]
        for i,s in enumerate(stats):
            cv2.putText(frame,s,(px+15,370+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(140,160,160),1)

        # Title
        cv2.putText(frame,"Real-Time Emotion Recognition System - DEMO",
                    (10,25),cv2.FONT_HERSHEY_DUPLEX,0.55,(0,220,255),1)

        # Status bar
        cv2.rectangle(frame,(0,FRAME_HEIGHT-35),(FRAME_WIDTH,FRAME_HEIGHT),(8,10,18),-1)
        cv2.putText(frame,
                    f"  DEMO MODE  |  Primary: {primary.capitalize()}  |  "
                    f"Secondary: {secondary.capitalize()}  |  Press Q to quit",
                    (10,FRAME_HEIGHT-12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,180),1)

        cv2.imshow("Emotion Recognition - DEMO", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()