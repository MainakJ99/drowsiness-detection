"""
detect.py
=========
Real-time driver drowsiness detection using webcam
and the trained DrowsinessCNN model.

Usage:
    python detect.py

Requirements:
    drowsiness_model.pth must exist (run train.py first)
"""

import cv2
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import CFG, DrowsinessCNN


# ─────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────

class Detector:
    FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    EYE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    def __init__(self):
        if not Path(CFG["model_path"]).exists():
            print(f"[ERROR] Model not found: {CFG['model_path']}")
            print("Run training first:  python train.py")
            exit(1)

        self.device = torch.device(CFG["device"])
        self.model  = DrowsinessCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(CFG["model_path"], map_location=self.device)
        )
        self.model.eval()
        print(f"[DETECT] Model loaded from {CFG['model_path']}")

        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((CFG["img_size"], CFG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.drowsy_cnt = 0

    # ─────────────────────────────────────────
    # CNN PREDICTION
    # ─────────────────────────────────────────

    def predict(self, crop_bgr):
        """Returns drowsy probability (0.0 – 1.0) for a BGR image crop."""
        img    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        return probs[1].item()   # index 1 = drowsy

    # ─────────────────────────────────────────
    # WEBCAM LOOP
    # ─────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[DETECT] Webcam started — press Q to quit\n")

        THRESH  = CFG["drowsy_threshold"]
        MAX_CNT = CFG["alert_frames_needed"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pre-process for face detection
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = self.FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            is_drowsy  = False
            eyes_found = False
            prob_val   = 0.0

            for (fx, fy, fw, fh) in faces:

                # Draw face bounding box
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 180, 0), 2)

                # Search for eyes in upper 60% of face only
                roi_gray = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
                roi_bgr  = frame[fy : fy + int(fh * 0.6), fx : fx + fw]

                eyes = self.EYE_CASCADE.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
                )

                if len(eyes) > 0:
                    eyes_found = True
                    for (ex, ey, ew, eh) in eyes[:2]:
                        crop = roi_bgr[ey : ey + eh, ex : ex + ew]
                        if crop.size == 0:
                            continue

                        prob     = self.predict(crop)
                        prob_val = prob
                        drowsy   = prob > THRESH
                        color    = (0, 0, 255) if drowsy else (0, 220, 0)
                        tag      = f"DROWSY {prob:.2f}" if drowsy else f"ALERT {1-prob:.2f}"

                        cv2.rectangle(frame,
                                      (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), color, 2)
                        cv2.putText(frame, tag, (fx+ex, fy+ey-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        if drowsy:
                            is_drowsy = True

                else:
                    # Fallback: run CNN on upper-face crop when eyes not detected
                    upper = frame[fy : fy + int(fh * 0.55), fx : fx + fw]
                    if upper.size > 0:
                        prob     = self.predict(upper)
                        prob_val = prob
                        if prob > THRESH:
                            is_drowsy = True
                        label_txt = f"face-crop {'DROWSY' if prob > THRESH else 'ALERT'} {prob:.2f}"
                        cv2.putText(frame, label_txt, (fx, fy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 0, 255) if prob > THRESH else (0, 220, 0), 1)

            # ── Update consecutive drowsy frame counter ──
            self.drowsy_cnt = (self.drowsy_cnt + 1) if is_drowsy else max(0, self.drowsy_cnt - 1)

            # ── Top status bar ──
            H, W = frame.shape[:2]

            if self.drowsy_cnt >= MAX_CNT:
                cv2.rectangle(frame, (0, 0), (W, 55), (0, 0, 180), -1)
                cv2.putText(frame, "  DROWSINESS ALERT!  WAKE UP!",
                            (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)

            elif len(faces) > 0:
                bar_col = (0, 160, 0) if not is_drowsy else (0, 80, 200)
                cv2.rectangle(frame, (0, 0), (W, 42), bar_col, -1)
                status = "ALERT" if not is_drowsy else "DROWSY..."
                cv2.putText(frame,
                            f"{status}   prob={prob_val:.2f}   cnt={self.drowsy_cnt}/{MAX_CNT}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            else:
                cv2.rectangle(frame, (0, 0), (W, 42), (50, 50, 50), -1)
                cv2.putText(frame, "No face detected — sit closer & improve lighting",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # ── Bottom info bar ──
            cv2.rectangle(frame, (0, H - 30), (W, H), (20, 20, 20), -1)
            eye_info = "Eyes: detected" if eyes_found else "Eyes: not found (face-crop mode)"
            cv2.putText(frame, eye_info + "   |   Q = quit",
                        (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[DETECT] Stopped.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    Detector().run()
