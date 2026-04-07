"""
Driver Drowsiness Detection
============================
Folder structure (your setup):
    train/
        Open_Eyes/     → alert  (label 0)
        Closed_Eyes/   → drowsy (label 1)

Steps:
    1. Train:   python drowsiness.py --mode train
    2. Detect:  python drowsiness.py --mode detect
"""

import os, time, cv2, torch, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG  — edit paths here if needed
# ─────────────────────────────────────────────
CFG = {
    "train_dir":   "train",          # folder containing Open_Eyes / Closed_Eyes
    "open_folder": "Open_Eyes",      # label = alert  (0)
    "closed_folder": "Closed_Eyes",  # label = drowsy (1)
    "classes":     ["alert", "drowsy"],
    "img_size":    64,
    "batch_size":  32,
    "epochs":      15,
    "lr":          1e-3,
    "val_split":   0.15,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "model_path":  "drowsiness_model.pth",
    # detection settings
    "drowsy_threshold":    0.60,   # CNN prob above this = drowsy
    "alert_frames_needed": 20,     # consecutive drowsy frames before alarm
}

print(f"[INFO] Device: {CFG['device']}")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class EyeDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, train_dir, transform=None):
        self.transform = transform
        self.samples   = []   # (path, label)

        open_dir   = Path(train_dir) / CFG["open_folder"]
        closed_dir = Path(train_dir) / CFG["closed_folder"]

        for path in open_dir.rglob("*"):
            if path.suffix.lower() in self.IMG_EXTS:
                self.samples.append((str(path), 0))   # 0 = alert

        for path in closed_dir.rglob("*"):
            if path.suffix.lower() in self.IMG_EXTS:
                self.samples.append((str(path), 1))   # 1 = drowsy

        print(f"[DATASET] Alert (open) : {sum(1 for _,l in self.samples if l==0)}")
        print(f"[DATASET] Drowsy (closed): {sum(1 for _,l in self.samples if l==1)}")
        print(f"[DATASET] Total         : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")   # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class DrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return train_tf, val_tf


def train():
    train_tf, val_tf = get_transforms()

    full_ds = EyeDataset(CFG["train_dir"], transform=train_tf)
    n_val   = int(len(full_ds) * CFG["val_split"])
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # apply val transform to val set
    val_ds.dataset = EyeDataset(CFG["train_dir"], transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=0)

    device    = torch.device(CFG["device"])
    model     = DrowsinessCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*50}\n{'TRAINING':^50}\n{'='*50}")
    print(f"Train: {n_train}  |  Val: {n_val}\n")

    for epoch in range(1, CFG["epochs"] + 1):
        # ── train ──
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == labels).sum().item()
            tt += imgs.size(0)

        # ── validate ──
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                vl += loss.item() * imgs.size(0)
                vc += (out.argmax(1) == labels).sum().item()
                vt += imgs.size(0)

        tr_loss, tr_acc = tl/tt, tc/tt
        vl_loss, vl_acc = vl/vt, vc/vt
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        flag = " ← saved" if vl_acc > best_val_acc else ""
        print(f"Epoch {epoch:02d}/{CFG['epochs']}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}{flag}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CFG["model_path"])

    # ── plots ──
    _plot_history(history)

    # ── classification report on val set ──
    model.load_state_dict(torch.load(CFG["model_path"], map_location=device))
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_p.extend(preds.tolist())
            all_l.extend(labels.tolist())

    print("\n[REPORT]\n", classification_report(all_l, all_p,
                                                target_names=CFG["classes"]))
    _plot_confusion(all_l, all_p)
    print(f"\n✅  Model saved → {CFG['model_path']}")
    print("Next: python drowsiness.py --mode detect")


def _plot_history(h):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(h["train_loss"]) + 1)
    axes[0].plot(ep, h["train_loss"], label="Train"); axes[0].plot(ep, h["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, h["train_acc"], label="Train"); axes[1].plot(ep, h["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
    print("[PLOT] training_curves.png saved")


def _plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CFG["classes"], yticklabels=CFG["classes"])
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)
    print("[PLOT] confusion_matrix.png saved")


# ─────────────────────────────────────────────
# REAL-TIME DETECTION
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
            print("Run training first:  python drowsiness.py --mode train")
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

    def predict(self, crop_bgr):
        """Returns drowsy probability (0-1) for a BGR image crop."""
        img    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        return probs[1].item()   # drowsy prob

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[DETECT] Webcam started — press Q to quit\n")

        THRESH  = CFG["drowsy_threshold"]
        MAX_CNT = CFG["alert_frames_needed"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = self.FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            is_drowsy  = False
            eyes_found = False
            prob_val   = 0.0

            for (fx, fy, fw, fh) in faces:
                # Draw face box
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 180, 0), 2)

                # Look for eyes in upper 60% of face only
                roi_gray = gray[fy : fy + int(fh*0.6), fx : fx+fw]
                roi_bgr  = frame[fy : fy + int(fh*0.6), fx : fx+fw]

                eyes = self.EYE_CASCADE.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
                )

                if len(eyes) > 0:
                    eyes_found = True
                    for (ex, ey, ew, eh) in eyes[:2]:
                        crop = roi_bgr[ey:ey+eh, ex:ex+ew]
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
                    # Fallback: feed upper-face crop into CNN
                    upper = frame[fy : fy + int(fh*0.55), fx : fx+fw]
                    if upper.size > 0:
                        prob     = self.predict(upper)
                        prob_val = prob
                        if prob > THRESH:
                            is_drowsy = True
                        cv2.putText(frame,
                                    f"face-crop {'DROWSY' if prob>THRESH else 'ALERT'} {prob:.2f}",
                                    (fx, fy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0,0,255) if prob>THRESH else (0,220,0), 1)

            # ── Counter ──
            self.drowsy_cnt = self.drowsy_cnt + 1 if is_drowsy else max(0, self.drowsy_cnt - 1)

            # ── Top status bar ──
            H, W = frame.shape[:2]
            if self.drowsy_cnt >= MAX_CNT:
                cv2.rectangle(frame, (0, 0), (W, 55), (0, 0, 180), -1)
                cv2.putText(frame, "  DROWSINESS ALERT!  WAKE UP!",
                            (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)

            elif len(faces) > 0:
                col = (0, 160, 0) if not is_drowsy else (0, 80, 200)
                cv2.rectangle(frame, (0, 0), (W, 42), col, -1)
                status = "ALERT" if not is_drowsy else "DROWSY..."
                cv2.putText(frame,
                            f"{status}   prob={prob_val:.2f}   cnt={self.drowsy_cnt}/{MAX_CNT}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            else:
                cv2.rectangle(frame, (0, 0), (W, 42), (50, 50, 50), -1)
                cv2.putText(frame, "No face detected — sit closer & improve lighting",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            # ── Bottom info bar ──
            cv2.rectangle(frame, (0, H-30), (W, H), (20, 20, 20), -1)
            eye_info = "Eyes: detected" if eyes_found else "Eyes: not found (face-crop mode)"
            cv2.putText(frame, eye_info + "   |   Q = quit",
                        (10, H-9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170,170,170), 1)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "detect"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        Detector().run()
