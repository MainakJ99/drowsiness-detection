"""
model.py
========
Shared config, EyeDataset, and DrowsinessCNN.
Imported by both train.py and detect.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "train_dir":          "train",          # folder containing Open_Eyes / Closed_Eyes
    "open_folder":        "Open_Eyes",      # label = alert  (0)
    "closed_folder":      "Closed_Eyes",    # label = drowsy (1)
    "classes":            ["alert", "drowsy"],
    "img_size":           64,
    "batch_size":         32,
    "epochs":             15,
    "lr":                 1e-3,
    "val_split":          0.15,
    "device":             "cuda" if torch.cuda.is_available() else "cpu",
    "model_path":         "drowsiness_model.pth",
    # detection settings
    "drowsy_threshold":   0.60,             # CNN prob above this = drowsy
    "alert_frames_needed": 20,              # consecutive drowsy frames before alarm
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

        print(f"[DATASET] Alert  (open)  : {sum(1 for _,l in self.samples if l==0)}")
        print(f"[DATASET] Drowsy (closed): {sum(1 for _,l in self.samples if l==1)}")
        print(f"[DATASET] Total          : {len(self.samples)}")

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
# SHARED TRANSFORMS
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
