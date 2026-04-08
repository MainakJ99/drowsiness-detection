py"""
train.py
========
Trains the DrowsinessCNN on your eye dataset.

Usage:
    python train.py

Outputs:
    drowsiness_model.pth   ← best checkpoint
    training_curves.png    ← loss / accuracy plots
    confusion_matrix.png   ← val-set evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import CFG, EyeDataset, DrowsinessCNN, get_transforms


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────

def plot_history(h):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(h["train_loss"]) + 1)

    axes[0].plot(ep, h["train_loss"], label="Train")
    axes[0].plot(ep, h["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, h["train_acc"], label="Train")
    axes[1].plot(ep, h["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("[PLOT] training_curves.png saved")


def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CFG["classes"], yticklabels=CFG["classes"])
    plt.title("Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[PLOT] confusion_matrix.png saved")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    train_tf, val_tf = get_transforms()

    # Build datasets
    full_ds = EyeDataset(CFG["train_dir"], transform=train_tf)
    n_val   = int(len(full_ds) * CFG["val_split"])
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Apply val transform to val split
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

        # ── Train ──
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

        # ── Validate ──
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

        tr_loss, tr_acc = tl / tt, tc / tt
        vl_loss, vl_acc = vl / vt, vc / vt
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

    # ── Plots ──
    plot_history(history)

    # ── Classification report ──
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
    plot_confusion(all_l, all_p)

    print(f"\n✅  Model saved → {CFG['model_path']}")
    print("Next step:  python detect.py")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()
