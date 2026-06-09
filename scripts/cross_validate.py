#!/usr/bin/env python3
"""
En iyi test performanslı model için fiş düzeyinde stratified K-Fold CV.

Kullanım:
    python scripts/cross_validate.py --model custom_cnn --k 5
"""

from __future__ import annotations

import argparse
import copy
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_models import build_model, specificity_score

CLASSES = ["Address", "Company", "Date", "Total"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_receipt_id(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^(.+)_(Address|Company|Date|Total)$", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def build_receipt_index(data_dir: Path):
    """Fiş düzeyinde örnek listesi: her fiş için 4 görüntü indeksi."""
    root = data_dir / "train"
    ds = datasets.ImageFolder(str(root))
    receipt_to_indices: dict[str, list[int]] = defaultdict(list)
    receipt_label: dict[str, int] = {}

    for idx, (path, _) in enumerate(ds.samples):
        rid = extract_receipt_id(Path(path))
        receipt_to_indices[rid].append(idx)

    complete = {r: idxs for r, idxs in receipt_to_indices.items() if len(idxs) == 4}
    receipts = sorted(complete.keys())
    # Stratify için dummy label (4 sınıf tam → 0)
    labels = [0] * len(receipts)
    return ds, receipts, labels


def train_fold(model, train_loader, val_loader, device, epochs=8, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    best_acc, best_state = 0.0, None

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--data_dir", type=Path, default=project_root / "data" / "processed" / "dataset")
    parser.add_argument("--model", type=str, default="custom_cnn")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "logs" / "cross_validation.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds, receipts, _ = build_receipt_index(args.data_dir)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    rows = []
    receipt_to_indices = {}
    for idx, (path, _) in enumerate(train_ds.samples):
        rid = extract_receipt_id(Path(path))
        receipt_to_indices.setdefault(rid, []).append(idx)

    for fold, (tr_rec_idx, va_rec_idx) in enumerate(skf.split(receipts, [0] * len(receipts)), 1):
        tr_ids = [receipts[i] for i in tr_rec_idx]
        va_ids = [receipts[i] for i in va_rec_idx]
        tr_idx = [i for r in tr_ids for i in receipt_to_indices[r]]
        va_idx = [i for r in va_ids for i in receipt_to_indices[r]]

        tr_ds = Subset(datasets.ImageFolder(str(args.data_dir / "train"), transform=tf), tr_idx)
        va_ds = Subset(datasets.ImageFolder(str(args.data_dir / "train"), transform=eval_tf), va_idx)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_model(args.model).to(device)
        model = train_fold(model, tr_loader, va_loader, device, epochs=args.epochs)
        y_true, y_pred, y_prob = evaluate(model, va_loader, device)

        y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
        rows.append({
            "Fold": f"Fold {fold}",
            "Doğruluk": round(accuracy_score(y_true, y_pred) * 100, 2),
            "Duyarlılık": round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
            "Özgüllük": round(specificity_score(y_true, y_pred) * 100, 2),
            "Kesinlik": round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
            "F1 Skor": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
            "AUC": round(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"), 4),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    avg = df.drop(columns=["Fold"]).mean(numeric_only=True)
    avg_row = {"Fold": "Ortalama", **{c: round(avg[c], 4) if c == "AUC" else round(avg[c], 2) for c in avg.index}}
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nKayıt: {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
