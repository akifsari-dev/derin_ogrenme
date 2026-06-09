#!/usr/bin/env python3
"""
Eğitilmiş modeller için test seti metrikleri, confusion matrix ve ROC üretir.

Kullanım:
    python scripts/evaluate_models.py --data_dir data/processed/dataset --models_dir outputs/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ["Address", "Company", "Date", "Total"]
MODEL_NAMES = ["alexnet", "mobilenetv2", "vgg16", "resnet50", "custom_cnn"]


def build_model(name: str, num_classes: int = 4) -> nn.Module:
    name = name.lower()
    if name == "alexnet":
        m = models.alexnet(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "custom_cnn":
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    else:
        raise ValueError(name)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.vstack(all_probs), np.array(all_preds), np.array(all_labels)


def specificity_score(y_true, y_pred, average="macro"):
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    if average == "macro":
        return float(np.mean(specs))
    return specs


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_prob, class_names, out_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = []
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (macro AUC={np.mean(aucs):.3f})")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return float(np.mean(aucs))


def evaluate_one(model_name, checkpoint, test_loader, device, figures_dir):
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])

    probs, preds, labels = predict(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    spec = specificity_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    macro_auc = roc_auc_score(label_binarize(labels, classes=list(range(4))), probs, average="macro", multi_class="ovr")

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, CLASSES, figures_dir / f"{model_name}_confusion_matrix.png")
    plot_roc(labels, probs, CLASSES, figures_dir / f"{model_name}_roc.png")

    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True)
    return {
        "model": model_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "specificity": round(spec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "auc": round(macro_auc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--data_dir", type=Path, default=project_root / "data" / "processed" / "dataset")
    parser.add_argument("--models_dir", type=Path, default=project_root / "outputs" / "models")
    parser.add_argument("--figures_dir", type=Path, default=project_root / "outputs" / "figures")
    parser.add_argument("--logs_dir", type=Path, default=project_root / "outputs" / "logs")
    parser.add_argument("--img_size", type=int, default=160)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(str(args.data_dir / "test"), transform=tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    results = []
    for name in MODEL_NAMES:
        ckpt = args.models_dir / f"{name}_best.pth"
        if not ckpt.exists():
            print(f"Atlandı (yok): {ckpt}")
            continue
        print(f"Değerlendiriliyor: {name}")
        results.append(evaluate_one(name, ckpt, test_loader, device, args.figures_dir))

    if not results:
        print("Hiç model bulunamadı. Önce eğitim yapın.")
        return

    df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "specificity", "f1_score", "auc"]]
    df.columns = ["Model", "Doğruluk (%)", "Kesinlik (%)", "Duyarlılık (%)", "Özgüllük (%)", "F1 Skor (%)", "AUC"]
    csv_path = args.logs_dir / "test_performance_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nKayıt: {csv_path}")

    with open(args.logs_dir / "test_performance_full.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
