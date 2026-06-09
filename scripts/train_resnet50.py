#!/usr/bin/env python3
"""
ResNet50 ile SROIE fiş sınıflandırma eğitimi.

Kullanım:
    python scripts/train_resnet50.py --data_dir data/processed/dataset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def plot_history(history: dict, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet50 eğitimi")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--data_dir", type=Path, default=project_root / "data" / "processed" / "dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true", help="Sıfırdan eğit")
    parser.add_argument("--output_dir", type=Path, default=project_root / "outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    use_pretrained = args.pretrained and not args.no_pretrained

    models_dir = args.output_dir / "models"
    figures_dir = args.output_dir / "figures"
    logs_dir = args.output_dir / "logs"
    for d in (models_dir, figures_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    model = build_model(num_classes=len(class_names), pretrained=use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_path = models_dir / "resnet50_best.pth"

    print(f"Device: {device}")
    print(f"Sınıflar: {class_names}")
    print(f"Pretrained: {use_pretrained}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": "resnet50",
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss={test_loss:.4f} | Test acc={test_acc:.4f}")
    print(f"En iyi val acc={best_val_acc:.4f} -> {best_path}")

    plot_history(history, figures_dir / "resnet50_history.png")

    summary = {
        "model": "resnet50",
        "pretrained": use_pretrained,
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "class_names": class_names,
    }
    (logs_dir / "resnet50_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
