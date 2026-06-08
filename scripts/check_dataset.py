#!/usr/bin/env python3
"""
İşlenmiş veri setini kontrol eder ve PyTorch ImageFolder uyumluluğunu test eder.

Kullanım:
    python scripts/check_dataset.py
    python scripts/check_dataset.py --data_dir data/processed/dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["Address", "Company", "Date", "Total"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(
        1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_structure(data_dir: Path) -> bool:
    """Klasör yapısını ve sınıf dağılımını kontrol eder."""
    print("=" * 60)
    print(f"Veri seti kontrolü: {data_dir.resolve()}")
    print("=" * 60)

    if not data_dir.exists():
        print(f"HATA: Veri klasörü bulunamadı: {data_dir}")
        print("Önce şu komutu çalıştırın:")
        print("  python scripts/prepare_sroie_dataset.py --raw_dir data/raw --output_dir data/processed/dataset")
        return False

    ok = True
    total = 0

    for split in SPLITS:
        split_dir = data_dir / split
        print(f"\n[{split.upper()}]")
        if not split_dir.exists():
            print(f"  UYARI: {split} klasörü yok")
            ok = False
            continue

        split_total = 0
        for cls in CLASSES:
            cls_dir = split_dir / cls
            count = count_images(cls_dir)
            split_total += count
            status = "OK" if count > 0 else "BOŞ"
            print(f"  {cls:10s}: {count:5d} görüntü  [{status}]")
            if count == 0:
                ok = False

        print(f"  {'TOPLAM':10s}: {split_total:5d} görüntü")
        total += split_total

    print(f"\nGenel toplam: {total} görüntü")
    return ok and total > 0


def test_imagefolder(data_dir: Path, batch_size: int = 8) -> bool:
    """PyTorch ImageFolder ile yükleme testi."""
    print("\n" + "=" * 60)
    print("PyTorch ImageFolder testi")
    print("=" * 60)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    all_ok = True
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        try:
            dataset = datasets.ImageFolder(str(split_dir), transform=transform)
        except FileNotFoundError as exc:
            print(f"  [{split}] HATA: {exc}")
            all_ok = False
            continue

        if len(dataset) == 0:
            print(f"  [{split}] UYARI: Boş veri seti")
            all_ok = False
            continue

        print(f"  [{split}] ImageFolder OK — {len(dataset)} örnek, {len(dataset.classes)} sınıf")
        print(f"          Sınıflar: {dataset.classes}")
        print(f"          class_to_idx: {dataset.class_to_idx}")

        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        images, labels = next(iter(loader))
        print(f"          Örnek batch tensor boyutu: {tuple(images.shape)}")
        print(f"          Örnek batch etiket boyutu: {tuple(labels.shape)}")
        print(f"          Piksel değer aralığı: [{images.min():.3f}, {images.max():.3f}]")

    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="İşlenmiş SROIE veri setini kontrol eder.")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=project_root / "data" / "processed" / "dataset",
        help="ImageFolder veri seti kökü (varsayılan: data/processed/dataset)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Test batch boyutu (varsayılan: 8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    structure_ok = check_structure(args.data_dir)
    if not structure_ok:
        return 1

    loader_ok = test_imagefolder(args.data_dir, batch_size=args.batch_size)
    if loader_ok:
        print("\nTüm kontroller başarılı. Veri seti eğitime hazır.")
        return 0

    print("\nBazı kontroller başarısız. split_report.txt dosyasını inceleyin.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
