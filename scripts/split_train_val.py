#!/usr/bin/env python3
"""
Mevcut ImageFolder train klasöründen fiş düzeyinde stratified val ayrımı yapar.

Kullanım:
    python scripts/split_train_val.py --data_dir data/processed/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["Address", "Company", "Date", "Total"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANDOM_STATE = 42
VAL_RATIO = 0.1

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def extract_receipt_id(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"^(.+)_(Address|Company|Date|Total)$", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def collect_train_receipts(train_dir: Path) -> Dict[str, Dict[str, Path]]:
    """receipt_id -> {class_name: file_path}"""
    receipts: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for cls in CLASSES:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Train sınıf klasörü yok: {cls_dir}")

        for img_path in cls_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rid = extract_receipt_id(img_path.name)
            receipts[rid][cls] = img_path

    return dict(receipts)


def ensure_val_dirs(data_dir: Path) -> None:
    for cls in CLASSES:
        (data_dir / "val" / cls).mkdir(parents=True, exist_ok=True)


def count_split(data_dir: Path, split: str) -> Dict[str, int]:
    counts = {}
    for cls in CLASSES:
        cls_dir = data_dir / split / cls
        if not cls_dir.exists():
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1 for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def split_train_val(
    data_dir: Path,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE,
    dry_run: bool = False,
) -> dict:
    data_dir = data_dir.resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train klasörü bulunamadı: {train_dir}")

    receipts = collect_train_receipts(train_dir)
    receipt_ids = sorted(receipts.keys())

    incomplete = [rid for rid, files in receipts.items() if len(files) != len(CLASSES)]
    if incomplete:
        logger.warning("%d fişte eksik sınıf var (ilk 5): %s", len(incomplete), incomplete[:5])

    complete_ids = [rid for rid in receipt_ids if len(receipts[rid]) == len(CLASSES)]
    logger.info("Toplam fiş: %d (tam: %d)", len(receipt_ids), len(complete_ids))

    try:
        train_ids, val_ids = train_test_split(
            complete_ids,
            test_size=val_ratio,
            random_state=random_state,
            stratify=[len(receipts[rid]) for rid in complete_ids],
        )
    except ValueError:
        train_ids, val_ids = train_test_split(
            complete_ids,
            test_size=val_ratio,
            random_state=random_state,
        )

    logger.info("Ayrım: %d train fiş, %d val fiş", len(train_ids), len(val_ids))

    if not dry_run:
        ensure_val_dirs(data_dir)
        # Önce val klasörünü temizle (yeniden çalıştırma için)
        for cls in CLASSES:
            for f in (val_dir / cls).iterdir():
                if f.is_file():
                    f.unlink()

        moved = 0
        for rid in val_ids:
            for cls, src in receipts[rid].items():
                dest = val_dir / cls / src.name
                shutil.move(str(src), str(dest))
                moved += 1
        logger.info("%d görüntü train -> val taşındı", moved)

    train_counts = count_split(data_dir, "train") if not dry_run else {}
    val_counts = count_split(data_dir, "val") if not dry_run else {}
    test_counts = count_split(data_dir, "test")

    if dry_run:
        train_counts = {cls: sum(1 for rid in train_ids for c in receipts[rid] if c == cls) for cls in CLASSES}
        val_counts = {cls: sum(1 for rid in val_ids for c in receipts[rid] if c == cls) for cls in CLASSES}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "val_ratio": val_ratio,
        "random_state": random_state,
        "train_receipts": len(train_ids),
        "val_receipts": len(val_ids),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "incomplete_receipts": len(incomplete),
    }
    return summary


def write_reports(summary: dict, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_key, split_name in [
        ("train_counts", "train"),
        ("val_counts", "val"),
        ("test_counts", "test"),
    ]:
        for cls, count in summary[split_key].items():
            rows.append({"split": split_name, "class": cls, "count": count})

    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(log_dir / "class_distribution.csv", index=False)

    total = int(dist_df["count"].sum())
    train_total = int(dist_df[dist_df["split"] == "train"]["count"].sum())
    val_total = int(dist_df[dist_df["split"] == "val"]["count"].sum())
    test_total = int(dist_df[dist_df["split"] == "test"]["count"].sum())

    lines = [
        "=" * 60,
        "Train / Val Ayrım Raporu (Kırpılmış Alan Görüntüleri)",
        f"Tarih: {summary['timestamp']}",
        "=" * 60,
        "",
        f"Veri klasörü: {summary['data_dir']}",
        f"Val oranı: {summary['val_ratio']*100:.0f}% | random_state={summary['random_state']}",
        "",
        f"Train fiş sayısı: {summary['train_receipts']}",
        f"Val fiş sayısı:   {summary['val_receipts']}",
        "",
        "--- Örnek Sayıları ---",
        f"Toplam görüntü: {total}",
        f"Train: {train_total}",
        f"Val:   {val_total}",
        f"Test:  {test_total}",
        "",
        "--- Sınıf Bazlı ---",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"\n[{split.upper()}]")
        key = f"{split}_counts"
        for cls in CLASSES:
            lines.append(f"  {cls:10s}: {summary[key].get(cls, 0)}")

    lines.extend([
        "",
        f"Eksik sınıflı fiş: {summary['incomplete_receipts']}",
        "",
        "Not: Aynı fişten türetilen 4 görüntü aynı split'te kaldı.",
    ])

    report_path = log_dir / "split_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Rapor: %s", report_path)
    logger.info("Dağılım: %s", log_dir / "class_distribution.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train klasöründen val ayrımı yapar.")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=project_root / "data" / "processed" / "dataset",
    )
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    parser.add_argument("--log_dir", type=Path, default=project_root / "outputs" / "logs")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        summary = split_train_val(
            args.data_dir,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if not args.dry_run:
        write_reports(summary, args.log_dir)
        print("\n" + "=" * 40)
        print(f"Train: {summary['train_counts']}")
        print(f"Val:   {summary['val_counts']}")
        print(f"Test:  {summary['test_counts']}")
        print("=" * 40)

    return 0


if __name__ == "__main__":
    sys.exit(main())
