#!/usr/bin/env python3
"""
SROIE (ICDAR 2019) ham veri setini PyTorch ImageFolder formatına dönüştürür.

Kullanım:
    python scripts/prepare_sroie_dataset.py --raw_dir data/raw --output_dir data/processed/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

CLASSES: List[str] = ["Address", "Company", "Date", "Total"]

ENTITY_KEY_MAP: Dict[str, str] = {
    "company": "Company",
    "address": "Address",
    "date": "Date",
    "total": "Total",
}

CLASS_ALIASES: Dict[str, str] = {
    "address": "Address",
    "company": "Company",
    "date": "Date",
    "total": "Total",
    "addr": "Address",
    "comp": "Company",
    "vendor": "Company",
    "merchant": "Company",
    "amount": "Total",
    "sum": "Total",
}

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ANNOTATION_EXTENSIONS: Set[str] = {".txt", ".json"}

IMAGE_DIR_NAMES: Set[str] = {
    "img",
    "image",
    "images",
    "imgs",
    "picture",
    "pictures",
}

ENTITY_DIR_NAMES: Set[str] = {
    "entities",
    "entity",
    "labels",
    "label",
    "annotation",
    "annotations",
}

SPLIT_NAMES: Set[str] = {"train", "val", "validation", "test", "testing"}

RANDOM_STATE = 42
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------


@dataclass
class RawLayout:
    """Ham veri seti klasör yapısı analiz sonucu."""

    layout_type: str  # "sroie_standard" | "imagefolder" | "flat"
    root: Path
    train_image_dir: Optional[Path] = None
    test_image_dir: Optional[Path] = None
    train_entity_dir: Optional[Path] = None
    test_entity_dir: Optional[Path] = None
    image_dirs: List[Path] = field(default_factory=list)
    entity_dirs: List[Path] = field(default_factory=list)
    existing_splits: Dict[str, Path] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ReceiptRecord:
    """Tek bir fiş kaydı."""

    receipt_id: str
    image_path: Path
    entities: Dict[str, str]
    split: str  # "train" | "val" | "test"
    source_entity_path: Optional[Path] = None


@dataclass
class ProcessingStats:
    """İşlem istatistikleri."""

    processed_files: int = 0
    skipped_files: int = 0
    missing_annotations: int = 0
    unreadable_annotations: int = 0
    missing_images: int = 0
    copied_images: int = 0
    records: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------


def setup_logging(log_dir: Path, verbose: bool = True) -> None:
    """Logging yapılandırması."""
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "prepare_dataset.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def normalize_class_name(name: str) -> Optional[str]:
    """Sınıf adını standart forma getirir."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    title = name.strip().title()
    if title in CLASSES:
        return title
    return None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_annotation_file(path: Path) -> bool:
    return path.suffix.lower() in ANNOTATION_EXTENSIONS


def find_image_for_stem(stem: str, image_dirs: Iterable[Path]) -> Optional[Path]:
    """Dosya adı köküne göre görüntü dosyasını bulur."""
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            # Büyük/küçük harf varyasyonları (Windows/Colab uyumu)
            for match in img_dir.glob(f"{stem}.*"):
                if match.suffix.lower() in IMAGE_EXTENSIONS:
                    return match
    return None


def parse_entity_file(path: Path) -> Tuple[Optional[Dict[str, str]], bool]:
    """
    Entity annotation dosyasını okur.

    Returns:
        (entities_dict, success) — başarısız okumada (None, False)
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None, False

        data = json.loads(text)
        if not isinstance(data, dict):
            return None, False

        entities: Dict[str, str] = {}
        for raw_key, value in data.items():
            norm_key = raw_key.strip().lower()
            if norm_key not in ENTITY_KEY_MAP:
                continue
            class_name = ENTITY_KEY_MAP[norm_key]
            if value is None:
                continue
            str_value = str(value).strip()
            if str_value:
                entities[class_name] = str_value

        return entities, True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.debug("Entity okunamadı %s: %s", path, exc)
        return None, False


def extract_receipt_id_from_filename(filename: str) -> str:
    """Dosya adından fiş kimliğini çıkarır (ör. X51009453804_company -> X51009453804)."""
    stem = Path(filename).stem
    for cls in CLASSES:
        suffix = f"_{cls.lower()}"
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    # receipt001_company gibi örnekler
    match = re.match(r"^(.+?)_(company|address|date|total)$", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    return stem


def ensure_class_dirs(output_dir: Path, splits: Iterable[str]) -> None:
    """ImageFolder uyumlu sınıf klasörlerini oluşturur."""
    for split in splits:
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def clear_output_dir(output_dir: Path) -> None:
    """Çıktı klasörünü temizler (yeniden oluşturma için)."""
    if output_dir.exists():
        for split in ("train", "val", "test"):
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
    ensure_class_dirs(output_dir, ("train", "val", "test"))


# ---------------------------------------------------------------------------
# Ham veri analizi
# ---------------------------------------------------------------------------


def _collect_dirs_by_name(root: Path, names: Set[str]) -> List[Path]:
    found: List[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in names:
            found.append(path)
    return sorted(set(found))


def _detect_imagefolder_layout(raw_dir: Path) -> Optional[RawLayout]:
    """Doğrudan train/val/test + sınıf klasörleri yapısını tespit eder."""
    split_dirs: Dict[str, Path] = {}
    for name in ("train", "val", "validation", "test"):
        candidate = raw_dir / name
        if candidate.is_dir():
            key = "val" if name == "validation" else name
            if key not in split_dirs:
                split_dirs[key] = candidate

    if not split_dirs:
        return None

    class_found = False
    for split_path in split_dirs.values():
        for child in split_path.iterdir():
            if child.is_dir() and normalize_class_name(child.name) in CLASSES:
                class_found = True
                break

    if not class_found:
        return None

    layout = RawLayout(
        layout_type="imagefolder",
        root=raw_dir,
        existing_splits=split_dirs,
    )
    layout.notes.append("Mevcut ImageFolder yapısı tespit edildi.")
    return layout


def _detect_sroie_standard_layout(raw_dir: Path) -> RawLayout:
    """SROIE train/test + img/entities yapısını tespit eder."""
    layout = RawLayout(layout_type="sroie_standard", root=raw_dir)

    # Kök veya bir alt klasörde SROIE2019 benzeri yapı ara
    search_roots = [raw_dir]
    for child in raw_dir.iterdir():
        if child.is_dir():
            search_roots.append(child)

    for root in search_roots:
        train_dir = root / "train"
        test_dir = root / "test"
        if not train_dir.is_dir():
            continue

        train_img = None
        test_img = None
        train_ent = None
        test_ent = None

        for sub in train_dir.iterdir():
            if sub.is_dir():
                low = sub.name.lower()
                if low in IMAGE_DIR_NAMES:
                    train_img = sub
                elif low in ENTITY_DIR_NAMES:
                    train_ent = sub

        if test_dir.is_dir():
            for sub in test_dir.iterdir():
                if sub.is_dir():
                    low = sub.name.lower()
                    if low in IMAGE_DIR_NAMES:
                        test_img = sub
                    elif low in ENTITY_DIR_NAMES:
                        test_ent = sub

        if train_ent or train_img:
            layout.train_image_dir = train_img
            layout.test_image_dir = test_img
            layout.train_entity_dir = train_ent
            layout.test_entity_dir = test_ent
            layout.root = root
            layout.notes.append(f"SROIE standart yapı: {root}")
            break

    layout.image_dirs = _collect_dirs_by_name(raw_dir, IMAGE_DIR_NAMES)
    layout.entity_dirs = _collect_dirs_by_name(raw_dir, ENTITY_DIR_NAMES)

    if layout.train_image_dir:
        layout.image_dirs.insert(0, layout.train_image_dir)
    if layout.test_image_dir:
        layout.image_dirs.append(layout.test_image_dir)

    layout.image_dirs = list(dict.fromkeys(layout.image_dirs))
    layout.entity_dirs = list(dict.fromkeys(layout.entity_dirs))

    return layout


def analyze_raw_structure(raw_dir: Path) -> RawLayout:
    """Ham klasör yapısını otomatik analiz eder."""
    raw_dir = raw_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Ham veri klasörü bulunamadı: {raw_dir}\n"
            "Kaggle'dan indirdiğiniz SROIE zip dosyasını data/raw içine çıkarın."
        )

    if not any(raw_dir.iterdir()):
        raise FileNotFoundError(
            f"Ham veri klasörü boş: {raw_dir}\n"
            "Lütfen SROIE2019 veri setini data/raw altına yerleştirin."
        )

    imagefolder = _detect_imagefolder_layout(raw_dir)
    if imagefolder is not None:
        logger.info("Yapı tipi: ImageFolder (mevcut sınıf klasörleri)")
        return imagefolder

    sroie = _detect_sroie_standard_layout(raw_dir)
    if sroie.train_entity_dir or sroie.train_image_dir or sroie.entity_dirs:
        logger.info("Yapı tipi: SROIE standart (img + entities)")
        return sroie

    # Düz yapı: tüm görüntüler ve annotationlar karışık
    sroie.layout_type = "flat"
    sroie.notes.append("Standart SROIE yapısı bulunamadı; düz tarama yapılacak.")
    logger.warning("Standart yapı tespit edilemedi, genel tarama kullanılacak.")
    return sroie


# ---------------------------------------------------------------------------
# Fiş kayıtları oluşturma
# ---------------------------------------------------------------------------


def build_receipt_records_from_entities(
    entity_dir: Path,
    image_dirs: List[Path],
    default_split: str,
    stats: ProcessingStats,
) -> List[ReceiptRecord]:
    """Entity dosyalarından fiş kayıtları oluşturur."""
    records: List[ReceiptRecord] = []
    entity_files = sorted(
        f for f in entity_dir.iterdir() if f.is_file() and is_annotation_file(f)
    )

    for entity_path in tqdm(entity_files, desc=f"Entity okuma ({default_split})", unit="fiş"):
        stats.processed_files += 1
        stem = entity_path.stem

        entities, ok = parse_entity_file(entity_path)
        if not ok:
            stats.unreadable_annotations += 1
            stats.skipped_files += 1
            logger.warning("Okunamayan annotation: %s", entity_path)
            continue

        if not entities:
            stats.missing_annotations += 1
            stats.skipped_files += 1
            logger.warning("Boş veya eksik entity alanları: %s", entity_path)
            continue

        image_path = find_image_for_stem(stem, image_dirs)
        if image_path is None:
            stats.missing_images += 1
            stats.skipped_files += 1
            logger.warning("Görüntü bulunamadı: %s (entity: %s)", stem, entity_path)
            continue

        records.append(
            ReceiptRecord(
                receipt_id=stem,
                image_path=image_path,
                entities=entities,
                split=default_split,
                source_entity_path=entity_path,
            )
        )

    return records


def collect_sroie_records(layout: RawLayout, stats: ProcessingStats) -> Tuple[List[ReceiptRecord], List[ReceiptRecord]]:
    """SROIE yapısından train ve test fiş kayıtlarını toplar."""
    train_records: List[ReceiptRecord] = []
    test_records: List[ReceiptRecord] = []

    train_image_dirs: List[Path] = []
    test_image_dirs: List[Path] = []

    if layout.train_image_dir:
        train_image_dirs.append(layout.train_image_dir)
    if layout.test_image_dir:
        test_image_dirs.append(layout.test_image_dir)

    # Yedek: tüm image klasörlerini de tara
    for d in layout.image_dirs:
        if d not in train_image_dirs and d not in test_image_dirs:
            train_image_dirs.append(d)

    if layout.train_entity_dir:
        train_records = build_receipt_records_from_entities(
            layout.train_entity_dir, train_image_dirs or layout.image_dirs, "train", stats
        )

    if layout.test_entity_dir:
        test_records = build_receipt_records_from_entities(
            layout.test_entity_dir, test_image_dirs or layout.image_dirs, "test", stats
        )

    # Entity klasörü train/test altında değilse genel tarama
    if not train_records and layout.entity_dirs:
        for ent_dir in layout.entity_dirs:
            parent = ent_dir.parent.name.lower()
            split = "test" if parent == "test" else "train"
            recs = build_receipt_records_from_entities(
                ent_dir, layout.image_dirs, split, stats
            )
            if split == "test":
                test_records.extend(recs)
            else:
                train_records.extend(recs)

    return train_records, test_records


def split_train_val_records(
    train_records: List[ReceiptRecord],
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[ReceiptRecord], List[ReceiptRecord]]:
    """
    Train kayıtlarını fiş düzeyinde %90/%10 train/val olarak ayırır.
    Aynı fişten türetilen örnekler aynı split'te kalır.
    """
    if not train_records:
        return [], []

    receipt_ids = sorted({r.receipt_id for r in train_records})
    id_to_records: Dict[str, List[ReceiptRecord]] = defaultdict(list)
    for rec in train_records:
        id_to_records[rec.receipt_id].append(rec)

    # Stratifikasyon: mevcut sınıf sayısına göre
    stratify_labels = [
        len(id_to_records[rid][0].entities) for rid in receipt_ids
    ]

    # Çok az örnekli sınıflarda stratify hata verebilir; güvenli fallback
    try:
        train_ids, val_ids = train_test_split(
            receipt_ids,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        logger.warning("Stratified split başarısız; rastgele bölünüyor.")
        train_ids, val_ids = train_test_split(
            receipt_ids,
            test_size=val_ratio,
            random_state=random_state,
        )

    train_out: List[ReceiptRecord] = []
    val_out: List[ReceiptRecord] = []

    for rid in train_ids:
        for rec in id_to_records[rid]:
            rec.split = "train"
            train_out.append(rec)

    for rid in val_ids:
        for rec in id_to_records[rid]:
            rec.split = "val"
            val_out.append(rec)

    logger.info(
        "Train/val fiş ayrımı: %d train fiş, %d val fiş (toplam %d)",
        len(train_ids),
        len(val_ids),
        len(receipt_ids),
    )
    return train_out, val_out


# ---------------------------------------------------------------------------
# ImageFolder çıktısı
# ---------------------------------------------------------------------------


def copy_receipt_to_class_folders(
    record: ReceiptRecord,
    output_dir: Path,
    stats: ProcessingStats,
) -> List[dict]:
    """Her sınıf için görüntü kopyası oluşturur ve özet kayıt döner."""
    summary_rows: List[dict] = []
    ext = record.image_path.suffix.lower() or ".jpg"

    for cls in CLASSES:
        if cls not in record.entities:
            continue

        dest_name = f"{record.receipt_id}_{cls.lower()}{ext}"
        dest_path = output_dir / record.split / cls / dest_name

        try:
            shutil.copy2(record.image_path, dest_path)
            stats.copied_images += 1
            summary_rows.append(
                {
                    "receipt_id": record.receipt_id,
                    "split": record.split,
                    "class": cls,
                    "filename": dest_name,
                    "source_image": str(record.image_path),
                    "entity_value": record.entities[cls],
                    "entity_file": str(record.source_entity_path or ""),
                }
            )
        except OSError as exc:
            stats.skipped_files += 1
            logger.error("Kopyalama hatası %s -> %s: %s", record.image_path, dest_path, exc)

    return summary_rows


def materialize_imagefolder(
    records: List[ReceiptRecord],
    output_dir: Path,
    stats: ProcessingStats,
) -> None:
    """Tüm kayıtları ImageFolder yapısına yazar."""
    for record in tqdm(records, desc="Görüntüler kopyalanıyor", unit="fiş"):
        rows = copy_receipt_to_class_folders(record, output_dir, stats)
        stats.records.extend(rows)


def process_existing_imagefolder(
    layout: RawLayout,
    output_dir: Path,
    stats: ProcessingStats,
) -> None:
    """Mevcut sınıf klasörlerinden normalize ederek yeniden oluşturur."""
    all_records: List[ReceiptRecord] = []

    for split_name, split_path in layout.existing_splits.items():
        norm_split = "val" if split_name == "validation" else split_name

        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            norm_cls = normalize_class_name(class_dir.name)
            if norm_cls is None:
                logger.warning("Bilinmeyen sınıf klasörü atlandı: %s", class_dir)
                continue

            for img_path in class_dir.iterdir():
                if not is_image_file(img_path):
                    continue
                stats.processed_files += 1
                receipt_id = extract_receipt_id_from_filename(img_path.name)
                all_records.append(
                    ReceiptRecord(
                        receipt_id=receipt_id,
                        image_path=img_path,
                        entities={norm_cls: "from_existing_folder"},
                        split=norm_split,
                    )
                )

    # Train split varsa ve val yoksa yeniden böl
    train_recs = [r for r in all_records if r.split == "train"]
    val_recs = [r for r in all_records if r.split == "val"]
    test_recs = [r for r in all_records if r.split == "test"]

    if train_recs and not val_recs and "val" not in layout.existing_splits:
        # Fiş düzeyinde grupla ve böl
        grouped: Dict[str, List[ReceiptRecord]] = defaultdict(list)
        for rec in train_recs:
            grouped[rec.receipt_id].append(rec)

        unique_train = [
            ReceiptRecord(
                receipt_id=rid,
                image_path=recs[0].image_path,
                entities={cls: "ok" for rec in recs for cls in rec.entities},
                split="train",
            )
            for rid, recs in grouped.items()
        ]
        split_train, split_val = split_train_val_records(unique_train)
        # Orijinal çoklu sınıf kayıtlarını yeniden eşle
        final_train: List[ReceiptRecord] = []
        final_val: List[ReceiptRecord] = []
        train_ids = {r.receipt_id for r in split_train}
        val_ids = {r.receipt_id for r in split_val}
        for rec in train_recs:
            if rec.receipt_id in train_ids:
                rec.split = "train"
                final_train.append(rec)
            elif rec.receipt_id in val_ids:
                rec.split = "val"
                final_val.append(rec)
        all_records = final_train + final_val + test_recs
        layout.notes.append("Mevcut train verisi %90/%10 train/val olarak yeniden bölündü.")

    materialize_imagefolder(all_records, output_dir, stats)


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------


def build_class_distribution_df(records: List[dict]) -> pd.DataFrame:
    """Split ve sınıf bazlı dağılım tablosu."""
    if not records:
        return pd.DataFrame(columns=["split", "class", "count"])

    df = pd.DataFrame(records)
    dist = (
        df.groupby(["split", "class"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "class"])
    )
    return dist


def write_reports(
    stats: ProcessingStats,
    output_dir: Path,
    log_dir: Path,
    layout: RawLayout,
) -> None:
    """CSV ve metin raporlarını yazar."""
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(stats.records)
    summary_path = log_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("Özet CSV: %s", summary_path)

    dist_df = build_class_distribution_df(stats.records)
    dist_path = log_dir / "class_distribution.csv"
    dist_df.to_csv(dist_path, index=False, encoding="utf-8")
    logger.info("Dağılım CSV: %s", dist_path)

    total_images = len(stats.records)
    split_counts = (
        summary_df.groupby("split").size().to_dict() if not summary_df.empty else {}
    )
    class_split_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not dist_df.empty:
        for _, row in dist_df.iterrows():
            class_split_counts[row["split"]][row["class"]] = int(row["count"])

    report_lines = [
        "=" * 60,
        "SROIE Veri Seti Dönüştürme Raporu",
        f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Ham veri kökü      : {layout.root}",
        f"Yapı tipi          : {layout.layout_type}",
        f"Çıktı klasörü      : {output_dir}",
        "",
        "--- Genel İstatistikler ---",
        f"Toplam görüntü sayısı          : {total_images}",
        f"Train örnek sayısı             : {split_counts.get('train', 0)}",
        f"Validation örnek sayısı        : {split_counts.get('val', 0)}",
        f"Test örnek sayısı              : {split_counts.get('test', 0)}",
        "",
        "--- Split Bazlı Sınıf Dağılımı ---",
    ]

    for split in ("train", "val", "test"):
        report_lines.append(f"\n[{split.upper()}]")
        counts = class_split_counts.get(split, {})
        if counts:
            for cls in CLASSES:
                report_lines.append(f"  {cls:10s}: {counts.get(cls, 0)}")
        else:
            report_lines.append("  (örnek yok)")

    report_lines.extend(
        [
            "",
            "--- İşlem Detayları ---",
            f"İşlenen dosya sayısı           : {stats.processed_files}",
            f"Atlanan dosya sayısı           : {stats.skipped_files}",
            f"Kopyalanan görüntü sayısı      : {stats.copied_images}",
            f"Eksik annotation sayısı        : {stats.missing_annotations}",
            f"Okunamayan annotation sayısı   : {stats.unreadable_annotations}",
            f"Eksik görüntü sayısı           : {stats.missing_images}",
            "",
            "--- Yapı Notları ---",
        ]
    )
    for note in layout.notes:
        report_lines.append(f"  - {note}")

    report_path = log_dir / "split_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Rapor: %s", report_path)

    # Konsola özet
    print("\n" + "\n".join(report_lines[:20]))
    if stats.missing_images > 0:
        print(
            f"\nUYARI: {stats.missing_images} fiş için görüntü bulunamadı. "
            "Kaggle zip dosyasında train/img ve test/img klasörlerinin "
            "data/raw altında olduğundan emin olun."
        )


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------


def prepare_dataset(raw_dir: Path, output_dir: Path, log_dir: Path) -> ProcessingStats:
    """Ana dönüştürme pipeline'ı."""
    stats = ProcessingStats()
    layout = analyze_raw_structure(raw_dir)

    for note in layout.notes:
        logger.info(note)

    clear_output_dir(output_dir)

    if layout.layout_type == "imagefolder":
        process_existing_imagefolder(layout, output_dir, stats)
    else:
        train_records, test_records = collect_sroie_records(layout, stats)

        if not train_records and not test_records:
            raise RuntimeError(
                "İşlenecek veri bulunamadı. Ham klasörde entity veya görüntü "
                "dosyalarını kontrol edin.\n"
                f"Aranan kök: {raw_dir.resolve()}\n"
                "Beklenen yapı örneği:\n"
                "  data/raw/SROIE2019/train/img/\n"
                "  data/raw/SROIE2019/train/entities/\n"
                "  data/raw/SROIE2019/test/img/\n"
                "  data/raw/SROIE2019/test/entities/"
            )

        train_split, val_split = split_train_val_records(train_records)
        all_records = train_split + val_split + test_records
        materialize_imagefolder(all_records, output_dir, stats)

    write_reports(stats, output_dir, log_dir, layout)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SROIE ham veri setini PyTorch ImageFolder formatına dönüştürür."
    )
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Ham SROIE veri seti klasörü (varsayılan: data/raw)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "data" / "processed" / "dataset",
        help="ImageFolder çıktı klasörü (varsayılan: data/processed/dataset)",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=project_root / "outputs" / "logs",
        help="Log ve rapor klasörü (varsayılan: outputs/logs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ayrıntılı log çıktısı",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_dir, verbose=args.verbose)

    logger.info("SROIE veri seti dönüştürme başlıyor...")
    logger.info("Ham klasör : %s", args.raw_dir.resolve())
    logger.info("Çıktı      : %s", args.output_dir.resolve())

    try:
        stats = prepare_dataset(args.raw_dir, args.output_dir, args.log_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1

    if stats.copied_images == 0:
        logger.error(
            "Hiç görüntü kopyalanamadı. Görüntü dosyalarının (train/img, test/img) "
            "ham veri klasöründe mevcut olduğunu doğrulayın."
        )
        return 1

    logger.info("Dönüştürme tamamlandı. %d görüntü oluşturuldu.", stats.copied_images)
    return 0


if __name__ == "__main__":
    sys.exit(main())

