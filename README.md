# SROIE Fiş Görüntü Sınıflandırma Projesi

ICDAR 2019 SROIE (Scanned Receipt OCR and Information Extraction) veri setini PyTorch `ImageFolder` formatına dönüştürerek derin öğrenme tabanlı görüntü sınıflandırma modelleri (AlexNet, MobileNetV2, VGG16, ResNet50, özgün CNN) eğitmek için hazırlanmış proje iskeleti.

## Problem Tanımı

Fiş görüntülerini aşağıdaki 4 bilgi alanı sınıfından birine göre sınıflandırma:

| Sınıf   | Açıklama                          |
|---------|-----------------------------------|
| Company | İşletme / şirket adı              |
| Address | Adres bilgisi                     |
| Date    | Tarih bilgisi                     |
| Total   | Toplam tutar                      |

Her fiş görüntüsü, entity annotation dosyasındaki mevcut alanlara göre ilgili sınıf klasörlerine kopyalanır (ör. `X51009453804_company.jpg`).

## Proje Klasör Yapısı

```
derin_ogrenme/
├── data/
│   ├── raw/                      # Ham Kaggle veri seti
│   └── processed/
│       └── dataset/              # ImageFolder çıktısı
│           ├── train/
│           │   ├── Address/
│           │   ├── Company/
│           │   ├── Date/
│           │   └── Total/
│           ├── val/
│           │   ├── Address/
│           │   ├── Company/
│           │   ├── Date/
│           │   └── Total/
│           └── test/
│               ├── Address/
│               ├── Company/
│               ├── Date/
│               └── Total/
├── outputs/
│   ├── logs/                     # dataset_summary.csv, split_report.txt
│   ├── figures/                  # Eğitim grafikleri (sonraki aşama)
│   └── models/                   # Kaydedilen modeller (sonraki aşama)
├── scripts/
│   ├── prepare_sroie_dataset.py  # Veri dönüştürme
│   └── check_dataset.py          # Veri seti kontrolü
├── requirements.txt
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Veri Seti Hazırlama

### 1. Kaggle'dan İndirme

[SROIE 2019 Kaggle veri setini](https://www.kaggle.com/datasets/urbikn/sroie-2019) indirin.

### 2. Ham Veriyi Yerleştirme

Zip dosyasını `data/raw` altına çıkarın. Aşağıdaki yapı oluşmalıdır:

```
data/raw/
└── SROIE2019/          # veya doğrudan train/test klasörleri
    ├── train/
    │   ├── img/        # Fiş görüntüleri (.jpg)
    │   ├── box/        # Metin kutusu annotationları
    │   └── entities/   # JSON entity dosyaları (.txt)
    └── test/
        ├── img/
        ├── box/
        └── entities/
```

> **Not:** Görüntü dosyaları (`train/img`, `test/img`) zorunludur. Yalnızca annotation dosyaları varsa dönüştürme scripti görüntü bulamadığı için çalışmaz.

Eğer veri seti proje kökünde `SROIE2019/` olarak duruyorsa, şu komutla taşıyabilirsiniz:

```bash
# Windows PowerShell
Move-Item -Path SROIE2019 -Destination data\raw\

# Linux / macOS
mv SROIE2019 data/raw/
```

### 3. Veri Dönüştürme

```bash
python scripts/prepare_sroie_dataset.py --raw_dir data/raw --output_dir data/processed/dataset
```

Script şunları yapar:

- Ham klasör yapısını otomatik analiz eder
- Entity dosyalarından Company, Address, Date, Total alanlarını okur
- Her sınıf için fiş görüntüsünü `{receipt_id}_{class}.jpg` adıyla kopyalar
- Orijinal **test** setini korur (eğitime dahil etmez)
- Orijinal **train** setini fiş düzeyinde %90 train / %10 validation olarak böler (`random_state=42`)
- Aynı fişten türetilen örneklerin train ve val arasında bölünmesini engeller

### 4. Veri Seti Kontrolü

```bash
python scripts/check_dataset.py --data_dir data/processed/dataset
```

Bu komut:

- `train`, `val`, `test` klasörlerini ve sınıf dağılımlarını listeler
- PyTorch `ImageFolder` ile okunup okunamadığını test eder
- Örnek bir batch yükleyip tensor boyutunu yazdırır

## Beklenen Veri Bölümleri

| Split | Yaklaşık fiş sayısı | Yaklaşık görüntü sayısı (×4 sınıf) |
|-------|---------------------|-------------------------------------|
| Train | ~563 fiş            | ~2250 görüntü                       |
| Val   | ~63 fiş             | ~250 görüntü                        |
| Test  | ~347 fiş (orijinal) | ~1380 görüntü                       |

> Tam sayılar entity dosyalarındaki eksik alanlara göre değişebilir. Detaylar `outputs/logs/split_report.txt` dosyasında yer alır.

## Çıktı Raporları

Dönüştürme sonrası `outputs/logs/` altında:

| Dosya                    | İçerik                                      |
|--------------------------|---------------------------------------------|
| `dataset_summary.csv`    | Her kopyalanan görüntünün detaylı kaydı     |
| `class_distribution.csv` | Split × sınıf örnek sayıları                |
| `split_report.txt`       | Özet istatistikler ve işlem raporu          |
| `prepare_dataset.log`    | Ayrıntılı işlem logu                        |

## Colab Kullanımı

Google Colab'da proje kökünü yükledikten sonra:

```python
# Bağımlılıkları kur
!pip install -r requirements.txt

# Veri setini dönüştür (Drive'a kopyaladıysanız yolu güncelleyin)
!python scripts/prepare_sroie_dataset.py --raw_dir data/raw --output_dir data/processed/dataset

# Kontrol
!python scripts/check_dataset.py
```

Eğitim scriptlerinde kullanılacak veri yolu:

```python
DATA_DIR = "/content/derin_ogrenme/data/processed/dataset"  # Colab yolu
# veya
DATA_DIR = "data/processed/dataset"  # Göreli yol

train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_transform)
test_dataset  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=test_transform)
```

## Sonraki Adımlar

Bu aşamada yalnızca veri hazırlama altyapısı kurulmuştur. Sonraki ödev aşamasında:

- AlexNet, MobileNetV2, VGG16, ResNet50 ve özgün CNN modelleri eğitilecek
- Eğitim grafikleri `outputs/figures/` altına kaydedilecek
- En iyi modeller `outputs/models/` altına kaydedilecek
