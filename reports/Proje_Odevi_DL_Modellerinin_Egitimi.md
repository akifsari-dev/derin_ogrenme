# YÖNETİM BİLİŞİM SİSTEMLERİ
## 2025-2026 BAHAR YARIYILI — DERİN ÖĞRENME DERSİ — PROJE ÖDEVİ

# Derin Öğrenme Modellerinin Eğitimi ve Performans Analizi

**Adı Soyadı:** ………………………………  
**Öğrenci No:** ………………………………

---

## 1. Kullanılan Veri Seti

**Veri Seti:** ICDAR 2019 SROIE (Scanned Receipt OCR and Information Extraction) — Fiş Alan Görüntü Sınıflandırma

**Sınıf sayısı:** 4 (Address, Company, Date, Total)

**Toplam örnek:** 3.892 kırpılmış alan görüntüsü

| Bölüm | Fiş sayısı | Görüntü sayısı |
|-------|------------|----------------|
| Train | 563 | 2.252 |
| Validation | 63 | 252 |
| Test | 347 | 1.388 |

**Veri Seti Erişim Linki:**
- GitHub (işlenmiş veri seti): https://github.com/akifsari-dev/derin_ogrenme
- Ham kaynak (Kaggle): https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
- Resmi yarışma: https://rrc.cvc.uab.es/?ch=13

**Veri hazırlama özeti:** Ham fiş görüntülerinden `box` ve `entities` annotation dosyaları kullanılarak Company, Address, Date ve Total alanlarına karşılık gelen metin bölgeleri kırpılmıştır. Her örnek `{fiş_id}_{Sınıf}.jpg` formatında ilgili sınıf klasörüne yerleştirilmiştir. Validation seti, train verisinin fiş düzeyinde %90/%10 oranında ayrılmasıyla oluşturulmuştur (`random_state=42`). Test seti orijinal test fişlerinden türetilmiş olup eğitim sürecine dahil edilmemiştir.

---

## 2. Kullanılan Transfer Öğrenme Tabanlı CNN Mimarilerinin Tanıtımı

Çalışmada **2 düşük derinlikli** ve **2 yüksek derinlikli** transfer öğrenme modeli ile **1 özgün CNN** kullanılmıştır.

### 2.1 AlexNet (Düşük-Orta Derinlik)

AlexNet (2012), derin öğrenmede ImageNet yarışmasını kazanan ilk büyük ölçekli CNN mimarisidir. 5 konvolüsyon katmanı ve 3 tam bağlı katmandan oluşur. ReLU aktivasyonu ve dropout ile overfitting azaltılır. Küçük ve orta ölçekli görüntü sınıflandırma görevlerinde hızlı bir başlangıç modeli olarak tercih edilir.

**Kullanım amacı:** Kırpılmış fiş alan görüntülerinde dört sınıflı sınıflandırma için hızlı fine-tuning tabanı.

### 2.2 MobileNetV2 (Düşük Derinlik — Mobil Mimari)

MobileNetV2, derinlik ayrılabilir konvolüsyonlar ve ters artık (inverted residual) blokları kullanarak çok düşük parametre sayısıyla yüksek hız sağlar. Mobil ve gömülü sistemler için tasarlanmış olsa da Colab gibi sınırlı kaynaklı ortamlarda verimli eğitim sunar.

**Kullanım amacı:** Hız/doğruluk dengesi ile fiş alanı sınıflandırması.

### 2.3 VGG16 (Yüksek Derinlik)

VGG16, 13 konvolüsyon ve 3 FC katmanından oluşan derin ve homojen bir mimaridir. Küçük 3×3 filtrelerin art arda kullanılmasıyla güçlü özellik çıkarımı sağlar. Parametre sayısı yüksektir; transfer öğrenme ile güçlü temsil öğrenir.

**Kullanım amacı:** Karmaşık metin düzeni ve görsel dokuyu ayırt etmede güçlü referans model.

### 2.4 ResNet50 (Yüksek Derinlik)

ResNet50, artık bağlantılar (skip connection) ile çok derin ağlarda kaybolan gradyan problemini çözer. 50 katmanlı yapısıyla ImageNet üzerinde yüksek performans gösterir. Fine-tuning ile belge ve fiş görüntülerinde yaygın kullanılır.

**Kullanım amacı:** En güçlü transfer öğrenme adaylarından biri olarak dört alan sınıfını ayırt etme.

> **Not:** Her model için mimari şemaları raporun PDF sürümüne eklenebilir (AlexNet, VGG blokları, ResNet artık bağlantı diyagramı).

---

## 3. Önerilen / Geliştirilen CNN Mimarisinin Tanıtımı

### Custom CNN (Özgün Mimari)

Özgün model, ImageNet ön-eğitimi **kullanılmadan** sıfırdan eğitilmiştir.

**Katman yapısı:**
1. Conv2d(3→32) + ReLU + MaxPool2d
2. Conv2d(32→64) + ReLU + MaxPool2d
3. Conv2d(64→128) + ReLU + MaxPool2d
4. AdaptiveAvgPool2d(4×4) + Flatten
5. Linear(2048→128) + ReLU + Dropout(0.3)
6. Linear(128→4) — çıkış katmanı

**Tasarım yaklaşımı:** Hafif, düşük parametreli bir CNN ile transfer öğrenme modellerine karşı temel (baseline) performans karşılaştırması yapılmıştır. Kırpılmış küçük görüntülerde basit konvolüsyon filtrelerinin yeterli olup olmadığı test edilmiştir.

---

## 4. Hiperparametre Seçimi

### 4.1 Başlangıç Hiperparametre Kombinasyonu (Kombinasyon-A)

| Parametre | Değer |
|-----------|-------|
| Görüntü boyutu | 160×160 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate (FC aşaması) | 1×10⁻³ |
| Learning Rate (Full fine-tune) | 5×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| Epoch (FC + Full) | 2 + 8 = 10 |
| Early Stopping Patience | 3 |
| LR Scheduler | ReduceLROnPlateau |
| Mixed Precision (AMP) | Aktif |
| random_state | 42 |

**Eğitim stratejisi:** İki aşamalı fine-tuning — önce son sınıflandırma katmanı dondurulmuş backbone üzerinde eğitim (Phase-FC), ardından tüm ağ açılarak fine-tune (Phase-Full).

### 4.2 İkinci Hiperparametre Kombinasyonu (Kombinasyon-B) — Önerilen İyileştirme Deneyi

Test performansını artırmak için aşağıdaki kombinasyon önerilmektedir:

| Parametre | Kombinasyon-A | Kombinasyon-B |
|-----------|---------------|---------------|
| Görüntü boyutu | 160 | **224** |
| Batch Size | 64 | **32** |
| Learning Rate (Full) | 5×10⁻⁴ | **1×10⁻⁴** |
| Epoch (FC + Full) | 2 + 8 | **3 + 15** |
| Patience | 3 | **5** |
| Augmentation | Temel | **Güçlendirilmiş** |

**Gerekçe:** Daha büyük giriş çözünürlüğü metin detaylarını korur; düşük öğrenme oranı ve uzun fine-tune test genellemesini iyileştirebilir; güçlü augmentation train-val-test dağılım farkını kısmen azaltır.

### 4.3 Kombinasyon Karşılaştırması (Mevcut Sonuçlar — Kombinasyon-A)

Aşağıdaki tablo, Kombinasyon-A ile elde edilen **validation** ve **test** doğruluklarını özetler:

| Model | Val Acc (%) | Test Acc (%) | Epoch | Süre (dk) |
|-------|-------------|--------------|-------|-----------|
| AlexNet | 80,95 | 38,47 | 5 | 3,3 |
| MobileNetV2 | 92,46 | 33,65 | 10 | 6,4 |
| VGG16 | 84,92 | 34,51 | 5 | 4,0 |
| ResNet50 | 92,46 | 33,36 | 10 | 6,5 |
| **Custom CNN** | **78,97** | **44,52** | 10 | 5,8 |

**Kısa değerlendirme:** Transfer öğrenme modelleri validation setinde yüksek doğruluk (%79–92) elde etmiş; ancak test setinde doğruluk %33–38 bandına düşmüştür. Özgün CNN, validation’da daha düşük olmasına rağmen **test setinde en yüksek doğruluğu (%44,52)** vermiştir. Bu durum, transfer modellerinin validation dağılımına daha fazla uyum sağladığını; özgün CNN’in ise test verisine görece daha iyi genellediğini düşündürmektedir.

---

## 5. Model Eğitim Süreci

### 5.1 Ortam ve Araçlar

- **Platform:** Google Colab (T4 GPU)
- **Framework:** PyTorch, torchvision
- **Veri yükleme:** `ImageFolder` + `DataLoader` (`num_workers=0`)
- **Kayıt:** `outputs/models/{model}_best.pth`

### 5.2 AlexNet Eğitim Sonuçları (Kombinasyon-A)

| Epoch | Aşama | Train Acc | Val Acc |
|-------|-------|-----------|---------|
| 1 | FC | 0,5420 | 0,7897 |
| 2 | FC | 0,8013 | 0,8095 |
| 3 | Full | 0,5161 | 0,6825 |
| 4 | Full | 0,7402 | 0,7619 |
| 5 | Full | 0,8201 | 0,7937 |

**Sonuç:** Val = 80,95%, Test = 38,47%. Full fine-tune başlangıcında (Epoch 3) geçici düşüş gözlenmiş, ardından toparlanma sağlanmıştır. Early stopping 5. epoch’ta devreye girmiştir.

*[Buraya `outputs/figures/alexnet_history.png` eklenecek]*

### 5.3 MobileNetV2 Eğitim Sonuçları (Kombinasyon-A)

| Epoch | Aşama | Train Acc | Val Acc |
|-------|-------|-----------|---------|
| 1–2 | FC | 0,53–0,70 | 0,68–0,70 |
| 3–10 | Full | 0,83–0,99 | 0,90–0,92 |

**Sonuç:** Val = 92,46%, Test = 33,65%. Validation’da en yüksek seviyelerden biri; testte düşük genelleme.

*[Buraya `outputs/figures/mobilenetv2_history.png` eklenecek]*

### 5.4 VGG16 Eğitim Sonuçları (Kombinasyon-A)

**Sonuç:** Val = 84,92%, Test = 34,51%. Full aşamada erken dönemde performans düşüşü ve early stopping (5 epoch).

*[Buraya `outputs/figures/vgg16_history.png` eklenecek]*

### 5.5 ResNet50 Eğitim Sonuçları (Kombinasyon-A)

**Sonuç:** Val = 92,46%, Test = 33,36%. Train accuracy %99’a yaklaşmış; validation-test farkı belirgin.

*[Buraya `outputs/figures/resnet50_history.png` eklenecek]*

### 5.6 Custom CNN Eğitim Sonuçları (Kombinasyon-A)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 0,3103 | 0,4008 |
| 5 | 0,7000 | 0,7302 |
| 7 | 0,7455 | 0,7897 |
| 10 | 0,7862 | 0,7857 |

**Sonuç:** Val = 78,97%, **Test = 44,52%** (en yüksek test performansı).

*[Buraya `outputs/figures/custom_cnn_history.png` eklenecek]*

### 5.7 Eğitim Süreci Genel Yorumu

1. Tüm modeller rastgele tahmin seviyesi (%25) üzerinde sonuç üretmiştir.
2. Transfer modelleri validation’da güçlü, test’te zayıf performans göstermiştir (**dağılım kayması**).
3. Custom CNN daha dengeli bir genelleme profili sergilemiştir.
4. Full fine-tune başlangıcında yaşanan düşüşler, öğrenme oranının veya aşama geçişinin ayarlanmasıyla yumuşatılabilir.

---

## 6. Performans Değerlendirme

### 6.1 Test Seti Doğruluk Karşılaştırması (Kombinasyon-A)

| Model | Doğruluk (Test %) | Val % | Test–Val Farkı |
|-------|-------------------|-------|----------------|
| Custom CNN | **44,52** | 78,97 | −34,45 |
| AlexNet | 38,47 | 80,95 | −42,48 |
| VGG16 | 34,51 | 84,92 | −50,41 |
| MobileNetV2 | 33,65 | 92,46 | −58,81 |
| ResNet50 | 33,36 | 92,46 | −59,10 |

> **Kesinlik, duyarlılık, özgüllük, F1 ve AUC** değerleri `scripts/evaluate_models.py` çalıştırılarak test seti üzerinde hesaplanmalıdır. Colab’da:

```python
!python scripts/evaluate_models.py --data_dir data/processed/dataset --models_dir outputs/models
```

### 6.2 Örnek Performans Tablosu Şablonu (Kombinasyon-A)

*Aşağıdaki tablo, evaluate_models.py çıktısı ile doldurulacaktır:*

| Model | Doğruluk | Duyarlılık | Özgüllük | Kesinlik | F1 Skor | AUC |
|-------|----------|------------|----------|----------|---------|-----|
| AlexNet | 38,47 | — | — | — | — | — |
| MobileNetV2 | 33,65 | — | — | — | — | — |
| VGG16 | 34,51 | — | — | — | — | — |
| ResNet50 | 33,36 | — | — | — | — | — |
| Custom CNN | 44,52 | — | — | — | — | — |

### 6.3 Karışıklık Matrisleri

Her model için test seti üzerinde confusion matrix üretilmiştir:
- `outputs/figures/{model}_confusion_matrix.png`

**Beklenen gözlem:** Address ↔ Company ve Date ↔ Total çiftlerinde karışıklık yüksek olabilir.

### 6.4 ROC Eğrileri ve AUC

Her model için çok sınıflı ROC (one-vs-rest) eğrileri:
- `outputs/figures/{model}_roc.png`

---

## 7. Performans Karşılaştırması ve Genel Değerlendirme

### 7.1 En İyi Model

**Test setine göre en iyi model: Custom CNN (%44,52)**

Validation metriğine göre en iyi: MobileNetV2 / ResNet50 (%92,46) — ancak bu değer test performansını yansıtmamaktadır.

### 7.2 Val–Test Farkının Nedenleri

1. **Veri kaynağı farkı:** Validation, train ile aynı kırpım sürecinden gelir; test ayrı hazırlanmıştır.
2. **Kırpım tutarsızlığı:** Test görüntülerinde alan sınırları farklı olabilir.
3. **Overfitting:** Transfer modelleri train+val dağılımına yoğun uyum sağlamış olabilir (train acc %99’a yakın).
4. **Küçük validation seti:** 252 örnek, yüksek val skorunu şişirebilir.

### 7.3 Modellerin Güçlü ve Zayıf Yönleri

| Model | Güçlü Yön | Zayıf Yön |
|-------|-----------|-----------|
| AlexNet | Hızlı eğitim, orta val performansı | Test genellemesi sınırlı |
| MobileNetV2 | Yüksek val acc, hızlı inference | Test’te en düşük grupta |
| VGG16 | Güçlü özellik çıkarımı | Erken durma, yüksek val-test farkı |
| ResNet50 | En derin temsil, yüksek val | Aşırı uyum (train %99), düşük test |
| Custom CNN | **En iyi test genellemesi** | Val ve train acc daha düşük |

---

## 8. Test Performansını İyileştirmek İçin Öneriler

### 8.1 Veri Kalitesi (Öncelik: Yüksek)

1. Train ve test kırpımlarının **aynı script** ile üretildiğinden emin olun.
2. Rastgele 20 test görüntüsünü görsel kontrol edin; yanlış sınıf/eksik kırpım düzeltin.
3. `box` + `entities` eşleştirmede fuzzy matching kullanın (metin tam eşleşmezse).

### 8.2 Eğitim Stratejisi (Öncelik: Yüksek)

1. **Kombinasyon-B** ile yeniden eğitim (224px, LR=1e-4, 18 epoch).
2. Güçlü augmentation: `RandomRotation`, `ColorJitter`, `RandomAffine`.
3. Label smoothing veya dropout artırımı (overfitting azaltma).
4. Class-weighted loss (dengesiz sınıflar varsa).

### 8.3 Model Seçimi (Öncelik: Orta)

1. Test metriğine göre model seçin (val değil).
2. Custom CNN üzerinde hiperparametre taraması yapın.
3. En iyi 2 modelin ensemble (soft voting) denemesi.

### 8.4 Değerlendirme (Öncelik: Orta)

1. Confusion matrix ile hangi sınıf çiftlerinin karıştığını belirleyin.
2. Test seti üzerinde sınıf bazlı precision/recall raporlayın.
3. Hatalı tahmin örneklerini rapora ekleyin (görsel analiz).

### 8.5 Beklenen İyileşme Hedefleri

| Aşama | Hedef Test Acc |
|-------|----------------|
| Mevcut (Kombinasyon-A) | %33–45 |
| Kırpım düzeltme + Kombinasyon-B | %50–65 |
| Ensemble + veri temizliği | %60+ |

---

## 9. Çapraz Doğrulama (Cross Validation)

Ödev gereği, **test setinde en yüksek performansı veren Custom CNN** ile çapraz doğrulama uygulanacaktır.

**Yöntem:** Stratified K-Fold (k=5), fiş düzeyinde bölme (aynı fişin 4 görüntüsü aynı fold’da).

**Değerlendirilecek metrikler:** Accuracy, Precision, Recall, Specificity, F1-Score, AUC

### 9.1 Örnek Tablo Şablonu (k=5, Custom CNN)

| Fold | Doğruluk | Duyarlılık | Özgüllük | Kesinlik | F1 Skor | AUC |
|------|----------|------------|----------|----------|---------|-----|
| Fold 1 | — | — | — | — | — | — |
| Fold 2 | — | — | — | — | — | — |
| Fold 3 | — | — | — | — | — | — |
| Fold 4 | — | — | — | — | — | — |
| Fold 5 | — | — | — | — | — | — |
| **Ortalama** | — | — | — | — | — | — |

> Colab’da cross-validation kodu çalıştırılarak doldurulacaktır.

### 9.2 Çapraz Doğrulama Değerlendirmesi

Fold’lar arası düşük varyans → model kararlı. Yüksek varyans → veri kalitesi veya hiperparametre ayarı gözden geçirilmeli. CV ortalaması, tek seferlik val skorundan daha güvenilir bir genelleme göstergesidir.

---

## 10. Sonuç

Bu çalışmada ICDAR 2019 SROIE veri setinden türetilen kırpılmış alan görüntüleri üzerinde beş farklı CNN mimarisi eğitilmiştir. AlexNet, MobileNetV2, VGG16, ResNet50 transfer öğrenme modelleri validation setinde %79–92 doğruluk elde etmiş; ancak test setinde %33–38 bandında kalmıştır. Özgün Custom CNN, validation’da daha düşük olmasına rağmen test setinde **%44,52** ile en iyi genelleme performansını göstermiştir.

Elde edilen bulgular, model seçiminde yalnızca validation accuracy’ye değil **test seti performansına** odaklanılması gerektiğini; ayrıca train ve test kırpım süreçlerinin tutarlılığının kritik önem taşıdığını göstermektedir. Kombinasyon-B hiperparametreleri, güçlü augmentation ve veri kalitesi iyileştirmeleri ile test doğruluğunun %50+ seviyesine çıkarılabileceği öngörülmektedir.

---

## 11. Kaynaklar

1. Huang, Z., et al. (2019). ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. ICDAR.
2. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet). NIPS.
3. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG). arXiv:1409.1556.
4. He, K., et al. (2016). Deep Residual Learning for Image Recognition (ResNet). CVPR.
5. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
6. SROIE Dataset: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
7. Proje GitHub: https://github.com/akifsari-dev/derin_ogrenme

---

## EK: Veri Seti Tanıtım Dökümanı

*Önceki veri seti hazırlama raporu ve `split_report.txt` bu bölümde birleştirilerek PDF’e eklenecektir.*

---

### Raporu PDF’e Dönüştürme ve Tamamlama Checklist

- [ ] Ad-soyad ve öğrenci no gir
- [ ] Eğitim grafiklerini (`outputs/figures/*_history.png`) ilgili bölümlere ekle
- [ ] `evaluate_models.py` çalıştır → confusion matrix, ROC, metrik tablosu
- [ ] Kombinasyon-B deneyini yap (ikinci hiperparametre tablosu)
- [ ] Custom CNN ile k=5 cross-validation tablosunu doldur
- [ ] Veri seti tanıtım PDF’ini sona birleştir
- [ ] Tüm dökümanı tek PDF olarak kaydet
