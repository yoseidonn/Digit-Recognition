# Digit Recognition

Bu proje, görüntülerdeki el yazısı rakamları tanıyan bir yapay zeka sistemidir. Hem kamera görüntüsünden gerçek zamanlı tanıma yapabilir, hem de tek görüntüler üzerinde çalışabilir. Aynı anda birden fazla rakamı tespit edip tanıyabilir.

## Özellikler

- Çoklu rakam tespiti ve tanıma
- Gerçek zamanlı kamera desteği
- Tek görüntü analizi
- Güven oranı göstergesi
- Adaptif görüntü işleme
- Görsel ve konsol çıktıları

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Modeli eğitin:
```bash
cd src
python train.py
```

## Kullanım

### 1. Kamera ile Gerçek Zamanlı Tanıma
```bash
python predict.py --camera
```
- Kameraya rakam gösterin
- Yeşil kutular: Yüksek güvenle tanınan rakamlar (>0.7)
- Turuncu kutular: Düşük güvenle tanınan rakamlar (<0.7)
- Çıkmak için 'q' tuşuna basın

### 2. Tek Görüntü Analizi
```bash
python predict.py --image path/to/image.jpg
```
- Program görüntüdeki tüm rakamları tespit eder
- Her rakam için güven oranını gösterir
- Sonuçları hem görsel hem konsol çıktısı olarak verir

## Proje Yapısı

```
.
├── src/
│   ├── model.py    # CNN model tanımı
│   ├── train.py    # Model eğitimi
│   └── predict.py  # Rakam tanıma sistemi
├── data/           # MNIST veri seti
├── models/         # Eğitilmiş model dosyaları
└── requirements.txt
```

## En İyi Sonuç İçin Öneriler

1. **Görüntü Kalitesi**:
   - Beyaz kağıt üzerine siyah kalemle yazın
   - Rakamları net ve okunaklı yazın
   - İyi aydınlatma kullanın

2. **Rakam Yazımı**:
   - Rakamları birbirinden ayrık yazın
   - Çok küçük veya çok büyük yazmaktan kaçının
   - MNIST stiline benzer şekilde yazın

3. **Kamera Kullanımı**:
   - Kamerayı sabit tutun
   - Kağıdı düz tutun
   - Gölge oluşturmaktan kaçının

## Hata Durumları

1. **Anomali / Düşük Güven Oranı (Turuncu Kutu)**:
   - Rakamı daha net yazın
   - Aydınlatmayı iyileştirin
   - Kağıdı kameraya daha yakın tutun

2. **Tespit Edilemeyen Rakamlar**:
   - Kontrastı artırın
   - Rakamı büyük yazın
   - Arka plan temizliğinden emin olun

## Teknik Detaylar

Daha detaylı teknik bilgi için `DOCUMENTATION.md` dosyasına bakın. 