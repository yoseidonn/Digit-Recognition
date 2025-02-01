# El Yazısı Rakam Tanıma Projesi - Teknik Dokümantasyon

## Proje Yapısı

```
.
├── src/           # Kaynak kodlar
├── data/          # MNIST veri seti
├── models/        # Eğitilmiş model dosyaları
└── requirements.txt
```

## Teknik Detaylar

### CNN (Convolutional Neural Network) Mimarisi

#### 1. Konvolüsyon Katmanları
- **Ne İşe Yarar?** Görüntüdeki özellikleri (kenarlar, dokular, desenler) tespit eder
- **Nasıl Çalışır?** 
  - Küçük bir filtre (kernel) görüntü üzerinde kaydırılır
  - Her pozisyonda piksel grupları üzerinde matematiksel işlemler yapılır
  - Sonuçta bir özellik haritası (feature map) oluşur
- **Projemizdeki Kullanımı:**
  ```python
  self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 1 kanal girişten 32 özellik haritası
  self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 32 kanaldan 64 özellik haritası
  ```

#### 2. Havuzlama (Pooling) Katmanları
- **Ne İşe Yarar?** Görüntü boyutunu küçültür ve önemli özellikleri korur
- **Nasıl Çalışır?**
  - Belirli bir bölgedeki en yüksek değeri alır (MaxPooling)
  - Görüntü boyutu küçülür ama önemli özellikler korunur
  - İşlem yükü azalır
- **Projemizdeki Kullanımı:**
  ```python
  x = F.max_pool2d(x, 2)  # 2x2'lik bölgeleri 1 piksele indirir
  ```

#### 3. Tam Bağlantılı Katmanlar
- **Ne İşe Yarar?** Özellik haritalarını kullanarak sınıflandırma yapar
- **Nasıl Çalışır?**
  - Özellik haritaları düz bir vektöre çevrilir
  - Her nöron bir önceki katmandaki tüm nöronlara bağlanır
  - Son katman sınıf sayısı kadar nöron içerir (bizim durumumuzda 10 rakam)
- **Projemizdeki Kullanımı:**
  ```python
  self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Özellik haritalarını 128 nörona bağla
  self.fc2 = nn.Linear(128, 10)          # 10 rakam için çıkış katmanı
  ```

### Aktivasyon Fonksiyonları

#### ReLU (Rectified Linear Unit)
- **Ne İşe Yarar?** Doğrusal olmayan özellikleri öğrenmeyi sağlar
- **Nasıl Çalışır?**
  - Negatif değerleri 0 yapar
  - Pozitif değerleri aynen geçirir
  - f(x) = max(0, x)
- **Neden Kullanıyoruz?**
  - Hesaplaması kolay
  - Gradyan kaybını önler
  - Eğitimi hızlandırır

### Dropout
- **Ne İşe Yarar?** Modelin ezberlememesini (overfitting) önler
- **Nasıl Çalışır?**
  - Eğitim sırasında rastgele nöronları devre dışı bırakır
  - Her epoch'ta farklı nöronlar kapatılır
  - Test sırasında tüm nöronlar aktif olur
- **Projemizdeki Kullanımı:**
  ```python
  self.dropout = nn.Dropout2d(0.25)  # Nöronların %25'ini kapat
  ```

### Veri Ön İşleme

#### Normalizasyon
- **Ne İşe Yarar?** Görüntü değerlerini belirli bir aralığa getirir
- **Nasıl Çalışır?**
  ```python
  transforms.Normalize((0.1307,), (0.3081,))  # MNIST için ortalama ve standart sapma
  ```

#### Görüntü İşleme
- **Gri Tonlama:** Renkli görüntüyü tek kanala indirir
- **Gaussian Blur:** Gürültüyü azaltır
- **Thresholding:** Rakamı arka plandan ayırır

### Model Eğitimi

#### Hiperparametreler
- **Batch Size:** 64 (Her adımda işlenen görüntü sayısı)
- **Learning Rate:** 0.01 (Öğrenme hızı)
- **Epochs:** 10 (Veri setinin kaç kez işleneceği)

#### Optimizasyon
- **SGD (Stochastic Gradient Descent)**
  - Kayıp fonksiyonunu minimize eder
  - Her batch sonrası ağırlıkları günceller

#### Kayıp Fonksiyonu
- **CrossEntropyLoss**
  - Çok sınıflı sınıflandırma için uygun
  - Model çıktısını olasılık dağılımına çevirir