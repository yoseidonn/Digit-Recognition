# Digit Recognition

Bu proje, kamera görüntüsünden el yazısı rakamları gerçek zamanlı olarak tanıyan bir CNN modeli kullanmaktadır. Model MNIST veri seti ile eğitilmiştir.

## Kurulum

### Sanal Ortam Kurulumu

Sanal ortamı oluşturun:
```bash
python3 -m venv env
``` 
veya
```bash
virtualenv env
```

Sanal ortamı aktif hale getirin:
```bash
source env/bin/activate
```

### Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

### Projeyi çalıştırmak:

- Model eğitimi için:
```bash
cd src
python train.py
```

- Kamera ile gerçek zamanlı tahmin için:
```bash
cd src
python predict.py
```

## Proje Yapısı

```
project/
├── data/           # MNIST veri seti otomatik olarak buraya indirilecek
├── models/         # Eğitilmiş model dosyaları
├── src/
│   ├── model.py    # CNN model tanımı
│   ├── train.py    # Model eğitimi
│   └── predict.py  # Gerçek zamanlı tahmin
└── requirements.txt
```

## Kullanım

1. Önce modeli eğitin (`train.py`). Bu işlem MNIST veri setini otomatik olarak indirecek ve modeli eğitecektir.
2. Eğitim tamamlandıktan sonra, `predict.py` ile kamera görüntüsünden gerçek zamanlı tahmin yapabilirsiniz.
3. Programı sonlandırmak için 'q' tuşuna basın.

## Model Mimarisi

- 2 Konvolüsyon katmanı
- 2 MaxPooling katmanı
- 2 Tam bağlantılı katman
- Dropout (0.25) düzenleştirme
- ReLU aktivasyon fonksiyonları
- Softmax çıkış katmanı 