"""
Giriş olarak 28x28 boyutunda gri tonlamalı görüntüler alır ve 0-9 arası rakamları sınıflandırır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        # İlk katman: 1 kanallı girişten 32 özellik haritası çıkarır
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # İkinci katman: 32 kanallı girişten 64 özellik haritası çıkarır
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Özellik haritalarını düz bir vektöre çevirip sınıflandırma yapar
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Aşırı öğrenmeyi önlemek için dropout
        self.dropout = nn.Dropout2d(0.25)
        
    def forward(self, x):
        # Her konvolüsyon sonrası aktivasyon ve boyut küçültme
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Çok boyutlu tensörü düz hale getir
        x = x.view(-1, 64 * 5 * 5)
        
        # Son katmanlarda sınıflandırma
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) 