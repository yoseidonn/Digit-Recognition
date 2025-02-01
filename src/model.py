"""
Geliştirilmiş CNN modeli - MNIST rakam tanıma için optimize edilmiş
Giriş: 28x28 boyutunda gri tonlamalı görüntüler
Çıkış: 0-9 arası rakam tahminleri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        # Daha derin konvolüsyon katmanları
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Dropout katmanları
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
    def forward(self, x):
        # Konvolüsyon bloğu 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Konvolüsyon bloğu 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Konvolüsyon bloğu 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Tam bağlantılı katmanlar
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1) 