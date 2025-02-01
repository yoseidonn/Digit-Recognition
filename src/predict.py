"""
Kamera görüntüsünden gerçek zamanlı rakam tanıma yapan script.
Görüntüyü işleyip MNIST formatına uygun hale getirir ve
eğitilmiş model ile tahmin yapar.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from model import DigitRecognitionCNN
import os


class DigitPredictor:
    """Rakam tanıma için gerekli işlemleri yapan sınıf."""
    
    def __init__(self, model_path):
        """Modeli yükler ve gerekli ayarları yapar."""
        # Donanım seçimi
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Eğitilmiş modeli yükle
        self.model = DigitRecognitionCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # MNIST veri setine uygun görüntü dönüşümleri
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(self, image):
        """Kamera görüntüsünü MNIST formatına uygun hale getirir."""
        # Siyah-beyaz dönüşümü
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü temizleme
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Rakamı belirginleştirme
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return thresh
    
    def predict(self, image):
        """Görüntüden rakam tahmini yapar."""
        # Görüntü ön işleme
        processed = self.preprocess_image(image)
        
        # MNIST boyutuna getir
        resized = cv2.resize(processed, (28, 28), interpolation=cv2.INTER_AREA)
        
        # PyTorch tensörüne çevir
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        
        # Model tahmini
        with torch.no_grad():
            output = self.model(tensor)
            prediction = output.argmax(dim=1, keepdim=True)
            probability = torch.exp(output).max().item()
        
        return prediction.item(), probability

def main():
    """Ana program döngüsü."""
    # Kamera ayarları
    cap = cv2.VideoCapture(0)
    
    # Tahmin için model yükle
    model_path = os.path.join('..', 'models', 'best_model.pth')
    predictor = DigitPredictor(model_path)
    
    # Gerçek zamanlı tahmin döngüsü
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Görüntüyü işle ve tahmin yap
        digit, prob = predictor.predict(frame)
        
        # Sonucu ekranda göster
        cv2.putText(frame, f'Digit: {digit} ({prob:.2f})', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Digit Recognition', frame)
        
        # Çıkış kontrolü
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 