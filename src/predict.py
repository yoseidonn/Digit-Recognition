"""
Kamera görüntüsünden veya tek görsellerden rakam tanıma yapan script.
Görüntüyü işleyip MNIST formatına uygun hale getirir ve
eğitilmiş model ile tahmin yapar. Birden fazla rakam varsa hepsini tespit eder.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from model import DigitRecognitionCNN
import os
import argparse
from pathlib import Path


class DigitPredictor:
    """Rakam tanıma için gerekli işlemleri yapan sınıf."""
    
    def __init__(self, model_path):
        """Modeli yükler ve gerekli ayarları yapar."""
        # Donanım seçimi
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Eğitilmiş modeli yükle
        self.model = DigitRecognitionCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # MNIST veri setine uygun görüntü dönüşümleri
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def find_digits(self, image):
        """Görüntüdeki tüm rakamları bulur."""
        # Görüntüyü ön işle
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptif eşikleme uygula (değişken ışık koşulları için daha iyi)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morfolojik işlemler
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # Konturları bul
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        digit_regions = []
        for contour in contours:
            # Çok küçük alanları filtrele
            area = cv2.contourArea(contour)
            if area < 100:  # minimum alan eşiği
                continue
                
            # Sınırlayıcı kutu koordinatlarını al
            x, y, w, h = cv2.boundingRect(contour)
            
            # Kare bölge oluştur (en-boy oranını koru)
            size = max(w, h)
            x_center = x + w // 2
            y_center = y + h // 2
            x = max(0, x_center - size // 2)
            y = max(0, y_center - size // 2)
            size = min(size, min(image.shape[1] - x, image.shape[0] - y))
            
            digit_regions.append((x, y, size, size))
            
        return thresh, digit_regions
    
    def preprocess_region(self, image, region):
        """Belirli bir bölgeyi MNIST formatına uygun hale getirir."""
        x, y, w, h = region
        digit_image = image[y:y+h, x:x+w]
        
        # Görüntüyü MNIST boyutuna getir
        digit_image = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)
        
        return digit_image

    def predict_region(self, image):
        """Görüntüden rakam tahmini yapar."""
        # PyTorch tensörüne çevir
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Model tahmini
        with torch.no_grad():
            output = self.model(tensor)
            prediction = output.argmax(dim=1, keepdim=True)
            probability = torch.exp(output).max().item()
        
        return prediction.item(), probability

    def process_image(self, image, draw_results=True):
        """Görüntüdeki tüm rakamları işler ve sonuçları görselleştirir."""
        # Orijinal görüntüyü kopyala
        output_image = image.copy()
        
        # Rakamları bul
        thresh, regions = self.find_digits(image)
        
        results = []
        for region in regions:
            x, y, w, h = region
            
            # Bölgeyi ön işle
            digit_image = self.preprocess_region(thresh, region)
            
            # Tahmin yap
            digit, prob = self.predict_region(digit_image)
            results.append((digit, prob, region))
            
            # Sonuçları görselleştir
            if draw_results:
                # Sınırlayıcı kutu
                color = (0, 255, 0) if prob > 0.7 else (0, 165, 255)
                cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
                
                # Tahmin ve güven oranı
                text = f"{digit} ({prob:.2f})"
                cv2.putText(output_image, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image, results

def process_single_image(predictor, image_path):
    """Tek bir görüntüyü işler ve sonuçları gösterir."""
    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Görüntüyü işle
    output_image, results = predictor.process_image(image)
    
    # Sonuçları yazdır
    print(f"\nResults for {image_path}:")
    for digit, prob, _ in results:
        print(f"Detected digit: {digit} (confidence: {prob:.2f})")
    
    # Görüntüleri göster
    cv2.imshow('Original', image)
    cv2.imshow('Detected Digits', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Ana program."""
    parser = argparse.ArgumentParser(description='Digit Recognition')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--camera', action='store_true', help='Use camera input')
    args = parser.parse_args()
    
    # Model yükle
    model_path = os.path.join('..', 'models', 'best_model.pth')
    predictor = DigitPredictor(model_path)
    
    if args.image:
        # Tek görüntü işle
        process_single_image(predictor, args.image)
    
    elif args.camera:
        # Kamera girişi
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Görüntüyü işle
            output_frame, results = predictor.process_image(frame)
            
            # Sonuçları göster
            cv2.imshow('Digit Recognition', output_frame)
            
            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Please specify either --image or --camera")

if __name__ == '__main__':
    main() 