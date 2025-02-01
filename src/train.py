"""
MNIST veri seti kullanarak rakam tanıma modelini eğiten script.
Geliştirilmiş eğitim süreci ve veri artırma teknikleri içerir.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitRecognitionCNN
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, device, train_loader, optimizer, epoch, scheduler):
    """Her epoch için modeli eğitir ve kayıp değerini gösterir."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # İstatistikleri hesapla
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    # Epoch sonunda scheduler'ı güncelle
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    return avg_loss

def test_model(model, device, test_loader):
    """Test veri seti üzerinde modelin performansını değerlendirir."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Eğitim parametreleri
    batch_size = 128  # Batch size artırıldı
    epochs = 30       # Epoch sayısı artırıldı
    learning_rate = 0.01
    
    # GPU varsa kullan, yoksa CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Geliştirilmiş veri dönüşümleri
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rastgele döndürme
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Rastgele kaydırma
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Veri setlerini hazırla
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # Model ve optimizasyon
    model = DigitRecognitionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Eğitim döngüsü
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        avg_loss = train_model(model, device, train_loader, optimizer, epoch, scheduler)
        accuracy = test_model(model, device, test_loader)
        
        # En iyi modeli kaydet
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join('..', 'models', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main() 