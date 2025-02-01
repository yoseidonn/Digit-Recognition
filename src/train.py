"""
MNIST veri seti kullanarak rakam tanıma modelini eğiten script.
Model performansını test eder ve en iyi modeli kaydet.
Eğitim sırasında ilerleme durumu ve doğruluk oranı gösterilir.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitRecognitionCNN
import os


def train_model(model, device, train_loader, optimizer, epoch):
    """Her epoch için modeli eğitir ve kayıp değerini gösterir."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
    # Temel eğitim ayarları
    batch_size = 64  # Her adımda işlenecek görüntü sayısı
    epochs = 10      # Tüm veri setinin kaç kez işleneceği
    learning_rate = 0.01  # Öğrenme hızı
    
    # GPU varsa kullan, yoksa CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MNIST veri seti için standart normalizasyon
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Eğitim ve test veri setlerini hazırla
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model ve optimizasyon ayarları
    model = DigitRecognitionCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Eğitim döngüsü
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)
        accuracy = test_model(model, device, test_loader)
        
        # Eğer bu en iyi sonuçsa modeli kaydet
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join('..', 'models', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main() 