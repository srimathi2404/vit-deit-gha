import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import deit_tiny_patch16_224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATA
# =========================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10("/tmp/data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10("/tmp/data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128)

# =========================
# MODEL (official DeiT)
# =========================
model = deit_tiny_patch16_224(pretrained=False)

# Change classifier to CIFAR
model.head = nn.Linear(model.head.in_features, 10)

model.to(DEVICE)

# =========================
# LOSS + OPTIM
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# =========================
# TRAIN
# =========================
def train():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================
# TEST
# =========================
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# =========================
# LOOP
# =========================
for epoch in range(20):
    loss = train()
    acc = test()

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")
    