import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import classification_report

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
NUM_CLASSES = 10

# =========================
# DATA
# =========================
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root="/tmp/data", train=True, download=True, transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root="/tmp/data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =========================
# MODEL (ViT from scratch)
# =========================
model = timm.create_model(
    'vit_tiny_patch16_224',
    pretrained=False,
    num_classes=NUM_CLASSES
).to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# =========================
# TRAIN FUNCTION
# =========================
def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    return total_loss / len(train_loader), acc

# =========================
# EVAL FUNCTION
# =========================
def evaluate():
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100. * correct / total
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    return acc

# =========================
# TRAIN LOOP
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train()
    test_acc = evaluate()

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%\n")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "/tmp/vit_best.pth")

print(f"Best Test Accuracy: {best_acc:.2f}%")