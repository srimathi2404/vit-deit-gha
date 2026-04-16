import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from models import deit_tiny_patch16_224

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATA
# =========================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# =========================
# STUDENT (DeiT)
# =========================
model = deit_tiny_patch16_224(pretrained=False)
model.head = nn.Linear(model.head.in_features, 10)
model = model.to(device)

# =========================
# TEACHER (ResNet18)
# =========================
teacher_model = models.resnet18(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)
teacher_model = teacher_model.to(device)

teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# =========================
# OPTIMIZER
# =========================
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# =========================
# DISTILLATION LOSS
# =========================
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    ce_loss = F.cross_entropy(student_logits, labels)

    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    return alpha * kd_loss + (1 - alpha) * ce_loss

# =========================
# TRAIN
# =========================
def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # student output
        outputs = model(images)

        # teacher output
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        loss = distillation_loss(outputs, teacher_outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(train_loader), acc

# =========================
# TEST
# =========================
from sklearn.metrics import classification_report

def test():
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    return acc

# =========================
# LOOP
# =========================
best_acc = 0

for epoch in range(30):
    train_loss, train_acc = train()
    test_acc = test()

    print(f"Epoch [{epoch+1}/30]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%\n")

    if test_acc > best_acc:
        best_acc = test_acc

print(f"Best Test Accuracy: {best_acc:.2f}%")