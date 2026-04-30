import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from vit import ViTSeg

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
IMAGE_SIZE = 224
NUM_CLASSES = 3   # pet / border / background

OUTPUT_DIR = "/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Transforms
# -----------------------
transform_img = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def transform_mask(mask):
    # Resize
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    mask = np.array(mask)

    # Oxford labels: {1,2,3}
    # Convert to {0,1,2}
    mask = mask - 1

    return torch.from_numpy(mask).long()

# -----------------------
# Dataset Wrapper
# -----------------------
class PetSegDataset(OxfordIIITPet):
    def __init__(self, root, split):
        super().__init__(
            root=root,
            split=split,
            target_types="segmentation",
            download=True
        )

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        return transform_img(img), transform_mask(mask)

# Load datasets
train_dataset = PetSegDataset(root="/data", split="trainval")
val_dataset   = PetSegDataset(root="/data", split="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------
# Model
# -----------------------
model = ViTSeg(
    image_size=IMAGE_SIZE,
    patch_size=16,
    dim=256,
    depth=6,
    heads=4,
    mlp_dim=512,
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Metrics
# -----------------------
def pixel_accuracy(pred, mask):
    pred = pred.argmax(dim=1)
    correct = (pred == mask).float().sum()
    total = torch.numel(mask)
    return (correct / total).item()

def compute_miou(pred, mask, num_classes=NUM_CLASSES):
    pred = pred.argmax(dim=1)

    ious = []
    for cls in range(num_classes):
        pred_i = (pred == cls)
        mask_i = (mask == cls)

        intersection = (pred_i & mask_i).sum().item()
        union = (pred_i | mask_i).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious) if ious else 0

# -----------------------
# Train
# -----------------------
def train_one_epoch():
    model.train()
    total_loss = 0
    total_acc = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)  # (B, C, H, W)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += pixel_accuracy(outputs, masks)

    return total_loss / len(train_loader), total_acc / len(train_loader)

# -----------------------
# Eval
# -----------------------
def evaluate():
    model.eval()
    total_acc = 0
    total_miou = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            outputs = model(imgs)

            total_acc += pixel_accuracy(outputs, masks)
            total_miou += compute_miou(outputs, masks)

    return total_acc / len(val_loader), total_miou / len(val_loader)

# -----------------------
# Training loop
# -----------------------
train_losses, train_accs, val_accs, val_mious = [], [], [], []

for epoch in range(EPOCHS):
    start = time.time()

    loss, train_acc = train_one_epoch()
    val_acc, val_miou = evaluate()

    train_losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    val_mious.append(val_miou)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | mIoU: {val_miou:.4f} | Time: {time.time()-start:.1f}s")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_{epoch}.pth")

# -----------------------
# Plots
# -----------------------
plt.plot(train_losses)
plt.title("Loss")
plt.savefig(f"{OUTPUT_DIR}/loss.png")
plt.close()

plt.plot(train_accs, label="train")
plt.plot(val_accs, label="val")
plt.legend()
plt.title("Accuracy")
plt.savefig(f"{OUTPUT_DIR}/accuracy.png")
plt.close()

plt.plot(val_mious)
plt.title("mIoU")
plt.savefig(f"{OUTPUT_DIR}/miou.png")
plt.close()

print("Training complete.")