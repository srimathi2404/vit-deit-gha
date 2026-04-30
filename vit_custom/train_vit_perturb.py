import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import time
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4

os.makedirs("outputs_perturb_vit", exist_ok=True)

# -----------------------
# Data
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model
# -----------------------
model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=6,
    heads=4,
    mlp_dim=512,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Perturbations
# -----------------------
def patch_shuffle(images, patch_size=4):
    B, C, H, W = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    idx = torch.randperm(patches.size(2))
    patches = patches[:, :, idx]
    patches = patches.view(B, C, H // patch_size, W // patch_size, patch_size, patch_size)
    patches = patches.permute(0,1,2,4,3,5).contiguous()
    return patches.view(B, C, H, W)

def add_noise(images, std=0.1):
    return torch.clamp(images + torch.randn_like(images)*std, 0, 1)

def random_mask(images, ratio=0.25):
    mask = (torch.rand_like(images[:, :1]) > ratio).float()
    return images * mask

# -----------------------
# Train
# -----------------------
def train():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

# -----------------------
# Robustness Test
# -----------------------
def test_robust():
    model.eval()
    results = {"clean":0, "shuffle":0, "noise":0, "mask":0}
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            preds = model(images).argmax(1)
            results["clean"] += preds.eq(labels).sum().item()

            preds = model(patch_shuffle(images.clone())).argmax(1)
            results["shuffle"] += preds.eq(labels).sum().item()

            preds = model(add_noise(images.clone())).argmax(1)
            results["noise"] += preds.eq(labels).sum().item()

            preds = model(random_mask(images.clone())).argmax(1)
            results["mask"] += preds.eq(labels).sum().item()

            total += labels.size(0)

    for k in results:
        results[k] = 100 * results[k] / total

    return results

# -----------------------
# Train loop
# -----------------------
for epoch in range(EPOCHS):
    start = time.time()
    train()
    print(f"Epoch {epoch+1} done in {time.time()-start:.2f}s")

# -----------------------
# Robustness
# -----------------------
res = test_robust()

print("\nViT Robustness:")
for k,v in res.items():
    print(f"{k}: {v:.2f}%")
# -----------------------
# Accuracy Drop (IMPORTANT)
# -----------------------
print("\n--- Accuracy Drop from Clean ---")

clean_acc = res["clean"]

for k in ["shuffle", "noise", "mask"]:
    drop = clean_acc - res[k]
    print(f"{k}: -{drop:.2f}%")

# Plot
plt.bar(res.keys(), res.values())
plt.title("ViT Robustness")
plt.savefig("outputs_perturb_vit/robustness.png")


with open("outputs_perturb_vit/robustness.txt", "w") as f:
    f.write("Accuracy:\n")
    for k, v in res.items():
        f.write(f"{k}: {v:.2f}%\n")

    f.write("\nDrop from clean:\n")
    for k in ["shuffle", "noise", "mask"]:
        drop = clean_acc - res[k]
        f.write(f"{k}: -{drop:.2f}%\n")