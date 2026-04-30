import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gha_experiments.vit_rpb import ViT
import time
import matplotlib.pyplot as plt
import os

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4

os.makedirs("outputs_gha", exist_ok=True)

# -----------------------
# Data
# -----------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
    dropout=0.1,
    emb_dropout=0.1
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Logs
# -----------------------
train_losses = []
train_accs = []
test_accs = []
epoch_times = []

# -----------------------
# Train
# -----------------------
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

# -----------------------
# Test
# -----------------------
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

    return 100. * correct / total

# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):
    start = time.time()

    train_loss, train_acc = train()
    test_acc = test()

    epoch_time = time.time() - start

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Time: {epoch_time:.2f}s")
    print("-" * 40)

# -----------------------
# Save plots
# -----------------------

# Loss curve
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs_gha/loss_curve_rpb.png")

# Accuracy curve
plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs_gha/accuracy_curve_rpb.png")

# Time per epoch
plt.figure()
plt.plot(epoch_times, label="Time per epoch")
plt.title("Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Seconds")
plt.legend()
plt.savefig("outputs_gha/time_curve_rpb.png")

print("Plots saved in outputs_gha/ folder")



# # -----------------------
# # Save Attention Map
# # -----------------------
# def save_attention_map():
#     model.eval()

#     images, _ = next(iter(test_loader))
#     images = images.to(DEVICE)

#     with torch.no_grad():
#         _, attn_maps = model(images, return_attn=True)

#     # Take last layer, first head
#     attn = attn_maps[-1][0, 0]  # (N, N)

#     # Remove CLS token
#     attn = attn[1:, 1:]

#     # Average attention across tokens
#     attn = attn.mean(dim=0)

#     # CIFAR: 32x32 with patch=4 → 8x8
#     attn = attn.reshape(8, 8).cpu()

#     import matplotlib.pyplot as plt
#     plt.imshow(attn, cmap='viridis')
#     plt.colorbar()
#     plt.title("Attention Map")
#     plt.savefig("outputs_gha/attention_map.png")
#     plt.close()

# save_attention_map()

# -----------------------
# Attention + Analysis
# -----------------------
def analyze_attention():
    model.eval()

    images, _ = next(iter(test_loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        _, attn_maps = model(images, return_attn=True)

    # Last layer
    attn = attn_maps[-1]  # (B, heads, N, N)

    # Take first image
    attn = attn[0]  # (heads, N, N)

    # Average over heads (IMPORTANT FIX)
    attn = attn.mean(dim=0)  # (N, N)

    # Remove CLS token
    attn = attn[1:, 1:]  # (64, 64)

    # -----------------------
    # 1. Attention Heatmap
    # -----------------------
    attn_heat = attn.mean(dim=0)  # average attention received per token

    attn_heat = attn_heat.reshape(8, 8).cpu()

    plt.figure()
    plt.imshow(attn_heat, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Map (Averaged)")
    plt.axis('off')
    plt.savefig("outputs_gha/attention_map_rpb.png")
    plt.close()

    # -----------------------
    # 2. Histogram
    # -----------------------
    attn_values = attn.flatten().cpu().numpy()

    plt.figure()
    plt.hist(attn_values, bins=50)
    plt.title("Attention Value Distribution")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.savefig("outputs_gha/attention_histogram_rpb.png")
    plt.close()

    # -----------------------
    # 3. Sparsity metric
    # -----------------------
    threshold = 0.01
    sparsity = (attn < threshold).float().mean().item()

    print(f"Attention Sparsity (< {threshold}): {sparsity*100:.2f}%")

    # -----------------------
    # 4. Top-k concentration
    # -----------------------
    topk = torch.topk(attn.flatten(), k=int(0.1 * attn.numel())).values
    concentration = topk.sum() / attn.sum()

    print(f"Top 10% Attention Mass: {concentration.item()*100:.2f}%")

    # -----------------------
    # 5. Save raw stats
    # -----------------------
    with open("outputs_gha/attention_stats_rpb.txt", "w") as f:
        f.write(f"Sparsity (<{threshold}): {sparsity*100:.2f}%\n")
        f.write(f"Top 10% mass: {concentration.item()*100:.2f}%\n")


analyze_attention()