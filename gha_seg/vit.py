import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


# -----------------------
# Helper
# -----------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# -----------------------
# Graph Attention (unchanged core)
# -----------------------
class GraphHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., k=16, alpha=0.2):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.k = k

        self.alpha = nn.Parameter(torch.tensor(alpha))

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Top-k sparse
        topk_vals, topk_idx = torch.topk(dots, self.k, dim=-1)
        sparse = torch.full_like(dots, -1e9)
        sparse.scatter_(-1, topk_idx, topk_vals)
        sparse_attn = self.attend(sparse)

        # Global
        global_attn = self.attend(dots)

        alpha = torch.clamp(self.alpha, 0, 1)
        attn = (1 - alpha) * sparse_attn + alpha * global_attn

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# -----------------------
# Transformer Encoder
# -----------------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                GraphHeadAttention(dim, heads, dim_head),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim)
                )
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.norm(x)


# -----------------------
# Decoder (VERY IMPORTANT)
# -----------------------
class SegmentationDecoder(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(dim // 2, dim // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(dim // 4, num_classes, 1)

    def forward(self, x):
        # x: (B, D, H, W)

        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)

        return x


# -----------------------
# Full Model
# -----------------------
class ViTSeg(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=256,
        depth=6,
        heads=4,
        mlp_dim=512,
        num_classes=21,
        channels=3
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0

        self.h = image_height // patch_height
        self.w = image_width // patch_width
        self.num_patches = self.h * self.w

        patch_dim = channels * patch_height * patch_width

        # Patch embedding
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head=64, mlp_dim=mlp_dim)

        # Decoder
        self.decoder = SegmentationDecoder(dim, num_classes)

    def forward(self, img):
        B = img.shape[0]

        x = self.to_patch(img)
        x = x + self.pos_embedding

        x = self.transformer(x)

        # Tokens → Feature map
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.h, w=self.w)

        # Decode
        x = self.decoder(x)

        # Final upsample to original size
        x = F.interpolate(x, size=img.shape[-2:], mode='bilinear', align_corners=False)

        return x