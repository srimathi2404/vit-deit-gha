import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class GraphHeadAttention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        k=16,
        alpha=0.0,
        grid_size=8
    ):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.k = k
        self.alpha = alpha
        self.grid_size = grid_size

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # ----------------------------------
        #  Relative Position Bias
        # ----------------------------------
        num_rel_positions = (2 * grid_size - 1) * (2 * grid_size - 1)
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(num_rel_positions))

        self.register_buffer(
            "rel_pos_index",
            self._create_relative_position_index(grid_size),
            persistent=False
        )

        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)


    def _create_relative_position_index(self, grid_size):
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing='ij'
        ))  # (2, H, W)

        coords_flat = coords.reshape(2, -1)  # (2, N)

        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords = rel_coords.permute(1, 2, 0)  # (N, N, 2)

        rel_coords[:, :, 0] += grid_size - 1
        rel_coords[:, :, 1] += grid_size - 1

        rel_coords[:, :, 0] *= (2 * grid_size - 1)

        rel_pos_index = rel_coords.sum(-1)  # (N, N)

        return rel_pos_index


    def forward(self, x, return_attn=False):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # ----------------------------------
        #  Relative Position Bias
        # ----------------------------------
        B, H, N, _ = dots.shape
        num_patches = self.grid_size * self.grid_size

        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.view(num_patches, num_patches)

        full_bias = torch.zeros(N, N, device=x.device)
        full_bias[1:, 1:] = bias  # skip CLS

        dots = dots + full_bias.unsqueeze(0).unsqueeze(0)

        # ----------------------------------
        #  Top-k sparse attention
        # ----------------------------------
        topk_vals, topk_idx = torch.topk(dots, self.k, dim=-1)

        mask = torch.zeros_like(dots)
        mask.scatter_(-1, topk_idx, 1.0)

        sparse_dots = dots.masked_fill(mask == 0, -torch.finfo(dots.dtype).max)
        sparse_attn = self.attend(sparse_dots)

        # ----------------------------------
        # Global attention
        # ----------------------------------
        global_attn = self.attend(dots)

        # ----------------------------------
        # Combine
        # ----------------------------------
        attn = sparse_attn

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attn:
            return out, attn

        return out
    
class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                GraphHeadAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))


    def forward(self, x, return_attn=False):
        attn_maps = []

        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_map = attn(x, return_attn=True)
                attn_maps.append(attn_map)
            else:
                attn_out = attn(x)

            x = attn_out + x
            x = ff(x) + x

        x = self.norm(x)

        if return_attn:
            return x, attn_maps

        return x

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img, return_attn=False):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        if return_attn:
            x, attn_maps = self.transformer(x, return_attn=True)
        else:
            x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        logits = self.mlp_head(x)

        if return_attn:
            return logits, attn_maps

        return logits