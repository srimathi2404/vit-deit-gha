
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphHeadAttention(nn.Module):
    """
    Graph Head Attention (GHA)
    Drop-in replacement for DeiT MultiHeadAttention
    Input shape: (B, N, C)
    """

    def __init__(self, dim, num_heads=3, topk=8):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk = topk

        self.scale = self.head_dim ** -0.5

        # Q K V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # feature transform for GAT
        self.Wc =  nn.Linear(self.head_dim, self.head_dim, bias=False)

        # GAT attention parameters
        self.a = nn.Parameter(
            torch.empty(1, num_heads, 2 * self.head_dim, 1)
        )
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(0.2)

        # output projections
        self.Wgat = nn.Linear(dim, dim, bias=False)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x,attn_mask=None, **kwargs):

        B, N, C = x.shape

        # -------------------------
        # Q K V projections
        # -------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)

        # feature transform for GAT
        v_feat = self.Wc(v)
        v_feat = v_feat.reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)

        # -------------------------
        # initial attention matrix
        # -------------------------

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # -------------------------
        # Top-k graph construction
        # -------------------------

        topk_vals, topk_idx = torch.topk(attn, self.topk, dim=-1)

        adj = torch.zeros_like(attn)
        adj.scatter_(-1, topk_idx, topk_vals)

        # -------------------------
        # Add self edges
        # -------------------------

        I = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0)
        adj = adj + I

        # -------------------------
        # Triangularization (paper method)
        # -------------------------

        upper = torch.triu(adj)
        lower = torch.tril(adj)

        upper_sym = upper + upper.transpose(-2, -1)
        lower_sym = lower + lower.transpose(-2, -1)

        adj_final = (upper_sym > 0) | (lower_sym > 0)
        adj_final = adj_final.float()

        # -------------------------
        # Graph Attention Network
        # -------------------------

        a1 = self.a[:, :, :self.head_dim, :]
        a2 = self.a[:, :, self.head_dim:, :]

        score_i = torch.matmul(v_feat, a1)
        score_j = torch.matmul(v_feat, a2)

        e = score_i + score_j.transpose(-1, -2)
        e = self.leaky_relu(e)

        # mask non edges
        mask = adj_final > 0
        e = e.masked_fill(~mask, -1e9)

        attn_graph = F.softmax(e, dim=-1)

        # -------------------------
        # message aggregation
        # -------------------------

        out = attn_graph @ v

        # concatenate heads
        out = out.permute(0,2,1,3).reshape(B,N,C)

        # projection
        out = self.Wgat(out)
        out = self.relu(out)

        return self.proj(out)