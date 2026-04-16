import torch
import torch.nn as nn
from models.models import deit_tiny_patch16_224
from gha_attention import GraphHeadAttention

def create_gha_model(num_classes=10, topk=8):
    model = deit_tiny_patch16_224(pretrained=False)

    # Replace attention in all transformer blocks
    for i in range(len(model.blocks)):
        model.blocks[i].attn = GraphHeadAttention(
            dim=model.embed_dim,
            num_heads=model.blocks[i].attn.num_heads,
            topk=topk
        )

    # Modify classification head
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model