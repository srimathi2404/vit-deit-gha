GHA Segmentation experiments

Files:
- `vit.py` - ViT implementation with a segmentation head (per-patch logits).
- `train_gha_seg.py` - Training script using Pascal VOC 2012, patch-wise segmentation labels.

Quick start:

1. Install dependencies (use your project environment). Ensure `torch`, `torchvision`, `einops`, `matplotlib` are available.
2. Run training:

```bash
python gha_seg/train_gha_seg.py
```

Notes:
- The script downloads VOC2012 to `./data/VOCdevkit`.
- It resizes images to 224x224 and uses patch size 16 (14x14 patches).
- Outputs and checkpoints are saved to `gha_seg/outputs`.
