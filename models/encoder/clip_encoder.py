"""Frozen CLIP visual encoder wrapper."""
import torch
import torch.nn as nn
import clip


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        return self.model.encode_image(images).float()
