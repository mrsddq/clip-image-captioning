from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.init_h = nn.Linear(embed_dim, hidden_dim * layers)
        self.init_c = nn.Linear(embed_dim, hidden_dim * layers)
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, clip_embeds: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        batch = clip_embeds.shape[0]
        h0 = self.init_h(clip_embeds).view(batch, self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.init_c(clip_embeds).view(batch, self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        output, _ = self.lstm(self.token_emb(token_ids), (h0, c0))
        return self.head(output)

    @torch.no_grad()
    def generate(self, clip_embeds, bos_id, eos_id, device, max_len=128):
        if clip_embeds.shape[0] != 1:
            raise ValueError("Greedy generation expects one image embedding")
        tokens = torch.tensor([[bos_id]], device=device)
        for _ in range(max_len - 1):
            logits = self(clip_embeds, tokens)[:, -1]
            logits[:, 0] = -torch.inf
            logits[:, bos_id] = -torch.inf
            next_token = logits.argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == eos_id:
                break
        return tokens[0].tolist()
