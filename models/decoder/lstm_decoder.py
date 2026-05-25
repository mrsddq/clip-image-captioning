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
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, clip_embeds: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        batch = clip_embeds.shape[0]
        h0 = self.init_h(clip_embeds).view(self.layers, batch, self.hidden_dim).contiguous()
        c0 = self.init_c(clip_embeds).view(self.layers, batch, self.hidden_dim).contiguous()
        output, _ = self.lstm(self.token_emb(token_ids), (h0, c0))
        return self.head(output)
