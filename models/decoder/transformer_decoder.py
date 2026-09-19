"""Transformer decoder for caption generation."""
import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, proj_dim=768,
                 num_layers=6, num_heads=8, ff_dim=2048, dropout=0.1, max_len=40):
        super().__init__()
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.token_emb = nn.Embedding(vocab_size, proj_dim)
        self.pos_emb = nn.Embedding(max_len + 1, proj_dim)
        layer = nn.TransformerDecoderLayer(proj_dim, num_heads, ff_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.head = nn.Linear(proj_dim, vocab_size)
        self.max_len = max_len

    def forward(self, clip_embeds, token_ids, tgt_mask=None, tgt_key_padding_mask=None):
        if tgt_mask is None:
            tgt_mask = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        memory = self.proj(clip_embeds).unsqueeze(1)
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.decoder(x, memory, tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask))

    @torch.no_grad()
    def generate(self, clip_embeds, bos_id, eos_id, device):
        if clip_embeds.shape[0] != 1:
            raise ValueError("Greedy generation expects one image embedding")
        tokens = torch.tensor([[bos_id]], device=device)
        for _ in range(self.max_len - 1):
            logits = self(clip_embeds, tokens)[:, -1]
            logits[:, 0] = -torch.inf  # PAD is never a caption token
            logits[:, bos_id] = -torch.inf
            next_tok = logits.argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break
        return tokens[0].tolist()
