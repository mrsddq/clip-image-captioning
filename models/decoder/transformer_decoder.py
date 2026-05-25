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
        memory = self.proj(clip_embeds).unsqueeze(1)
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.decoder(x, memory, tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask))

    @torch.no_grad()
    def generate(self, clip_embeds, bos_id, eos_id, device):
        memory = self.proj(clip_embeds).unsqueeze(1)
        tokens = torch.tensor([[bos_id]], device=device)
        for _ in range(self.max_len):
            pos = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
            x = self.token_emb(tokens) + self.pos_emb(pos)
            next_tok = self.head(self.decoder(x, memory))[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break
        return tokens[0].tolist()

    @torch.no_grad()
    def beam_search(self, clip_embed, bos_id, eos_id, device, num_beams=3):
        memory = self.proj(clip_embed.unsqueeze(0)).unsqueeze(1)
        beams = [(torch.tensor([[bos_id]], device=device), 0.0)]
        for _ in range(self.max_len):
            candidates = []
            for tokens, score in beams:
                if tokens[0, -1].item() == eos_id:
                    candidates.append((tokens, score))
                    continue
                pos = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
                x = self.token_emb(tokens) + self.pos_emb(pos)
                logits = self.head(self.decoder(x, memory))[:, -1]
                log_probs = torch.log_softmax(logits, dim=-1)
                values, indices = torch.topk(log_probs, k=num_beams, dim=-1)
                for value, index in zip(values[0], indices[0]):
                    next_tokens = torch.cat([tokens, index.view(1, 1)], dim=1)
                    candidates.append((next_tokens, score + float(value)))
            beams = sorted(candidates, key=lambda item: item[1] / item[0].shape[1], reverse=True)[:num_beams]
            if all(tokens[0, -1].item() == eos_id for tokens, _ in beams):
                break
        return beams[0][0][0].tolist()
