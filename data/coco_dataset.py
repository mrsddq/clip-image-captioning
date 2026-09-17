"""COCO captions paired with cached CLIP embeddings; no downloads on import."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """Deterministic UTF-8 baseline: PAD=0, BOS=1, EOS=2, bytes=3..258."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    def __len__(self):
        return 259
    def encode(self, text, max_len):
        if max_len < 3:
            raise ValueError("max_len must be at least 3")
        return [1] + [b + 3 for b in text.encode("utf-8")[:max_len - 2]] + [2]
    def decode(self, ids):
        values = []
        for token in ids:
            if token == 2:
                break
            if token >= 3:
                values.append(token - 3)
        return bytes(values).decode("utf-8", errors="replace")


class COCOCaptionDataset(Dataset):
    def __init__(self, annotation_file, embedding_dir, tokenizer=None, max_len=128, embed_dim=None):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.max_len, self.embed_dim = max_len, embed_dim
        self.embedding_dir = Path(embedding_dir)
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        self.samples = payload.get("annotations", [])
        if not self.samples:
            raise ValueError("COCO annotations must not be empty")
        for row in self.samples:
            if not isinstance(row.get("caption"), str) or not row["caption"].strip():
                raise ValueError("Every annotation needs a nonempty caption")
            if not isinstance(row.get("image_id"), int) or row["image_id"] < 0:
                raise ValueError("image_id must be a nonnegative integer")
            if not (self.embedding_dir / f"{row['image_id']}.pt").is_file():
                raise FileNotFoundError(f"Missing cached embedding for image {row['image_id']}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        row = self.samples[index]
        embedding = torch.load(self.embedding_dir / f"{row['image_id']}.pt", map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 1 or not torch.isfinite(embedding).all():
            raise ValueError("Cached embeddings must be finite one-dimensional tensors")
        if self.embed_dim is not None and embedding.numel() != self.embed_dim:
            raise ValueError("Cached embedding dimension does not match config")
        ids = self.tokenizer.encode(row["caption"], self.max_len)
        inputs = torch.tensor(ids + [0] * (self.max_len - len(ids)))
        labels = inputs.clone()
        labels[len(ids):] = -100
        return {"clip_embed": embedding.float(), "input_ids": inputs, "labels": labels}
