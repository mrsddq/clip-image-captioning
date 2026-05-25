from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import COCOCaptionDataset
from models.decoder.lstm_decoder import LSTMCaptionDecoder
from models.decoder.transformer_decoder import CaptionDecoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_decoder(cfg: dict, vocab_size: int):
    decoder_cfg = cfg["decoder"]
    if decoder_cfg.get("type", "transformer") == "lstm":
        return LSTMCaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(decoder_cfg.get("hidden_dim", 512)),
            layers=int(decoder_cfg.get("layers", 2)),
            dropout=float(decoder_cfg.get("dropout", 0.2)),
        )
    return CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=int(cfg["embed_dim"]),
        proj_dim=int(cfg["proj_dim"]),
        num_layers=int(decoder_cfg["layers"]),
        num_heads=int(decoder_cfg["heads"]),
        ff_dim=int(decoder_cfg["ff_dim"]),
        dropout=float(decoder_cfg["dropout"]),
        max_len=int(cfg["training"]["max_len"]),
    )


def main(config: str) -> Path:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise ImportError("transformers is required for tokenization.") from exc
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = COCOCaptionDataset(
        cfg["data"]["train_ann"],
        Path(cfg["data"]["embedding_cache"]) / "train",
        tokenizer,
        max_len=int(cfg["training"]["max_len"]),
    )
    loader = DataLoader(train_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_decoder(cfg, len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for _epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        for batch in loader:
            clip_embed = batch["clip_embed"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(clip_embed, input_ids[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    output_path = checkpoint_dir / "caption_decoder.pt"
    torch.save(model.state_dict(), output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP caption decoder.")
    parser.add_argument("--config", default="configs/clip_cap.yaml")
    args = parser.parse_args()
    print(main(args.config))
