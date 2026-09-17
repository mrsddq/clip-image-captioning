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


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        labels = batch["labels"][:, 1:].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(batch["clip_embed"].to(device), batch["input_ids"][:, :-1].to(device))
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite caption loss; checkpoint was not updated")
            count = int((labels != -100).sum())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (loss / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_sum += float(loss.detach())
        token_count += count
    if not token_count:
        raise ValueError("No caption tokens available")
    return loss_sum / token_count


def main(config: str) -> Path:
    from data import ByteTokenizer
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    if int(cfg["training"]["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if float(cfg["training"].get("label_smoothing", 0.0)) != 0:
        raise ValueError("Only label_smoothing=0 is supported by token NLL training")
    if int(cfg["decoder"].get("beam_size", 1)) != 1:
        raise ValueError("Only greedy decoding (beam_size=1) is supported")
    tokenizer = ByteTokenizer()
    loaders = {}
    for split in ("train", "val"):
        dataset = COCOCaptionDataset(cfg["data"][f"{split}_ann"], Path(cfg["data"]["embedding_cache"]) / split,
                                     tokenizer, int(cfg["training"]["max_len"]), int(cfg["embed_dim"]))
        loaders[split] = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=split == "train")
    train_ids = {row["image_id"] for row in loaders["train"].dataset.samples}
    if train_ids & {row["image_id"] for row in loaders["val"].dataset.samples}:
        raise ValueError("Train and validation image IDs overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_decoder(cfg, len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    output = Path(cfg["logging"]["checkpoint_dir"]) / "caption_decoder.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(int(cfg["training"]["epochs"])):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        val_loss = run_epoch(model, loaders["val"], device)
        print(f"epoch={epoch + 1} train_token_nll={train_loss:.6f} val_token_nll={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"format_version": 1, "tokenizer": "utf8-byte-v1", "config": cfg,
                        "state_dict": model.state_dict(), "epoch": epoch + 1, "val_token_nll": val_loss}, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a caption decoder on cached CLIP embeddings.")
    parser.add_argument("--config", default="configs/clip_cap.yaml")
    print(main(parser.parse_args().config))
