"""Train CLIP captioning model.
Usage: python scripts/train.py --config configs/clip_cap.yaml
"""
import argparse, yaml, torch
from pathlib import Path


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from models.encoder.clip_encoder import CLIPEncoder
    from models.decoder.transformer_decoder import CaptionDecoder

    encoder = CLIPEncoder(cfg["clip_model"], device).to(device)
    # decoder = CaptionDecoder(vocab_size=..., embed_dim=cfg["embed_dim"], ...).to(device)

    print("Plug in COCO DataLoader and tokenizer to begin training.")
    # opt = torch.optim.Adam(decoder.parameters(), lr=cfg["training"]["lr"])
    # for epoch in range(cfg["training"]["epochs"]):
    #     for images, captions in train_loader:
    #         embeds = encoder(images.to(device))
    #         logits = decoder(embeds, token_ids)
    #         loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    #         ...


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/clip_cap.yaml")
    main(p.parse_args().config)
