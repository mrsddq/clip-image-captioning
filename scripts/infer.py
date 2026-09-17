"""Greedy caption generation from an embedding or an image using a trained checkpoint."""
import argparse
import torch
from data import ByteTokenizer
from scripts.train import build_decoder


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("tokenizer") != "utf8-byte-v1":
        raise ValueError("Expected a version 1 byte-tokenizer checkpoint from scripts.train")
    model = build_decoder(payload["config"], len(ByteTokenizer()))
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload["config"]


def caption_embedding(model, cfg, embedding):
    if embedding.ndim != 1 or embedding.numel() != int(cfg["embed_dim"]) or not torch.isfinite(embedding).all():
        raise ValueError("Embedding must be a finite vector matching embed_dim")
    tokenizer = ByteTokenizer()
    kwargs = {"max_len": int(cfg["training"]["max_len"])} if cfg["decoder"].get("type") == "lstm" else {}
    ids = model.generate(embedding.float().unsqueeze(0), tokenizer.bos_token_id, tokenizer.eos_token_id, "cpu", **kwargs)
    return tokenizer.decode(ids)


def main(checkpoint, image=None, embedding=None):
    model, cfg = load_checkpoint(checkpoint)
    if embedding is not None:
        vector = torch.load(embedding, map_location="cpu", weights_only=True)
    else:
        from PIL import Image
        from models.encoder.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder(model_name=cfg["clip_model"], device="cpu").eval()
        with Image.open(image) as source:
            tensor = encoder.preprocess(source.convert("RGB")).unsqueeze(0)
        vector = encoder(tensor).squeeze(0)
    return caption_embedding(model, cfg, vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--embedding")
    args = parser.parse_args()
    print(main(args.checkpoint, args.image, args.embedding))
