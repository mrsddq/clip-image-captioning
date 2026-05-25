from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from models.encoder.clip_encoder import CLIPEncoder


def main(images_dir: Path, annotation_file: Path, output_dir: Path, clip_model: str) -> None:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    image_by_id = {img["id"]: img["file_name"] for img in payload["images"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CLIPEncoder(model_name=clip_model, device=str(device)).to(device).eval()
    for image_id, file_name in tqdm(image_by_id.items()):
        output_path = output_dir / f"{image_id}.pt"
        if output_path.exists():
            continue
        image = encoder.preprocess(Image.open(images_dir / file_name).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = encoder(image).squeeze(0).cpu()
        torch.save(embedding, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CLIP image embeddings for COCO captions.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default="ViT-B/32")
    args = parser.parse_args()
    main(args.images_dir, args.annotation_file, args.output_dir, args.clip_model)
