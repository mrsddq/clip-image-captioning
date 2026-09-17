"""Report token-weighted NLL and generated captions; not BLEU/CIDEr."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import COCOCaptionDataset
from scripts.infer import caption_embedding, load_checkpoint
from scripts.train import run_epoch


def main(checkpoint, annotations, embeddings, output):
    model, cfg = load_checkpoint(checkpoint)
    dataset = COCOCaptionDataset(annotations, embeddings, max_len=int(cfg["training"]["max_len"]), embed_dim=int(cfg["embed_dim"]))
    nll = run_epoch(model, DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"])), "cpu")
    predictions = []
    seen = set()
    for index, row in enumerate(dataset.samples):
        if row["image_id"] not in seen:
            predictions.append({"image_id": row["image_id"], "caption": caption_embedding(model, cfg, dataset[index]["clip_embed"])})
            seen.add(row["image_id"])
    report = {"token_nll": nll, "byte_perplexity": math.exp(nll), "annotations": len(dataset), "images": len(seen),
              "tokenizer": "utf8-byte-v1", "predictions": predictions}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", default="outputs/metrics/captions.json")
    a = p.parse_args()
    print(json.dumps(main(a.checkpoint, a.annotations, a.embeddings, a.output)))
