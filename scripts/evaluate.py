"""Evaluate BLEU/METEOR/CIDEr on COCO val split.
Usage: python scripts/evaluate.py --checkpoint outputs/logs/best_model.pt
"""
import argparse, json, torch


def main(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {ckpt}")
    print("Plug in COCO val DataLoader and tokenizer.")
    # Generate captions for val set, then:
    # from pycocoevalcap.eval import COCOEvalCap
    # coco_eval = COCOEvalCap(coco, results)
    # coco_eval.evaluate()
    # for k, v in coco_eval.eval.items():
    #     print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    main(p.parse_args().checkpoint)
