"""Caption a single image.
Usage: python scripts/infer.py --checkpoint best.pt --image photo.jpg
"""
import argparse, torch
from PIL import Image


def main(ckpt, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.encoder.clip_encoder import CLIPEncoder
    encoder = CLIPEncoder(device=device).to(device)
    print(f"Image: {img_path}")
    print("Load tokenizer, run encoder, pass to decoder.generate(), decode tokens.")
    # img = encoder.preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    # embed = encoder(img)
    # tokens = decoder.generate(embed, bos_id, eos_id, device)
    # print("Caption:", tokenizer.decode(tokens))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    a = p.parse_args()
    main(a.checkpoint, a.image)
