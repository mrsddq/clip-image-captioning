# CLIP image captioning: reproducible decoder baseline

[![CI](https://github.com/mrsddq/clip-image-captioning/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/clip-image-captioning/actions/workflows/ci.yml)

Train a causal Transformer or LSTM caption decoder on frozen CLIP image embeddings.
The data loader, training/validation loop, versioned checkpoint, greedy inference,
and token-weighted evaluation are implemented. CI executes both decoders end to end
on tiny synthetic embeddings, offline and on CPU. This checks software behavior;
it is **not evidence of caption quality on COCO**.

## Design choices

- A deterministic UTF-8 byte tokenizer avoids network/tokenizer dependencies and
  stores its version in checkpoints. It is a simple baseline, not a pretrained
  language model. `max_len` counts bytes including BOS/EOS, not words.
- Transformer training uses a causal attention mask; padding is excluded from
  next-token loss. Validation NLL is weighted by non-padding target token count.
- Checkpoints include architecture/configuration, tokenizer version, epoch and
  validation loss; inference reconstructs the exact architecture.
- Train/validation image-ID overlap, missing embeddings, nonfinite vectors, and
  embedding-size mismatches fail explicitly. Use source-provenance records when
  preparing a cache: dimensionality alone cannot identify its CLIP encoder.
- Best validation checkpoint, gradient clipping, fixed seeds, greedy generation.

## CPU/offline verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
python -m pytest -q
```

Tests generate local synthetic COCO records and tensors; no model downloads.

## Real-data workflow

Obtain licensed COCO image/caption splits yourself. Install the optional encoder
stack with `pip install -r requirements.txt`; CLIP weight retrieval requires
network access on its first use. Cache training and validation embeddings separately:

```bash
python -m scripts.precompute_embeddings --images-dir data/train2017 --annotation-file data/annotations/captions_train2017.json --output-dir data/embeddings/train --clip-model ViT-B/32
python -m scripts.precompute_embeddings --images-dir data/val2017 --annotation-file data/annotations/captions_val2017.json --output-dir data/embeddings/val --clip-model ViT-B/32
python -m scripts.train --config configs/clip_cap.yaml
python -m scripts.infer --checkpoint outputs/logs/caption_decoder.pt --image assets/photo.jpg
python -m scripts.infer --checkpoint outputs/logs/caption_decoder.pt --embedding data/embeddings/val/42.pt
python -m scripts.evaluate --checkpoint outputs/logs/caption_decoder.pt --annotations data/annotations/captions_val2017.json --embeddings data/embeddings/val --output outputs/metrics/captions.json
```

Set `decoder.type: lstm` for the recurrent baseline. Evaluation writes generated
captions, byte perplexity and token NLL. It does **not** label them BLEU, CIDEr or
semantic quality. Reserve a held-out test split for final evaluation; validation
is used for checkpoint selection. Legacy raw state-dict checkpoints are not
compatible with the versioned format and should be retrained or explicitly migrated.

## Results and limitations

No verified public metrics are committed yet. No trained weights or dataset are
included. Full COCO training and CLIP-image inference have not been run in the CPU
synthetic test suite. Captions may be inaccurate; byte-level decoding can produce
invalid UTF-8 (shown as replacement characters). Beam search and pretrained-language
model adaptation are future work, not implemented claims.

[Portfolio Evidence Plan](docs/PORTFOLIO_EVIDENCE.md) · [Model card](docs/MODEL_CARD.md)
