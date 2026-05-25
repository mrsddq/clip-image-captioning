# Reproducibility Plan

## Environment

- Python: 3.10
- Dependencies: pinned in `requirements.txt`
- Config: `configs/clip_cap.yaml`

## Dataset Contract

For real experiments, record dataset version, split, tokenizer version, CLIP checkpoint, decoder checkpoint, and embedding cache checksum.

## Run Order

1. Precompute or validate CLIP embeddings.
2. Train decoder from `configs/clip_cap.yaml`.
3. Evaluate BLEU-1/2/3/4, METEOR, CIDEr, and qualitative examples.
4. Save generated captions and failure cases.
5. Fill `docs/RESULTS_TEMPLATE.md`.

`outputs/metrics/smoke_test_results.csv` is a schema example only. It is not captioning performance.
