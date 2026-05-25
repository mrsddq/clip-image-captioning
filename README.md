# CLIP-Based Image Captioning

[![CI](https://github.com/mrsddq/clip-image-captioning/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/clip-image-captioning/actions/workflows/ci.yml)

Portfolio-ready image captioning project using CLIP-style visual embeddings and a Transformer decoder.

The repository provides model structure, configs, and train/evaluate/infer entry points. It does not commit COCO data, model weights, or unverified captioning metrics.

## Highlights

- Frozen CLIP visual encoder pattern
- Transformer decoder architecture
- Config-driven training workflow
- COCO Captions data layout
- Results template for BLEU/METEOR/CIDEr reporting

## Structure

```text
configs/
  clip_cap.yaml
docs/
  ABLATION_PLAN.md
  ARCHITECTURE_RATIONALE.md
  DEPLOYMENT_NOTES.md
  REPRODUCIBILITY.md
  RESULTS_TEMPLATE.md
models/
  encoder/clip_encoder.py
  decoder/transformer_decoder.py
scripts/
  train.py
  evaluate.py
  infer.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Layout

```text
data/
  annotations/
    captions_train2017.json
    captions_val2017.json
  images/
    train2017/
    val2017/
  embeddings/
```

Precomputing CLIP embeddings is recommended to speed up decoder experiments.

## Train

```bash
python -m scripts.train --config configs/clip_cap.yaml
```

## Evaluate

```bash
python -m scripts.evaluate --checkpoint outputs/logs/best_model.pt
```

## Inference

```bash
python -m scripts.infer --checkpoint outputs/logs/best_model.pt --image path/to/image.jpg
```

## Results

No verified public metrics are committed yet. Record real BLEU, METEOR, CIDEr, and qualitative examples in [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md).

Research support docs:

- [Reproducibility Plan](docs/REPRODUCIBILITY.md)
- [Architecture Rationale](docs/ARCHITECTURE_RATIONALE.md)
- [Ablation Plan](docs/ABLATION_PLAN.md)
- [Deployment Notes](docs/DEPLOYMENT_NOTES.md)

`outputs/metrics/smoke_test_results.csv` is a schema artifact only, not a benchmark.

Recommended artifacts:

- `assets/architecture.png`
- `assets/caption-examples.png`
- `assets/metrics-table.png`
- `assets/failure-case.png`

## Limitations

- Dataset and weights are not included.
- Frozen CLIP embeddings may limit domain adaptation.
- Beam search and advanced decoding are extension points.
