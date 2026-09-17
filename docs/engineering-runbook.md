# CLIP captioning engineering runbook

This repository trains a Transformer or LSTM caption decoder on cached CLIP image
embeddings. Follow the [README offline setup](../README.md#cpuoffline-verification)
and [real-data workflow](../README.md#real-data-workflow). Optional CLIP image
encoding uses the fuller dependency stack and can retrieve pretrained weights on
first use; importing or testing the decoder path does not require that download.

## Local verification

Run from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Dependency installation needs package-network access. Once installed, the test
suite runs on CPU with generated fixtures and does not download model weights or
datasets. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell
and run `python -m pytest -q`.

Tests run both decoder architectures through training, validation, checkpoint
loading and caption evaluation on generated embeddings. Regression checks cover
causal masking, LSTM sample isolation, padding labels, overlapping image IDs, and
nonfinite losses. They do not assess COCO caption quality.

## Data and artifact contract

- Use separate COCO annotation/cache directories for training and validation;
  each annotation's integer image ID maps to a one-dimensional `<image_id>.pt`
  embedding. Track the CLIP encoder/version and image provenance with the cache.
- The UTF-8 byte tokenizer reserves PAD/BOS/EOS tokens. `max_len` counts bytes,
  including BOS/EOS; truncation may split UTF-8 sequences. This is a deterministic
  baseline, not a pretrained language model.
- Training selects the lowest validation token NLL and writes a versioned
  `caption_decoder.pt` containing weights, architecture/configuration and tokenizer
  identity. Nonfinite losses fail before checkpoint selection.
- Use the README's `--embedding` inference command to stay on the cached-feature
  path. The `--image` path requires CLIP and its matching encoder configuration.
- Evaluation writes generated captions, token NLL and byte perplexity. These are
  not BLEU/CIDEr or semantic-quality measurements. Keep a held-out test split
  separate from checkpoint-selection data.
- Keep datasets, feature tensors, checkpoints and generated reports outside git.
  Publish only licensed examples and results with the exact run provenance.

Full COCO training, caption quality, beam search and pretrained language-model
adaptation are not demonstrated by the synthetic test suite.
