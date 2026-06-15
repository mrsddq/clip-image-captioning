# Portfolio Evidence Plan

This project should be shown as an image captioning workflow with CLIP embeddings, decoder training, evaluation, and qualitative caption review. Do not claim captioning quality without a documented run.

## Reproducible Demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_portfolio_contract.py"
python -m scripts.precompute_embeddings --config configs/clip_cap.yaml
python -m scripts.train --config configs/clip_cap.yaml
python -m scripts.evaluate --checkpoint outputs/logs/best_model.pt
python -m scripts.infer --checkpoint outputs/logs/best_model.pt --image path/to/image.jpg
```

## Evidence To Capture

| Artifact | Portfolio Use |
|---|---|
| `assets/caption-examples.png` | Shows image, generated caption, and reference caption. |
| `assets/failure-case.png` | Shows a captioning failure with explanation. |
| `outputs/metrics/caption_eval.json` | Records BLEU, METEOR, ROUGE-L, CIDEr, and dataset split. |
| `docs/RESULTS.md` | Summarizes only verified captioning runs. |

## Demo Narrative

1. Explain why embeddings are precomputed before decoder experiments.
2. Show the decoder config and beam-search settings.
3. Present both metric output and qualitative examples.
4. Discuss failure cases such as counting, spatial relationships, and rare objects.

## Evidence Checklist Before Pinning

- [ ] COCO or shareable sample subset identified.
- [ ] Caption example grid added to `assets/`.
- [ ] Real metric table added to `docs/RESULTS.md`.
- [ ] CI badge green on the latest commit.
- [ ] Model and dataset limitations documented.
