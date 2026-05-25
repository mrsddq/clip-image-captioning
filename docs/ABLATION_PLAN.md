# Ablation Plan

| Experiment | Variable | Fixed Controls | Metric |
|---|---|---|---|
| baseline | frozen CLIP + decoder | dataset split | CIDEr, BLEU-4 |
| encoder tuning | frozen vs fine-tuned | decoder, split | CIDEr |
| decoder size | layers/heads | encoder, split | CIDEr vs latency |
| decoding | greedy vs beam search | checkpoint | BLEU/CIDEr and qualitative fluency |
| max length | caption length cap | checkpoint | truncation and repetition rate |

Each ablation should include config, command, seed, metrics CSV, and generated caption examples.
