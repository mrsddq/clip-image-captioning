# Architecture Rationale

CLIP provides strong image representations learned from image-text pairs. A Transformer decoder can learn the language-generation layer while optionally keeping the visual encoder frozen for faster experiments.

Design choices:

- frozen CLIP encoder by default for reproducibility and lower compute
- decoder depth/head count configured in YAML
- embedding cache path included to avoid repeated encoder passes

Upgrade path:

- compare frozen vs fine-tuned CLIP
- add beam search and nucleus sampling
- add Hugging Face tokenizer/model integration
- add Gradio demo for image upload and caption output
