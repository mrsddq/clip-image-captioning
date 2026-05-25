# Model Card: CLIP Image Captioning

## Dataset
Target dataset: COCO Captions 2017. CLIP image embeddings are cached before training to reduce repeated image loading.

## Model
Image encoder: frozen CLIP. Baseline decoder: 2-layer LSTM. Main decoder: Transformer decoder with optional beam search.

## Evaluation
Primary metrics: BLEU-4 and CIDEr. Secondary metrics: METEOR, ROUGE-L, and SPICE when `pycocoevalcap` is available.

## Limitations
Captioning models can hallucinate details. Generated text should be evaluated against image evidence and failure cases.

## Ethical Considerations
Captioning systems may encode dataset biases around people, activities, and locations. Report qualitative failures.
