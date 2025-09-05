# CLIP-Based Image Captioning

Image captioning system using OpenAI CLIP as a frozen visual encoder, with a Transformer decoder trained to generate natural language captions. Trained on COCO Captions.

## Results

| Metric | Value | Split |
|---|---|---|
| BLEU-4 | 0.32 | COCO val2017 |
| BLEU-1 | _add_ | COCO val2017 |
| METEOR | _add_ | COCO val2017 |
| CIDEr | _add_ | COCO val2017 |

## Architecture

```
Image (224×224×3)
  └─ CLIP ViT-B/32 encoder  [frozen]
       └─ 512-dim embedding
            └─ Linear projection  →  768-dim
                 └─ Transformer decoder (6 layers, 8 heads)
                      └─ Linear + Softmax
                           └─ Generated caption
```

The CLIP encoder is kept frozen throughout training. Only the projection layer and Transformer decoder are trained.

## Quickstart

```bash
git clone https://github.com/your-username/clip-image-captioning
cd clip-image-captioning
pip install -r requirements.txt
```

## Data

Download COCO Captions:

```bash
# Images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

Place as:
```
data/
  annotations/
    captions_train2017.json
    captions_val2017.json
  embeddings/      ← pre-computed CLIP embeddings cached here
```

Pre-computing embeddings is recommended — it removes the encoder from the training loop and speeds up iteration significantly.

## Training

```bash
python scripts/train.py --config configs/clip_cap.yaml
```

Key config in `configs/clip_cap.yaml`:

```yaml
clip_model: ViT-B/32
embed_dim: 512
proj_dim: 768
decoder_layers: 6
decoder_heads: 8
epochs: 30
batch_size: 64
lr: 3e-4
max_len: 40
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/logs/best_model.pt
```

Computes BLEU-1 through BLEU-4 on COCO val2017 using `pycocoevalcap`.

## Inference

```bash
python scripts/infer.py --checkpoint outputs/logs/best_model.pt --image path/to/image.jpg
```

## Sample Outputs

| File | Contents |
|---|---|
| `assets/01_architecture.png` | Full pipeline diagram |
| `assets/02_caption_examples.png` | 5-row table: image, reference, predicted caption |
| `assets/03_bleu_table.png` | BLEU-1 through BLEU-4 with evaluation split |
| `assets/04_training_loss.png` | Cross-entropy loss vs epoch |
| `assets/05_embedding_viz.png` | t-SNE of CLIP embeddings (optional) |

## Limitations

- CLIP encoder is frozen — image representations not fine-tuned for captioning domain
- BLEU-4 of 0.32 is competitive for a lightweight decoder; larger decoders improve further
- No beam search by default — greedy decoding used for speed

## Environment

```
Python 3.10
torch==2.1.0
clip @ git+https://github.com/openai/CLIP.git
transformers==4.35.0
pycocotools==2.0.7
pycocoevalcap==1.2
numpy==1.26.0
```
