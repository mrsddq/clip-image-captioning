import json
import pytest
import torch
import yaml
from data import ByteTokenizer, COCOCaptionDataset
from models.decoder.transformer_decoder import CaptionDecoder
from models.decoder.lstm_decoder import LSTMCaptionDecoder
from scripts.train import main
from scripts.evaluate import main as evaluate
from scripts.infer import load_checkpoint


def test_causal_mask_prevents_future_caption_leakage():
    torch.manual_seed(3)
    model = CaptionDecoder(20, 4, 8, 1, 2, 16, 0.0, 8).eval()
    image = torch.randn(1, 4)
    a, b = torch.tensor([[1, 4, 5, 6]]), torch.tensor([[1, 4, 7, 8]])
    torch.testing.assert_close(model(image, a)[:, :2], model(image, b)[:, :2])


def test_lstm_batch_does_not_mix_image_states():
    torch.manual_seed(4)
    model = LSTMCaptionDecoder(20, 4, 8, 2, 0.0).eval()
    images, ids = torch.randn(2, 4), torch.tensor([[1, 4], [1, 5]])
    together = model(images, ids)
    torch.testing.assert_close(together[0], model(images[:1], ids[:1])[0])


def test_byte_tokenizer_roundtrip_and_padding_labels(tmp_path):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode("café", 20)) == "café"
    (tmp_path / "a.json").write_text(json.dumps({"annotations": [{"image_id": 1, "caption": "hi"}]}))
    torch.save(torch.zeros(4), tmp_path / "1.pt")
    item = COCOCaptionDataset(tmp_path / "a.json", tmp_path, max_len=8, embed_dim=4)[0]
    assert item["labels"].tolist() == [1, 107, 108, 2, -100, -100, -100, -100]
    torch.save(torch.full((4,), float("nan")), tmp_path / "1.pt")
    with pytest.raises(ValueError, match="finite"):
        COCOCaptionDataset(tmp_path / "a.json", tmp_path)[0]


@pytest.mark.parametrize("decoder_type", ["transformer", "lstm"])
def test_offline_train_checkpoint_evaluate(tmp_path, decoder_type):
    torch.set_num_threads(1)
    for split, image_id in [("train", 1), ("val", 2)]:
        (tmp_path / split).mkdir()
        torch.save(torch.ones(4), tmp_path / split / f"{image_id}.pt")
        (tmp_path / f"{split}.json").write_text(json.dumps({"annotations": [{"image_id": image_id, "caption": "hi"}]}))
    cfg = {"seed": 42, "clip_model": "ViT-B/32", "embed_dim": 4, "proj_dim": 8,
           "decoder": {"type": decoder_type, "layers": 1, "heads": 2, "ff_dim": 16, "hidden_dim": 8, "dropout": 0.0},
           "training": {"epochs": 1, "batch_size": 1, "lr": 0.001, "max_len": 8},
           "data": {"train_ann": str(tmp_path / "train.json"), "val_ann": str(tmp_path / "val.json"), "embedding_cache": str(tmp_path)},
           "logging": {"checkpoint_dir": str(tmp_path / "checkpoints")}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = main(str(config))
    model, saved_cfg = load_checkpoint(checkpoint)
    assert saved_cfg == cfg
    report = evaluate(checkpoint, tmp_path / "val.json", tmp_path / "val", tmp_path / "report.json")
    assert report["images"] == report["annotations"] == 1
    assert report["token_nll"] > 0
    assert isinstance(report["predictions"][0]["caption"], str)
    cfg["data"]["val_ann"] = cfg["data"]["train_ann"]
    torch.save(torch.ones(4), tmp_path / "val" / "1.pt")
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="overlap"):
        main(str(config))
