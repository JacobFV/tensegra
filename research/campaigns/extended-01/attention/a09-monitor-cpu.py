"""Additional development inference on the registered 173M monitoring population."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from topoformer.campaign_attention_records import RecordAttention
from topoformer.campaign_attention_selector_study import evaluate

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(2)
root = Path.home() / "topoformer-campaign01/attention/a09-receipted/results"
out = Path.home() / "topoformer-campaign01/attention/a09-monitor2000-cpu"
out.mkdir(exist_ok=False)
checkpoint = root / "checkpoint.pt"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "e8f95b43e897fc5189cbc5ea7db331a5eb98a48ac06cac4567a9eacaafd1c024"
cfg = json.loads((root / "config.json").read_text())
cfg.update(conditions=cfg["curve_conditions"], eval_seed=cfg["curve_eval_seed"], eval_examples=cfg["curve_examples"], device="cpu")
torch.manual_seed(cfg["seed"])
model = RecordAttention(width=cfg["width"])
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
start = time.monotonic()
evaluate(model, cfg, out, 2000, "cpu")
receipt = dict(cpu_inference_seconds=time.monotonic()-start, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), device="cpu", note="Additional development inference; CPU numerical differences possible; not reconstructed historical monitoring.")
(out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
(out / "config.json").write_text(json.dumps(cfg, indent=2)+"\n")
print(json.dumps(receipt), flush=True)
