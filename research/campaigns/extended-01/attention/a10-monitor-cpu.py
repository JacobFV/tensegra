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
root = Path.home() / "topoformer-campaign01/attention/a10-receipted/results"
out = Path.home() / "topoformer-campaign01/attention/a10-monitor3000-cpu"
out.mkdir(exist_ok=False)
checkpoint = root / "checkpoint.pt"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "8a1b65b1abae768971ef96bb8562c454c7b597a4e51182ce6585e0e60e9d58ef"
cfg = json.loads((root / "config.json").read_text())
cfg.update(conditions=cfg["curve_conditions"], eval_seed=cfg["curve_eval_seed"], eval_examples=cfg["curve_examples"], device="cpu")
torch.manual_seed(cfg["seed"])
model = RecordAttention(width=cfg["width"])
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
start = time.monotonic()
evaluate(model, cfg, out, 3000, "cpu")
receipt = dict(cpu_inference_seconds=time.monotonic()-start, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), device="cpu", note="Additional development inference; CPU numerical differences possible; not reconstructed historical monitoring.")
(out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
(out / "config.json").write_text(json.dumps(cfg, indent=2)+"\n")
print(json.dumps(receipt), flush=True)
