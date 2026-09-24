#!/usr/bin/env python3
"""Run one explicitly released frozen job, with durable process accounting."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

JOBS = {
    "a14-profile": ("67dc519", "campaign-a14-profile.json", "040c663b647a6f281f1e607c87554289c925c1237c4c57ba33cb45826dc2c1d2", "campaign_attention_confirmation_study", 60),
    "a14-1401": ("67dc519", "campaign-a14-confirm-1401.json", "432037e227e9bb7447150503450e535df82e5f31e190d4555cd5e4fc605112fd", "campaign_attention_confirmation_study", 480),
    "a14-1402": ("67dc519", "campaign-a14-confirm-1402.json", "b0acfcab1028a14836cfd02aeba79fb226a1e96315a3775fddf2623e2e3802b7", "campaign_attention_confirmation_study", 480),
    "a14-1403": ("67dc519", "campaign-a14-confirm-1403.json", "c666b0c599ce639fe4d18ab4dd28875b35df50bbf08ca5c1a232f0359c7cbe45", "campaign_attention_confirmation_study", 480),
    "a14-references": ("67dc519", "campaign-a14-engineering-references.json", "e41acf2fb2995e82600d7bd54e83952501c25a0985c9618383af7d2ea76b2919", "campaign_attention_confirmation_study", 120),
    "a13-profile": ("53bdfed", "campaign-a13-profile.json", "e04a027cb2d8d9118285f236da62763022fc9189a0ed96df546707f4efe4eb99", "campaign_attention_shared_address_study", 60),
    "a13-main": ("53bdfed", "campaign-a13-shared-address.json", "ec442effb25da9871da8e3932f3c079d40e974f94b66f9cd6feaf99a05be6f44", "campaign_attention_shared_address_study", 180),
    "a12-profile": ("71f1971", "campaign-a12-profile.json", "ad409bea603954ec8603487af0940a48369a6bd261856bbc3ec5303f7d296e83", "campaign_attention_record_diagnostic_study", 60),
    "a12-main": ("71f1971", "campaign-a12-read-localization.json", "b62eae8650a9f8d1e813edf3c09ad2c73d96ef7bd30b512f93d60334a8aa0518", "campaign_attention_record_diagnostic_study", 180),
    "a11": ("b02f3b6", "campaign-a11-record-continuation.json", "696319d7298d04132fc2f7ac30d4490dcfc501efd1ffbcedec7e6d1269cd0b99", "campaign_attention_records_study", 180),
    "a10": ("afc6596", "campaign-a10-record-continuation.json", "bc46451cc7c4e34b0314d176d20663d733f5ec74e813b1cce18e967c2233e6cb", "campaign_attention_records_study", 120),
    "a09": ("0880c64", "campaign-a09-record-extension.json", "89c486b467c178ed354d443d1b0023521f855b649ca45abfcfa6727571462f66", "campaign_attention_records_study", 180),
    "a08": ("e789147", "campaign-a08-frozen.json", "04b61a0acd05ece7cab53eb97759b4e9203a6f1011b3ff05b97d76b694a9e2e6", "campaign_attention_corruption_study", 300),
}


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=JOBS)
    parser.add_argument("--release", required=True, help="Coordinator's explicit per-job release identifier")
    args = parser.parse_args()
    source, config_name, config_sha, module, cap = JOBS[args.job]
    stage = Path(f"/tmp/campaign-{args.job}-source-{source}")
    config = stage / "configs" / config_name
    assert hashlib.sha256(config.read_bytes()).hexdigest() == config_sha
    lock = open("/tmp/topoformer-attention-gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    idle = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True)
    if idle.strip():
        raise SystemExit(f"GPU not idle; refusing launch: {idle}")
    root = Path.home() / "topoformer-campaign01" / "attention" / f"{args.job}-receipted"
    root.mkdir(parents=True, exist_ok=False)
    command = [str(Path.home() / "topoformer-stage8-cuda/bin/python"), "-m", f"topoformer.{module}", "--config", str(config), "--output", str(root / "results")]
    receipt = dict(job=args.job, source=source, config_sha256=config_sha, cap_seconds=cap,
                   release=args.release, hostname=socket.gethostname(), command=command,
                   launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   started_utc=utc(), status="started", gpu_idle_query=idle)
    atomic_json(root / "receipt.json", receipt)
    environment = dict(os.environ, PYTHONPATH=str(stage / "src"), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    started = time.monotonic()
    process = None
    try:
        with (root / "process.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=stage, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result = process.wait(timeout=cap)
                receipt["status"] = "completed" if result == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result = 124
                receipt["status"] = "timeout"
            log.flush()
            os.fsync(log.fileno())
        receipt["exit_code"] = result
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        receipt.update(status="launcher_error", error=repr(error), exit_code=1)
        raise
    finally:
        receipt.update(ended_utc=utc(), full_process_occupancy_seconds=time.monotonic() - started)
        atomic_json(root / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
