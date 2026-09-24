"""Coordinator-released C04 phase subprocess with external occupancy accounting."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True); parser.add_argument('--python', required=True)
    parser.add_argument('--cap', type=float, required=True)
    parser.add_argument('--phase', choices=('hybrid','n1_static','n1_roles','n2_rekey','timing'), required=True)
    args = parser.parse_args()
    if args.cap <= 0: raise ValueError('positive process cap required')
    config = Path(args.config)
    if args.cap > json.loads(config.read_text())['requested_phase_caps_seconds'][args.phase]:
        raise ValueError('cap exceeds registered request')
    receipt = Path(args.prefix + '.occupancy.json'); log = Path(args.prefix + '.log')
    if receipt.exists() or log.exists() or Path(args.output).exists():
        raise RuntimeError('immutable launch output already exists')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; coordinator must release an idle slot')
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONPATH': 'src'}
    module = 'topoformer.campaign_composition_confirm_' + ('hybrid' if args.phase == 'hybrid' else 'timing' if args.phase == 'timing' else 'neural')
    command = [args.python, '-m', module, '--config', str(config), '--output', args.output, '--device', 'cuda']
    if args.phase not in ('hybrid','timing'): command += ['--arm', args.phase]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(); tick = time.monotonic(); timed_out = False
    with log.open('x') as stream:
        try:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=args.cap).returncode
        except subprocess.TimeoutExpired:
            code = 124; timed_out = True
    record = dict(started_utc=started, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  process_occupancy_seconds=time.monotonic()-tick, cap_seconds=args.cap, exit_code=code, timed_out=timed_out,
                  command=command, phase=args.phase, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    raise SystemExit(code)


if __name__ == '__main__': main()
