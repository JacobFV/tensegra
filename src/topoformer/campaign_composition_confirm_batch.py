"""Run one explicitly coordinator-released C04 lineage, stopping on failure."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def persist(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--released-replicate', type=int, required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config['replicate'] != args.released_replicate:
        raise ValueError('release/config lineage mismatch')
    caps = {'hybrid':900, 'n1_static':180, 'n1_roles':180, 'n2_rekey':600, 'timing':120}
    if config['requested_phase_caps_seconds'] != caps or config['updates'] != 4000:
        raise ValueError('not the frozen main configuration')
    prefix = Path(args.prefix).resolve()
    if list(prefix.parent.glob(prefix.name + '*')):
        raise RuntimeError('immutable lineage output already exists')
    receipts = []
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    persist(Path(str(prefix) + '.started.json'), {
        'started_utc':started, 'replicate':config['replicate'],
        'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'requested_caps_seconds':caps, 'total_cap_seconds':sum(caps.values()),
        'batch_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    code = 0
    failure = None
    try:
        for phase, cap in caps.items():
            phase_config = config_path
            if phase == 'timing':
                timing = dict(config)
                timing['neural_checkpoints'] = {}
                for arm in ('n1_static', 'n1_roles', 'n2_rekey'):
                    checkpoint = Path(str(prefix) + '-' + arm) / 'endpoint.pt'
                    timing['neural_checkpoints'][arm] = {
                        'path':str(checkpoint),
                        'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                phase_config = Path(str(prefix) + '-timing-config.json')
                persist(phase_config, timing)
            phase_prefix = str(prefix) + '-' + phase
            command = [sys.executable, '-m', 'topoformer.campaign_composition_confirm_launch',
                       '--config', str(phase_config), '--output', phase_prefix,
                       '--prefix', phase_prefix, '--python', sys.executable,
                       '--phase', phase, '--cap', str(cap)]
            code = subprocess.run(command).returncode
            receipt_path = Path(phase_prefix + '.occupancy.json')
            if receipt_path.exists():
                receipts.append(json.loads(receipt_path.read_text()))
            if code:
                failure = {'phase':phase, 'exit_code':code}
                break
    except Exception as error:
        code = 1
        failure = {'exception':repr(error)}
    finally:
        persist(Path(str(prefix) + '.batch.json'), {
            'started_utc':started, 'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'replicate':config['replicate'], 'exit_code':code, 'failure':failure,
            'receipts':receipts,
            'total_process_occupancy_seconds':sum(r['process_occupancy_seconds'] for r in receipts),
            'gpu_processes':subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True),
            'full_process_state':subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,args'], text=True)})
    raise SystemExit(code)


if __name__ == '__main__':
    main()
