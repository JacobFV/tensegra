"""Coordinator-only budget ledger. This records releases; it does not launch jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def record(root, action, job_id, seconds, kind='exploration', source='', receipt=''):
    queue_path, budget_path = root/'queue.json', root/'budget.json'
    queue = json.loads(queue_path.read_text())
    budget = json.loads(budget_path.read_text())
    now = datetime.now(timezone.utc)
    if seconds < 0:
        raise ValueError('Negative occupancy/budget')
    if action == 'start':
        if queue['running'] is not None:
            raise ValueError('A GPU release is already active')
        if now >= datetime.fromisoformat(budget['deadline_utc'].replace('Z', '+00:00')):
            raise ValueError('Campaign elapsed deadline reached')
        ceiling = budget['gpu_seconds_ceiling']
        if kind != 'confirmation':
            confirmed = sum(j['seconds'] for j in budget['jobs'] if j['kind'] == 'confirmation')
            ceiling -= max(0, budget['confirmation_reserve_seconds'] - confirmed)
        if budget['charged_seconds'] + seconds > ceiling:
            raise ValueError('Requested release exceeds available budget/reserve')
        queue['running'] = dict(id=job_id, kind=kind, source=source,
                                cap_seconds=seconds, released_utc=now.isoformat())
        queue['ready'] = [x for x in queue['ready']
                          if (x.get('id') if isinstance(x, dict) else x) != job_id]
        budget['reserved_running_seconds'] = seconds
    else:
        running = queue['running']
        if running is None or running['id'] != job_id:
            raise ValueError('Completion does not match active GPU release')
        entry = dict(running, status=action, seconds=seconds,
                     receipt=receipt, recorded_utc=now.isoformat())
        budget['jobs'].append(entry)
        budget['charged_seconds'] += seconds
        budget['reserved_running_seconds'] = 0
        queue['completed'].append(entry)
        queue['running'] = None
    write(budget_path, budget)
    write(queue_path, queue)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['start', 'completed', 'failed', 'interrupted'])
    p.add_argument('job_id')
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--kind', default='exploration')
    p.add_argument('--source', default='')
    p.add_argument('--receipt', default='')
    p.add_argument('--root', type=Path, default=Path('research/campaigns/extended-01'))
    a = p.parse_args()
    record(a.root, a.action, a.job_id, a.seconds, a.kind, a.source, a.receipt)
