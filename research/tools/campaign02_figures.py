#!/usr/bin/env python3
"""Audit-bound extended-02 static figures. Extraction is stdlib-only; no inference."""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

BASE = 'research/campaigns/extended-02/'
RESULTS = 'research/results/campaign-02/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(root):
    inputs = {}

    def read(path):
        inputs[path] = digest(root / path)
        return json.loads((root / path).read_text())

    def verify(bindings, prefix=''):
        for path, expected in bindings.items():
            path = prefix + path
            assert digest(root / path) == expected, f'Audit binding mismatch: {path}'
            inputs[path] = expected

    e03 = read(BASE + 'review/e03-raw-audit.json')
    verify(e03['source_artifact_hashes'])
    e04 = read(BASE + 'review/e04-raw-audit.json')
    verify(e04['inputs'], RESULTS + 'e04-transfer/')
    e05 = read(BASE + 'review/e05-raw-audit.json')
    verify(e05['inputs'], RESULTS)
    e02 = read(BASE + 'reference-tight-repair-audit.json')
    verify({RESULTS + 'e02-tight-repair/episodes.jsonl.gz': e02['archive_sha256'],
            RESULTS + 'e02-tight-repair/summary.json': e02['summary_sha256']})
    rows = []
    for arm in ('lightweight', 'recurrent'):
        for update, success in e03[arm]['reported_development_success_by_update']:
            rows.append(dict(panel='e03', arm=arm, update=update,
                             success_fraction=success, examples=e03[arm]['rows']))
        for curve in e05['arms'][arm]['curves']:
            rows.append(dict(panel='e05', arm=arm, examples=e05['arms'][arm]['unique_development_support'], **curve))
    for row in e04['results']:
        rows.append(dict(panel='e04', condition=row['condition'], arm=row['arm'],
                         success_count=row['success_count'], mean_utility=row['mean_utility'],
                         mean_cost=row['mean_cost'], truncated=row['truncated'], examples=512))
    for row in e02['cells']:
        rows.append(dict(panel='e02', **row))
    assert len(rows) == len({json.dumps(row, sort_keys=True) for row in rows})
    return dict(rows=rows, inputs=inputs,
                support=dict(learned_initializers_per_family=e03['paired']['independent_lineages_per_architecture'],
                             e03_shared_dev=e03['paired']['shared_unique_episodes'],
                             e05_shared_dev={a: e05['arms'][a]['unique_development_support'] for a in e05['arms']},
                             e02_underlying_worlds=e02['unique_instances']),
                paired=dict(e05=e05['paired_final_recurrent_to_lightweight'], e02=e02['paired_success']),
                scopes=dict(e03=e03['scope'], e04=e04['scope'], e05=e05['scope'], e02=e02['limitations']))


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'topoformer-campaign-02', 'savefig.dpi': 180})

    def select(panel, **keys):
        return [r for r in data['rows'] if r['panel'] == panel and all(r.get(k) == v for k, v in keys.items())]

    def finish(fig, name, caption):
        fig.text(.02, .015, caption, fontsize=8, va='bottom')
        fig.tight_layout(rect=(0, .14, 1, 1))
        for ext in ('png', 'svg', 'pdf'):
            metadata = {'Date': None} if ext == 'svg' else ({'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {})
            fig.savefig(output / f'{name}.{ext}', metadata=metadata)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for i, arm in enumerate(('lightweight', 'recurrent')):
        early = sorted(select('e03', arm=arm), key=lambda r: r['update'])
        mixed = sorted(select('e05', arm=arm), key=lambda r: r['update'])
        axes[0].plot([r['update'] for r in early], [r['success_fraction'] * r['examples'] for r in early], 'o-', color=f'C{i}', label=arm)
        axes[1].plot([r['update'] for r in mixed], [r['success_count'] for r in mixed], 'o-', color=f'C{i}', label=arm)
        axes[2].plot([r['update'] for r in mixed], [r['utility'] for r in mixed], 'o-', color=f'C{i}', label=arm)
    for ax, title in zip(axes, ('E03 easy acquisition', 'E05 mixed acquisition', 'E05 mixed utility')):
        ax.set(title=title, xlabel='Optimizer update')
        ax.legend(fontsize=7)
        ax.axvline(600, color='gray', linewidth=.7, linestyle=':')
    for ax in axes[:2]:
        ax.set(ylabel='Verified successes / 128', ylim=(-3, 134))
    axes[2].set(ylabel='Mean recorded utility')
    finish(fig, 'acquisition',
           'One initialization per learned family. Each study reuses the same paired 128 DEV worlds across its six evaluations; no best-checkpoint selection.\n'
           'Fixed endpoint: update 600. E03 and E05 change teacher/curriculum/input distribution, so their difference is not a controlled forgetting effect.\n'
           'DEV curves are learned closed-loop actions; teacher rollout success is a different training metric. Width 1024 does not imply parameter/work matching.')

    conditions = ('easy', 'medium', 'hard_feasible', 'work_tight', 'expensive_tools', 'expensive_travel', 'obstacle')
    arms = ('lightweight', 'recurrent', 'reference-cheap', 'reference-always_tool', 'reference-cheap_first')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, title in zip(axes, ('success_count', 'mean_utility'), ('Verified success / 512', 'Mean recorded utility')):
        matrix = [[select('e04', condition=c, arm=a)[0][metric] for c in conditions] for a in arms]
        bounds = dict(vmin=0, vmax=512, cmap='Blues') if metric == 'success_count' else dict(vmin=-.1, vmax=1, cmap='RdYlGn')
        ax.imshow(matrix, aspect='auto', **bounds)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                text = str(value) if metric == 'success_count' else f'{value:.3f}'
                color = 'white' if metric == 'success_count' and value > 280 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)
        ax.set(title='E04 frozen transfer · ' + title, xticks=range(7),
               xticklabels=['Easy', 'Medium', 'Hard\nfeasible', 'Tight\nwork', 'Tool\nprice', 'Travel\nprice', 'Obstacle'],
               yticks=range(5), yticklabels=['Lightweight', 'Recurrent', 'Supplied cheap', 'Supplied always-tool', 'Supplied cheap-first'])
    finish(fig, 'frozen-transfer',
           '512 paired worlds per cell, shared by all five policies; policies are not independent replications. One learned initialization per family.\n'
           'References consume richer raw public JSON than learned encodings. Utility uses the archived price regime (compute price zero); actual compute is separate.\n'
           'Fixed learned endpoints from E03; inspected development transfer, no checkpoint choice or general architecture-superiority claim.')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, budget in zip(axes, ('standard', 'tight')):
        for i, mode in enumerate(('cheap_first', 'cheap_first_fallback_v2')):
            ss = [select('e02', condition=budget + '_' + menu, mode=mode)[0] for menu in ('fixed', 'remaining')]
            ax.plot([0, 1], [r['successes'] for r in ss], 'o-', color=f'C{i}', label='Original reference' if i == 0 else 'Route fallback v2')
            for j, row in enumerate(ss):
                ax.annotate(str(row['successes']), (j, row['successes']), xytext=(4, 6 if i else -14), textcoords='offset points', fontsize=8, color=f'C{i}')
        ax.set(title='E02 ' + ('standard public work budget' if budget == 'standard' else 'tight 128-work budget'),
               xticks=[0, 1], xticklabels=['Fixed call menu', '+ exact remaining budget'],
               ylabel='Verified success / 512', ylim=(100, 550), xlim=(-.1, 1.1))
        ax.legend(fontsize=7)
    finish(fig, 'reference-budget-repair',
           'Supplied reference-policy/action-menu interventions, not learned repair. 4096 episodes reuse 512 underlying worlds across conditions and policies.\n'
           'Vertical axis zoomed. Remaining failures do not prove that more than 128 work is necessary; solver counterfactuals were not rerun.\n'
           'Original outcomes are preserved under their original catalogue. Development anchors are not fresh confirmation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path(BASE + 'figures'))
    parser.add_argument('--render-data', type=Path)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_data:
        data = json.loads(gzip.decompress(args.render_data.read_bytes()))
    else:
        data = extract(args.root.resolve())
        raw = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        packed = gzip.compress(raw, mtime=0)
        (args.output / 'plotted-data.json.gz').write_bytes(packed)
        manifest = dict(rows=len(data['rows']), inputs=data['inputs'],
                        gzip_sha256=hashlib.sha256(packed).hexdigest(), json_sha256=hashlib.sha256(raw).hexdigest(),
                        support=data['support'], scope='Audited extended-02 metrics only; no model/solver inference')
        (args.output / 'data-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    if not args.extract_only:
        render(data, args.output)
    print(json.dumps(dict(rows=len(data['rows']), inputs=len(data['inputs']), wall_seconds=time.perf_counter()-start,
                          rendered=not args.extract_only)))


if __name__ == '__main__':
    main()
