"""Assemble research/campaigns/extended-05/review/a-independent-audit.json from the component outputs.
usage: python compose.py <scratch_dir> <out.json>"""
import json
import sys

S, OUT = sys.argv[1:3]
L = lambda n: json.load(open(f'{S}/{n}.json'))  # noqa: E731
hr, ev, rc, it, dc = L('hr_audit'), L('eval_audit'), L('receipts_audit'), L('integrity'), L('d_compare')
root = {
    'A-HR': {'G1a_excl_abstain': .0076962890625, 'G1b': -.0021267578125},
    'A-HR2': {'G1a_multi_T': .026045507812500004, 'T_plus_OI': .028306640625, 'B': .0061, 'G1b_multi_T': .021401302083333334},
    'A-PI-T': {'PI-T-1': [.035055966504822345, .043941414239581644, .03625684717331146],
               'PI-T-2_vs_best': [-.001058877245177614, -.0011784099791684044, -.0016895395454386009]},
    'A-CF-T': {'PI-T-1': [.02473643514661472, .02148507196279814, .02768350658756613],
               'PI-T-2_vs_best': [-.0020154203221353084, -.001817662412201826, -.0011205949749339972]},
}
mine = {
    'A-HR': {'G1a_excl_abstain': hr['G1a']['G1a_excl_abstain']['mean'],
             'G1b_rootgrid': hr['G1b']['G1b_OI_rootgrid_(m<=.05)']['mean'],
             'G1b_widegrid': hr['G1b']['G1b_OI_excl_abstain']['mean']},
    'A-HR2': {'G1a_multi_T': hr['G1a']['G1a_multi_T']['mean'], 'T_plus_OI': hr['G1a']['G1a_multi_T_plus_OI_excl_abstain']['mean'],
              'B': hr['G1a']['G1a_B_budget_from_AHR(budget+call_now)']['mean'],
              'G1b_multi_T_single_type': hr['G1b']['G1b_multi_T_single_type_all_anchors']['mean'],
              'G1b_multi_T_per_anchor': hr['G1b']['G1b_multi_T_all_anchors']['mean'],
              'G1b_multi_T_trigger_only': hr['G1b']['G1b_multi_T_trigger_anchors']['mean']},
}
for n in ('A-PI-T', 'A-CF-T'):
    mine[n] = {'PI-T-1': [L_['PI-T-1']['utility_diff'] for L_ in ev[n]['lineages'].values()],
               'PI-T-2_vs_best': [L_['PI-T-2']['diff_vs_best'] for L_ in ev[n]['lineages'].values()]}
disc = []
for n in ('A-PI-T', 'A-CF-T'):
    for k in ('PI-T-1', 'PI-T-2_vs_best'):
        for a, b in zip(root[n][k], mine[n][k]):
            if abs(a - b) > 1e-9:
                disc.append((n, k, a, b))
if abs(root['A-HR']['G1a_excl_abstain'] - mine['A-HR']['G1a_excl_abstain']) > 1e-9:
    disc.append(('A-HR', 'G1a', root['A-HR']['G1a_excl_abstain'], mine['A-HR']['G1a_excl_abstain']))
if abs(root['A-HR2']['G1a_multi_T'] - mine['A-HR2']['G1a_multi_T']) > 1e-9:
    disc.append(('A-HR2', 'G1a_multi', root['A-HR2']['G1a_multi_T'], mine['A-HR2']['G1a_multi_T']))
out = dict(version='e05-a-independent-audit-v1', base_commit='88bd3b0c',
           reproduction=dict(root=root, auditor=mine, exact_numeric_discrepancies=disc,
                             note='G1b values are independent re-implementations (own ridge, own fold assignment); '
                                  'they reproduce sign/gate outcome, not digits.'),
           hr=hr, eval={n: {k: v for k, v in ev[n].items()} for n in ('A-PI-T', 'A-CF-T')},
           eval_disjoint_240M_260M=ev['disjoint_240M_vs_260M'],
           d_teacher_rerun=dc, integrity=it,
           receipts=dict(groups=rc['groups'], commits=rc['commits'], controller_first_commit=rc['controller_first_commit'],
                         registry_commits=rc['registry_commits'],
                         receipts=[{k: r[k] for k in ('name', 'started', 'ended', 'exit', 'cpu', 'cwd')} for r in rc['receipts']]),
           auditor_compute=dict(remote_metered_core_s=19.29,
                                remote_receipts=['aaudit-pack-20260927T040542-1081186 (0.23)',
                                                 'aaudit-dspot-20260927T041252-1081725 (0.01, exit 127: python not on PATH)',
                                                 'aaudit-dspot-20260927T041255-1081737 (3.09, exit 1: spawn without __main__ guard)',
                                                 'aaudit-dspot-20260927T041313-1081783 (3.49, exit 1: result key)',
                                                 'aaudit-dspot-20260927T041327-1081878 (12.47, exit 0)'],
                                unmetered_remote='directory listings and two ssh cat transfers (< 1 core-s, estimated)',
                                local_core_s_estimate=640,
                                local_note='first hr_audit run used unrestricted multithreaded BLAS (~587 core-s user time for 37 s wall); '
                                           'later runs pinned to 2 threads (~16-21 core-s each)'))
json.dump(out, open(OUT, 'w'), indent=1, default=str)
print('discrepancies', disc)
