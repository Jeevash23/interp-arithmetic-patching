"""Independent check of Study 5 (PROTOCOL.md section 11), written during the test run, before its records were pulled.

It does not import code/analyze.py. From the raw per-pair records it re-parses every answer from the generated text with
the frozen strict parser, recomputes the outcomes that the decision rules use, recomputes the exact intervals, and
compares them with results/test-analysis.json; it also checks the freeze records, the order of the stages, the
first-stage checks, the span self-writes and the selected sites. Writes results/verification.json.

  python -B verify.py
"""
import gzip
import hashlib
import json
import re
import sys
from math import comb
from pathlib import Path
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import release_paths as _release_paths  # noqa: E402

STUDY = Path(__file__).resolve().parent
checks = []


def check(name, ok, detail=None):
    checks.append(dict(check=name, passed=bool(ok), detail=detail))
    print(('ok   ' if ok else 'FAIL ') + name + ('' if ok or detail is None else f': {detail}'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def records(folder):
    out = []
    for p in sorted((STUDY / 'results' / folder).glob('*.jsonl.gz')):
        with gzip.open(p, 'rt') as fh:
            out += [json.loads(s) for s in fh if s.strip()]
    return out


def parse(text):                     # the frozen parser (code/frozen/tasks.py)
    text = re.sub(r'<\|(?:im_end|endoftext)\|>', '', text).strip()
    return int(text) if re.fullmatch(r'-?\d+', text) else None


def calc(op, a, b):
    return {'add': a + b, 'sub': a - b, 'mul': a * b}[op]


def cp(k, n):
    """Exact two-sided 95% interval by bisection on the binomial tails (independent of code/stats.py)."""
    def tail_le(k, p):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - tail_le(k - 1, p) < 0.025)
    upper = 1.0 if k == n else solve(lambda p: tail_le(k, p) > 0.025)
    return lower, upper


def main():
    # 1. Freeze records
    freeze = json.loads((STUDY / 'FREEZE.json').read_text())
    changed = [f for f, d in freeze['sha256'].items() if not _release_paths.matches(STUDY / f, d, f)]
    check('every file in FREEZE.json is unchanged', not changed, changed)
    prot = json.loads((STUDY / 'PROTOCOL-FREEZE.json').read_text())
    merged = dict(prot['sha256'])
    for p in sorted(STUDY.glob('PROTOCOL-FREEZE-amendment-*.json')):
        merged.update(json.loads(p.read_text())['sha256'])
    stale = [f for f, d in merged.items() if freeze['sha256'].get(f) != d]
    check('the protocol freeze with its amendment records agrees with FREEZE.json', not stale, stale)
    check('PROTOCOL.md is the file frozen before the first cloud stage',
          _release_paths.matches(STUDY / 'PROTOCOL.md', prot['sha256']['PROTOCOL.md'], 'PROTOCOL.md'))
    orch = {s: json.loads((STUDY / f'results/orchestration/{s}.json').read_text()) for s in ('first', 'devsites', 'test')}
    check('the first stage started after the protocol freeze', orch['first']['started_utc'] > prot['frozen_at_utc'])
    check('the test stage started after FREEZE.json', orch['test']['started_utc'] > freeze['frozen_at_utc'],
          [orch['test']['started_utc'], freeze['frozen_at_utc']])
    check('every stage finished', all(o['status'] == 'finished' for o in orch.values()))

    # 2. First-stage checks
    first = json.loads((STUDY / 'results/first/checks.json').read_text())
    rep = first['reproduction']
    check('reproduction: every saved Study 3 record regenerated token for token',
          rep['pass_check'] and rep['tokens_equal'] == rep['records'] == 27 and all(c['tokens_equal'] and c['prompt_equal'] for c in rep['checks']))
    check('instrumentation passed (frozen checks and span null writes at most 1e-5)',
          first['instrumentation']['pass_check'] and all(c['max_logit_abs'] <= 1e-5 for c in first['instrumentation']['span_checks']))
    check('class means computed on 24 problems per value', all(c == 24 for c in first['means']['counts']))

    # 3. Test records
    rows = rd(STUDY / 'data/test.jsonl')
    by_id = {r['id']: r for r in rows}
    group_donor = {r['group']: r['donor'] for r in rows}
    recs = records('test')
    index = {}
    for r in recs:
        assert r['key'] not in index, r['key']
        index[r['key']] = r
    check('records for all 72 test pairs', {r['id'] for r in recs} == set(by_id), len({r['id'] for r in recs}))
    bad = [r['key'] for r in recs if 'text' in r and parse(r['text']) != r['answer']]
    check('every stored answer equals the frozen parser applied to the generated text', not bad, bad[:5])
    base = {i: index[f'{i}/baseline/recipient']['answer'] for i in by_id}
    selfs = [r for r in recs if r.get('condition') == 'self']
    check('span self-writes reproduce the unpatched output', selfs and all(r['answer'] == base[r['id']] for r in selfs), len(selfs))
    sel = json.loads((STUDY / 'results/development-selection.json').read_text())
    cost = json.loads((STUDY / 'results/cost-decision.json').read_text())
    roles_swept = sorted({r['role'] for r in recs if r['kind'] == 'direction' and r['condition'] == 'donor' and r['alpha'] == 1})
    expected_roles = set(cost['test_direction_sweep_roles']) | {d['role'] for d in sel['direction_sites_test'] if d['alpha'] == 1}
    check('the direction sweep covers the positions of the cost decision', set(roles_swept) == expected_roles, roles_swept)

    def target(row, cond):
        r = row['recipient']
        if cond in ('donor', 'noise'):
            z = row['donor']['z']
        elif cond == 'random_source':
            z = group_donor[(row['group'] + 1) % len(group_donor)]['z']
        else:
            z = row['controls'][cond]['z']
        return calc(r['remaining'], z, r['k'])

    def count(keyfn, cond='donor'):
        k = n = 0
        for i, row in by_id.items():
            rec = index[keyfn(i)]
            n += 1
            k += rec['answer'] == target(row, cond)
        return k, n

    analysis = json.loads((STUDY / 'results/test-analysis.json').read_text())
    mism = []
    for d in sel['direction_sites_test']:
        L, role, a = d['layer'], d['role'], d['alpha']
        for cond, fn in [('donor', lambda i: f'{i}/direction/{L}/{role}/{a}/donor'),
                         ('noise', lambda i: f'{i}/direction-noise/{L}/{role}/{a}'),
                         ('same_leading_different_value', lambda i: f'{i}/direction/{L}/{role}/{a}/same_leading_different_value')]:
            k, n = count(fn, cond)
            key = {'donor': 'donor', 'noise': 'noise', 'same_leading_different_value': 'same_leading'}[cond]
            got = analysis['sites']['direction'][role][key]
            lo, hi = cp(k, n)
            if (got['k'], got['n']) != (k, n) or abs(got['exact95'][0] - lo) > 1e-6 or abs(got['exact95'][1] - hi) > 1e-6:
                mism.append((role, cond, k, n, got['k'], got['n']))
    for s in sel['span_sites']:
        L, span = s['layer'], s['span']
        k, n = count(lambda i: f'{i}/span/{L}/{span}/donor')
        got = analysis['sites']['span'][span]['donor']
        if (got['k'], got['n']) != (k, n):
            mism.append((span, 'donor', k, n, got['k'], got['n']))
        for src in ['same_z_decomposition', 'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value', 'random_source']:
            k, n = count(lambda i: f'{i}/span/{L}/{span}/{src}', src)
            got = analysis['sites']['span'][span]['sources'][src]
            if (got['k'], got['n']) != (k, n):
                mism.append((span, src, k, n, got['k'], got['n']))
        k, n = count(lambda i: f'{i}/span-noise/{L}/{span}', 'noise')
        got = analysis['sites']['span'][span]['noise']
        if (got['k'], got['n']) != (k, n):
            mism.append((span, 'noise', k, n, got['k'], got['n']))
    check('selected-site counts and exact intervals recomputed from the raw records', not mism, mism)

    sweep_mism = []
    for role in roles_swept:
        for row in analysis['sweeps']['direction'][f'{role}/1']:
            L = row['layer']
            k, n = count(lambda i: f'{i}/direction/{L}/{role}/1/donor')
            u = sum(index[f'{i}/direction/{L}/{role}/1/donor']['answer'] == base[i] for i in by_id)
            if (k, n, u) != (row['cf'], row['n'], row['unchanged']):
                sweep_mism.append((role, L))
    for span in ('post_operand', 'first_step'):
        for row in analysis['sweeps']['span'][span]:
            L = row['layer']
            k, n = count(lambda i: f'{i}/span/{L}/{span}/donor')
            u = sum(index[f'{i}/span/{L}/{span}/donor']['answer'] == base[i] for i in by_id)
            fo = sum(index[f'{i}/span/{L}/{span}/donor']['answer'] == calc(by_id[i]['recipient']['remaining'],
                     calc(by_id[i]['donor']['first'], by_id[i]['recipient']['a'], by_id[i]['recipient']['b']), by_id[i]['recipient']['k'])
                     for i in by_id)
            if (k, n, u, fo) != (row['cf'], row['n'], row['unchanged'], row['first_op']):
                sweep_mism.append((span, L))
    check('every sweep cell (complete counterfactual, unchanged, first operation) recomputed', not sweep_mism, sweep_mism[:5])

    # 4. Rule verdicts from the recomputed counts and the saved conservative bounds
    R = analysis['rules']
    v = {}
    for role in ('first_stage_end', 'prompt_end'):
        site = analysis['sites']['direction'][role]
        a, diff = site['donor'], site['donor_minus_noise']
        v[role] = ('carries' if a['rate'] >= 0.2 and a['lower'] >= 0.1 and diff['cluster95'][0] > 0
                   else 'does not carry' if a['upper'] < 0.2 else 'inconclusive')
    for span in ('post_operand', 'first_step'):
        site = analysis['sites']['span'][span]
        a, diff = site['donor'], site['donor_minus_noise']
        v[span] = ('carries' if a['rate'] >= 0.2 and a['lower'] >= 0.1 and diff['cluster95'][0] > 0
                   else 'does not carry' if a['upper'] < 0.2 else 'inconclusive')
    check('R1 and R2 site verdicts follow from the saved bounds',
          all(R['R1']['sites'][r] == v[r] for r in ('first_stage_end', 'prompt_end'))
          and all(R['R2']['sites'][s] == v[s] for s in ('post_operand', 'first_step')), v)
    result = dict(checks=checks, passed=sum(c['passed'] for c in checks), total=len(checks))
    (STUDY / 'results/verification.json').write_text(json.dumps(result, indent=1) + '\n')
    print(f"{result['passed']} of {result['total']} checks passed")
    sys.exit(0 if result['passed'] == result['total'] else 1)


if __name__ == '__main__':
    main()
