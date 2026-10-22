"""Study 5 analysis (PROTOCOL.md sections 6, 7, 9 and 10). Reads the gzipped per-pair records pulled from the Volume.

  python -B code/analyze.py select        # after the first stage: development sweeps -> results/development-selection.json
  python -B code/analyze.py development   # after the second development pass -> results/development-analysis.json
  python -B code/analyze.py test          # after the test stage -> results/test-analysis.json
  python -B code/analyze.py summary       # results/summary.json (every number the paper uses, by name)

Standard library and NumPy only. Refuses to overwrite the selection.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
from stats import paired_difference, proportion  # noqa: E402  copy of Study 3's estimators

ROLES = ['a_end', 'b_end', 'first_stage_end', 'prompt_end']
POST = ['first_stage_end', 'prompt_end']
SPANS = ['post_operand', 'first_step']
BLOCKS = 36
SIM_SEED, SIM_DRAWS = 20261017, 2000


def calc(op, a, b):
    return {'add': a + b, 'sub': a - b, 'mul': a * b}[op]


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def load(folder):
    recs = []
    for p in sorted((STUDY / 'results' / folder).glob('*.jsonl.gz')):
        with gzip.open(p, 'rt') as fh:
            recs += [json.loads(s) for s in fh if s.strip()]
    return recs


class Split:
    def __init__(self, name, folders):
        self.rows = rd(STUDY / 'data' / f'{name}.jsonl')
        self.by_id = {r['id']: r for r in self.rows}
        self.group_donor = {r['group']: r['donor'] for r in self.rows}
        self.records = []
        for f in folders:
            self.records += load(f)
        self.index = {}
        for r in self.records:
            self.index.setdefault(r['key'], r)
        self.unpatched = {}
        for r in self.records:
            if r.get('condition') == 'recipient_full':
                prev = self.unpatched.setdefault(r['id'], r['answer'])
                assert prev == r['answer'], 'unpatched answers differ between stages'
        self.ids = [r['id'] for r in self.rows]
        self.groups = [r['group'] for r in self.rows]

    def source(self, row, name):
        if name in ('donor', 'noise'):
            return row['donor']
        if name == 'random_source':
            return self.group_donor[(row['group'] + 1) % len(self.group_donor)]
        return row['controls'][name]

    def target(self, row, name):
        r = row['recipient']
        return calc(r['remaining'], self.source(row, name)['z'], r['k'])

    def classes(self, rec):
        row = self.by_id[rec['id']]
        r, d = row['recipient'], row['donor']
        ans = rec['answer']
        target = self.target(row, rec['condition'])
        unchanged = ans == self.unpatched[rec['id']]
        alt = row['digit_alternatives']
        return dict(cf=ans == target, unchanged=unchanged,
                    donor_copy=ans == d['y'], first_op=ans == calc(r['remaining'], calc(d['first'], r['a'], r['b']), r['k']),
                    invalid=ans is None,
                    # descriptive (AMENDMENTS.md, amendment 2): the donor's leading digit spliced into the recipient's
                    # answer or intermediate (Study 3's splice outcomes), and a changed answer that shares only the
                    # counterfactual's leading digit
                    answer_splice=ans == alt['donor_answer_leading_digit_splice'],
                    z_splice=ans == alt['intermediate_leading_digit_splice'],
                    cf_leading_only=(ans is not None and not unchanged and ans != target
                                     and str(ans)[0] == str(target)[0] and len(str(ans)) == len(str(target))))

    def get(self, key_fn):
        """Records for every pair, in split order, by a key built from the pair id; None where missing."""
        return [self.index.get(key_fn(i)) for i in self.ids]

    def rate(self, key_fn, cls='cf', interval=True):
        recs = self.get(key_fn)
        if any(r is None for r in recs):
            return None
        flags = [self.classes(r)[cls] for r in recs]
        if not interval:
            return dict(k=sum(flags), n=len(flags), rate=sum(flags) / len(flags))
        return proportion(flags, self.groups)

    def flags(self, key_fn, cls='cf'):
        return [self.classes(r)[cls] for r in self.get(key_fn)]


def dkey(layer, role, alpha, source='donor'):
    return lambda i: f'{i}/direction/{layer}/{role}/{alpha}/{source}'


def nkey(layer, role, alpha):
    return lambda i: f'{i}/direction-noise/{layer}/{role}/{alpha}'


def skey(layer, span, source='donor'):
    return lambda i: f'{i}/span/{layer}/{span}/{source}'


def snkey(layer, span):
    return lambda i: f'{i}/span-noise/{layer}/{span}'


def sweep_tables(s, alphas):
    out = dict(direction={}, span={}, scores={})
    for alpha in alphas:
        for role in ROLES:
            rows = []
            for layer in range(BLOCKS):
                a = s.rate(dkey(layer, role, alpha), 'cf', interval=False)
                if a is None:
                    continue
                e = s.rate(dkey(layer, role, alpha), 'unchanged', interval=False)
                extra = {c: s.rate(dkey(layer, role, alpha), c, interval=False)['k']
                         for c in ('donor_copy', 'answer_splice', 'z_splice', 'cf_leading_only', 'invalid')}
                rows.append(dict(layer=layer, cf=a['k'], unchanged=e['k'], n=a['n'], **extra))
            if rows:
                out['direction'][f'{role}/{alpha}'] = rows
    for span in SPANS:
        rows = []
        for layer in range(BLOCKS):
            a = s.rate(skey(layer, span), 'cf', interval=False)
            if a is None:
                continue
            e = s.rate(skey(layer, span), 'unchanged', interval=False)
            f = s.rate(skey(layer, span), 'first_op', interval=False)
            extra = {c: s.rate(skey(layer, span), c, interval=False)['k']
                     for c in ('donor_copy', 'answer_splice', 'z_splice', 'cf_leading_only', 'invalid')}
            rows.append(dict(layer=layer, cf=a['k'], unchanged=e['k'], first_op=f['k'], n=a['n'], **extra))
        if rows:
            out['span'][span] = rows
    base = {r['id']: r for r in s.records if r['kind'] == 'unpatched_score'}
    for role in POST:
        rows = []
        for layer in range(BLOCKS):
            recs = [s.index.get(f'{i}/direction-score/{layer}/{role}/1') for i in s.ids]
            if any(r is None for r in recs) or not base:
                continue
            delta = [r['counterfactual_log_probability'] - r['recipient_log_probability'] for r in recs]
            before = [base[r['id']]['counterfactual_log_probability'] - base[r['id']]['recipient_log_probability'] for r in recs]
            rows.append(dict(layer=layer, mean_delta=float(np.mean(delta)), mean_change=float(np.mean(np.subtract(delta, before))),
                             flips=int(sum(d > 0 for d in delta)), n=len(delta)))
        if rows:
            out['scores'][role] = rows
    return out


def select(s):
    """PROTOCOL.md section 7."""
    tables = sweep_tables(s, [1, 4])
    direction = []
    for role in ROLES:
        cells = [(alpha, r) for alpha in [1, 4] for r in tables['direction'][f'{role}/{alpha}']]
        best = max(c[1]['cf'] for c in cells)
        if best > 0:
            alpha, r = sorted(cells, key=lambda c: (-c[1]['cf'], c[0], c[1]['layer']))[0]
            rule = 'largest development rate of the complete counterfactual'
        else:
            alpha, r = sorted(cells, key=lambda c: (c[1]['unchanged'], c[0], c[1]['layer']))[0]
            rule = 'no success at any block: lowest rate of an unchanged answer'
        direction.append(dict(layer=r['layer'], role=role, alpha=alpha, in_sweep=True, rule=rule, development_cf=r['cf'],
                              development_unchanged=r['unchanged'], n=r['n']))
    spans = []
    for span in SPANS:
        rows = tables['span'][span]
        if max(r['cf'] for r in rows) > 0:
            r = sorted(rows, key=lambda x: (-x['cf'], x['layer']))[0]
            rule = 'largest development rate of the complete counterfactual'
        else:
            r = sorted(rows, key=lambda x: (x['unchanged'], x['layer']))[0]
            rule = 'no success at any block: lowest rate of an unchanged answer'
        spans.append(dict(layer=r['layer'], span=span, rule=rule, development_cf=r['cf'], development_unchanged=r['unchanged'], n=r['n']))
    test_sites = [dict(d, in_sweep=d['alpha'] == 1) for d in direction]
    return dict(rule='PROTOCOL.md section 7, on the 36 development pairs', direction_sites=direction,
                direction_sites_test=test_sites, span_sites=spans), tables


def site_results(s, sel, n_label):
    """Selected-site results with intervals: donor, controls and Gurnee's measure."""
    out = dict(direction={}, span={})
    base = {r['id']: r for r in s.records if r['kind'] == 'unpatched_score'}
    for site in sel['direction_sites']:
        layer, role, alpha = site['layer'], site['role'], site['alpha']
        res = dict(site=site, donor=s.rate(dkey(layer, role, alpha)), donor_unchanged=s.rate(dkey(layer, role, alpha), 'unchanged'),
                   noise=s.rate(nkey(layer, role, alpha)), noise_unchanged=s.rate(nkey(layer, role, alpha), 'unchanged'),
                   same_leading=s.rate(dkey(layer, role, alpha, 'same_leading_different_value')))
        if res['donor'] and res['noise']:
            res['donor_minus_noise'] = paired_difference(s.flags(dkey(layer, role, alpha)), s.flags(nkey(layer, role, alpha)), s.groups)
        recs = [s.index.get(f'{i}/direction-score/{layer}/{role}/{alpha}') for i in s.ids]
        if all(r is not None for r in recs) and base:
            delta = [r['counterfactual_log_probability'] - r['recipient_log_probability'] for r in recs]
            before = [base[r['id']]['counterfactual_log_probability'] - base[r['id']]['recipient_log_probability'] for r in recs]
            res['gurnee'] = dict(mean_delta=float(np.mean(delta)), mean_change=float(np.mean(np.subtract(delta, before))),
                                 flips=proportion([d > 0 for d in delta], s.groups))
        out['direction'][role] = res
    for site in sel['span_sites']:
        layer, span = site['layer'], site['span']
        res = dict(site=site, donor=s.rate(skey(layer, span)), donor_unchanged=s.rate(skey(layer, span), 'unchanged'),
                   donor_first_op=s.rate(skey(layer, span), 'first_op'),
                   noise=s.rate(snkey(layer, span)), noise_unchanged=s.rate(snkey(layer, span), 'unchanged'),
                   sources={src: s.rate(skey(layer, span, src)) for src in ['same_z_decomposition', 'same_z_different_answer',
                                                                            'different_z_same_answer', 'same_leading_different_value',
                                                                            'random_source']})
        if res['donor'] and res['noise']:
            res['donor_minus_noise'] = paired_difference(s.flags(skey(layer, span)), s.flags(snkey(layer, span)), s.groups)
        out['span'][span] = res
    return out


def site_verdict(res):
    a = res.get('donor')
    if not a:
        return 'missing'
    diff = res.get('donor_minus_noise')
    if a['rate'] >= 0.20 and a['lower'] >= 0.10 and diff and diff['cluster95'][0] > 0:
        return 'carries'
    if a['upper'] < 0.20:
        return 'does not carry'
    return 'inconclusive'


def overall(verdicts):
    if 'carries' in verdicts:
        return 'supported'
    if all(v == 'does not carry' for v in verdicts):
        return 'not supported'
    return 'inconclusive'


def r3(res, kind):
    if kind == 'direction':
        sl = res.get('same_leading')
        return dict(same_leading=sl, passed=bool(sl and sl['lower'] >= 0.10))
    sl, sz = res['sources'].get('same_leading_different_value'), res['sources'].get('same_z_decomposition')
    return dict(same_leading=sl, same_z=sz, passed=bool(sl and sz and sl['lower'] >= 0.10 and sz['lower'] >= 0.10))


def r4(s):
    carriers, none_above = [], True
    cells = [('direction', role, lambda l, role=role: dkey(l, role, 1)) for role in POST]
    cells += [('span', span, lambda l, span=span: skey(l, span)) for span in SPANS]
    for kind, name, fn in cells:
        for layer in range(BLOCKS):
            p = s.rate(fn(layer))
            if p is None:
                continue
            if p['rate'] >= 0.25 and p['lower'] > 0.10:
                carriers.append(dict(kind=kind, name=name, layer=layer, k=p['k'], n=p['n'], lower=p['lower']))
            if p['upper'] >= 0.20:
                none_above = False
    return dict(blocks_that_carry=carriers, no_block=not carriers and none_above,
                result='a block carries' if carriers else ('no such block' if none_above else 'inconclusive'))


def rules(s, sites):
    v1 = {role: site_verdict(sites['direction'][role]) for role in POST}
    v2 = {span: site_verdict(sites['span'][span]) for span in SPANS}
    R1, R2 = overall(list(v1.values())), overall(list(v2.values()))
    R3 = {}
    for role in POST:
        if v1[role] == 'carries':
            R3[f'direction/{role}'] = r3(sites['direction'][role], 'direction')
    for span in SPANS:
        if v2[span] == 'carries':
            R3[f'span/{span}'] = r3(sites['span'][span], 'span')
    R4 = r4(s)
    if R1 == 'supported' or R2 == 'supported':
        verdict = 'the intermediate is formed and carried'
    elif R1 == 'not supported' and R2 == 'not supported' and R4['result'] == 'no such block':
        verdict = 'no evidence that these writes carry the intermediate'
    else:
        verdict = 'inconclusive'
    return dict(R1=dict(sites=v1, overall=R1), R2=dict(sites=v2, overall=R2), R3=R3, R4=R4, verdict=verdict,
                operand_positions={role: site_verdict(sites['direction'][role]) for role in ['a_end', 'b_end']})


def by_operation(s, key_fn):
    out = {}
    for op in ['add', 'sub', 'mul']:
        flags = [s.classes(r)['cf'] for r in s.get(key_fn) if r and s.by_id[r['id']]['recipient']['remaining'] == op]
        out[op] = dict(k=sum(flags), n=len(flags))
    return out


def operating_characteristics(s, sel):
    """Verdict frequencies of R1 and R2 on simulated test sets: 24 donor groups drawn with replacement from the
    development groups, each with its three pairs (exact and cluster intervals recomputed). Seeded."""
    rng = np.random.default_rng(SIM_SEED)
    groups = sorted(set(s.groups))
    by_group = defaultdict(list)
    for idx, g in enumerate(s.groups):
        by_group[g].append(idx)
    cells = {f'direction/{d["role"]}': (dkey(d['layer'], d['role'], d['alpha']), nkey(d['layer'], d['role'], d['alpha']))
             for d in sel['direction_sites'] if d['role'] in POST}
    cells.update({f'span/{x["span"]}': (skey(x['layer'], x['span']), snkey(x['layer'], x['span'])) for x in sel['span_sites']})
    out = {}
    from stats import clopper_pearson
    for name, (kf, nf) in cells.items():
        a, b = s.flags(kf), s.flags(nf)
        counts = defaultdict(int)
        for _ in range(SIM_DRAWS):
            draw = rng.choice(groups, size=24, replace=True)
            idx = [i for g in draw for i in by_group[g]]
            fa, fb = [a[i] for i in idx], [b[i] for i in idx]
            k, n = sum(fa), len(fa)
            cp = clopper_pearson(k, n)
            # cluster interval by a small inner bootstrap
            inner = rng.integers(0, 24, (400, 24))
            gsum = np.array([sum(fa[3 * j:3 * j + 3]) for j in range(24)])
            gdiff = np.array([sum(fa[3 * j:3 * j + 3]) - sum(fb[3 * j:3 * j + 3]) for j in range(24)])
            boot = gsum[inner].sum(1) / (3 * 24)
            dboot = gdiff[inner].sum(1) / (3 * 24)
            lower, upper = min(cp[0], np.quantile(boot, 0.025)), max(cp[1], np.quantile(boot, 0.975))
            rate = k / n
            if rate >= 0.20 and lower >= 0.10 and np.quantile(dboot, 0.025) > 0:
                counts['carries'] += 1
            elif upper < 0.20:
                counts['does not carry'] += 1
            else:
                counts['inconclusive'] += 1
        out[name] = {k: v / SIM_DRAWS for k, v in counts.items()}
    return dict(method='24 donor groups drawn with replacement from the 12 development groups, %d draws, seed %d; inner '
                       'cluster bootstrap of 400 resamples' % (SIM_DRAWS, SIM_SEED), cells=out)


def analyse(split):
    folders = {'development': ['development', 'devsites'], 'test': ['test']}[split]
    s = Split(split, folders)
    sel = json.loads((STUDY / 'results/development-selection.json').read_text())
    if split == 'test':
        sel = dict(sel, direction_sites=sel['direction_sites_test'])
    sites = site_results(s, sel, split)
    result = dict(split=split, pairs=len(s.rows), donor_groups=len(set(s.groups)),
                  baselines=dict(recipient_correct=sum(s.unpatched[i] == s.by_id[i]['recipient']['y'] for i in s.ids),
                                 donor_correct=sum(s.index[f'{i}/baseline/donor']['answer'] == s.by_id[i]['donor']['y'] for i in s.ids
                                                   if f'{i}/baseline/donor' in s.index)),
                  sweeps=sweep_tables(s, [1, 4] if split == 'development' else [1]), sites=sites, rules=rules(s, sites),
                  by_operation={f'direction/{d["role"]}': by_operation(s, dkey(d['layer'], d['role'], d['alpha'])) for d in sel['direction_sites']})
    result['by_operation'].update({f'span/{x["span"]}': by_operation(s, skey(x['layer'], x['span'])) for x in sel['span_sites']})
    self_checks = [r for r in s.records if r.get('condition') == 'self']
    result['span_self_writes'] = dict(n=len(self_checks), equal_to_unpatched=sum(r['answer'] == s.unpatched[r['id']] for r in self_checks))
    seconds = [r['seconds'] for r in s.records if 'seconds' in r and r['kind'] in ('direction', 'span', 'baseline')]
    result['timing'] = dict(generations=len(seconds), mean_seconds=float(np.mean(seconds)) if seconds else None,
                            total_generation_hours=float(np.sum(seconds)) / 3600 if seconds else None)
    if split == 'development':
        result['operating_characteristics'] = operating_characteristics(s, sel)
    target = STUDY / 'results' / f'{split}-analysis.json'
    target.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(rules=result['rules'], timing=result['timing']), indent=1))


def summary():
    """results/summary.json: the design constants, the first-stage checks, the selection, the development and test
    analyses, run times and cost, by name (the paper's make_numbers.py reads this file)."""
    manifest = json.loads((STUDY / 'data/data-manifest.json').read_text())
    checks = json.loads((STUDY / 'results/first/checks.json').read_text())
    sel = json.loads((STUDY / 'results/development-selection.json').read_text())
    dev = json.loads((STUDY / 'results/development-analysis.json').read_text())
    test_path = STUDY / 'results/test-analysis.json'
    test = json.loads(test_path.read_text()) if test_path.exists() else None
    rep = checks['reproduction']
    span_lengths = {}
    for r in load('development'):
        if r['kind'] == 'span' and r['condition'] == 'donor':
            span_lengths.setdefault(r['span'], len(r['positions']))
    fit = rd(STUDY / 'data/fitting.jsonl')
    out = dict(
        study='5-direction-and-span',
        design=dict(fit_problems=len(fit), fit_per_value=manifest['fitting.jsonl']['per_value'],
                    fit_values=manifest['fitting.jsonl']['values'], span_lengths=span_lengths,
                    development_pairs=dev['pairs'], test_pairs=test['pairs'] if test else None, blocks=BLOCKS),
        first_stage=dict(instrumentation=checks['instrumentation']['pass_check'],
                         frozen_instrumentation=checks['instrumentation']['frozen_instrumentation'],
                         span_null_checks=len(checks['instrumentation']['span_checks']),
                         span_null_max=max(c['max_logit_abs'] for c in checks['instrumentation']['span_checks']),
                         reproduction=dict(records=rep['records'], tokens_equal=rep['tokens_equal'],
                                           vectors_checked=rep['vectors_checked'], vectors_bytes_equal=rep['vectors_bytes_equal'],
                                           vector_max_abs_difference=rep['vector_max_abs_difference'], passed=rep['pass_check']),
                         gpu=checks['runtime']['gpu']),
        selection=sel, development=dev, test=test)
    for name in ('billing-log.jsonl', 'transfer-log.jsonl'):
        path = STUDY / 'results' / name
        if path.exists():
            out[name.split('.')[0].replace('-', '_')] = rd(path)
    (STUDY / 'results/summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print('wrote results/summary.json')


def main():
    cmd = sys.argv[1]
    if cmd == 'select':
        target = STUDY / 'results/development-selection.json'
        if target.exists():
            raise SystemExit('the selection exists; not overwritten')
        s = Split('development', ['development'])
        sel, tables = select(s)
        target.write_text(json.dumps(sel, indent=1) + '\n')
        (STUDY / 'results/development-sweeps.json').write_text(json.dumps(tables, indent=1) + '\n')
        print(json.dumps(sel, indent=1))
    elif cmd in ('development', 'test'):
        analyse(cmd)
    elif cmd == 'summary':
        summary()
    else:
        raise SystemExit(cmd)


if __name__ == '__main__':
    main()
