"""How often each answer-prefix rule of PROTOCOL.md section 12 would reach each verdict on a 48-group test split that
behaved like the development split: draw 48 donor groups with replacement from the 12 development groups (each
group keeps its three pairs and their saved outputs), apply the rules with the study's estimators (2,000 bootstrap
resamples to keep the run short) and repeat 400 times. Development data only; run before the freeze.

  python -B code/operating_characteristics.py <candidate>   # writes results/<candidate>/development/operating-characteristics.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
import analyze as A  # noqa: E402
import stats as S  # noqa: E402

REPLICATES, RESAMPLES, SEED, TEST_GROUPS = 400, 2000, 20261011, 48


def main():
    candidate = sys.argv[1]
    S.RESAMPLES = RESAMPLES
    rows, base, controls, text, sweep, sites, competence = A.load(candidate, 'development')
    site = {r: (l, r) for l, r in sites}
    by = {}
    for r in controls:
        if (r['layer'], r['role']) == site['prompt_end']:
            by[r['id'], r['condition']] = r
    per = {}
    for row in rows:
        i = row['id']; rec = base[i + '/recipient_full']; donor_run = base[i + '/donor_full']
        d, sl = by[i, 'donor'], by[i, 'same_leading_different_value']
        c = A.classify(d['answer'], row, rec['answer'], donor_run['answer'])
        per[i] = dict(cls=c, informative=A.informative(row),
                      identical=d['output_ids'] == sl['output_ids'] and d['answer'] != rec['answer'])
    groups = sorted({r['group'] for r in rows})
    members = {g: [r['id'] for r in rows if r['group'] == g] for g in groups}
    rng = np.random.default_rng(SEED)
    counts = Counter()
    for _ in range(REPLICATES):
        pick = rng.choice(groups, size=TEST_GROUPS, replace=True)
        recs = [(per[i], j) for j, g in enumerate(pick) for i in members[g]]
        gr = [j for _, j in recs]
        changed = S.proportion([p['cls'] != 'unchanged' for p, _ in recs], gr)
        inf = [(p, j) for p, j in recs if p['informative']]
        sm = S.proportion([p['cls'] in A.H2_CLASSES for p, _ in inf], [j for _, j in inf]) if inf else None
        ident = S.proportion([p['identical'] for p, _ in recs], gr)
        cf = S.proportion([p['cls'] == 'counterfactual' for p, _ in recs], gr)
        copy = S.proportion([p['cls'] == 'copy' for p, _ in recs], gr)
        r1 = ('inconclusive' if sm is None else 'H1' if sm['upper'] < 0.15 else
              'H2' if sm['rate'] >= 0.15 and sm['lower'] > 0.05 else 'inconclusive')
        r2 = 'H1' if ident['upper'] < 0.20 else 'H2' if ident['rate'] >= 0.30 and ident['lower'] > 0.10 else 'inconclusive'
        cc = Counter(p['cls'] for p, _ in recs if p['cls'] != 'unchanged')
        r4 = cc.get('copy', 0) > 0 and all(cc.get('copy', 0) >= n for k, n in cc.items() if k != 'copy') and copy['lower'] > 0.10
        r0 = changed['rate'] >= 0.20
        if cf['rate'] >= 0.20:
            v = 'complete transfer'
        elif r0 and r4 and r1 == 'H1' and r2 == 'H1':
            v = 'H1 supported'
        elif r0 and 'H2' in (r1, r2) and 'H1' not in (r1, r2):
            v = 'H2 supported'
        elif 'H1' in (r1, r2) and 'H2' in (r1, r2):
            v = 'mixed'
        else:
            v = 'inconclusive'
        counts['verdict', v] += 1
        counts['R1', r1] += 1
        counts['R2', r2] += 1
        counts['R4', str(r4)] += 1
    out = dict(method=__doc__.strip().split('\n\n')[0].replace('\n', ' '), replicates=REPLICATES, bootstrap_resamples=RESAMPLES,
               seed=SEED, development_classes=dict(Counter(p['cls'] for p in per.values())),
               fractions={k: {v: counts[kk, v] / REPLICATES for (kk, v) in sorted(counts) if kk == k} for k in ['verdict', 'R1', 'R2', 'R4']})
    path = STUDY / 'results' / candidate / 'development/operating-characteristics.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['fractions'], indent=1))


if __name__ == '__main__':
    main()
