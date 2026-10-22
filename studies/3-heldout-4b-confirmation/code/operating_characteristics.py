"""How often each PREDICTIONS.md rule would give each verdict if the test pairs behaved like the development pairs.

Draws 24 donor groups with replacement from the 12 development groups (each group keeps its three pairs and their
saved outcomes), applies the decision rules with the study's estimators (exact interval and a donor-cluster bootstrap,
here with 2,000 resamples to keep the run short), and repeats 400 times. Development data only; written and run
before the freeze.

  python -B code/operating_characteristics.py      # writes results/operating-characteristics.json
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

REPLICATES, RESAMPLES, SEED = 400, 2000, 20261010


def site_flags(rows, base, by):
    out = {}
    for row in rows:
        i, r, dn = row['id'], row['recipient'], row['donor']
        g = lambda z: A.calc(r['remaining'], z, r['k'])
        d, t = by[i, 'donor'], by[i, 'text_first_answer_digit']
        rec = base[i + '/recipient_full']
        p1, p2 = g(A.calc(r['first'], dn['a'], r['b'])), g(A.calc(r['first'], r['a'], dn['b']))
        if d['output_ids'][0] == t['forced_output_ids'][0]:
            own = d['output_ids'] == t['output_ids']
        elif d['output_ids'][0] == rec['output_ids'][0]:
            own = d['output_ids'] == by[i, 'unpatched']['output_ids']
        else:
            own = None  # not derivable from the development records
        donor_first = base[i + '/donor_full']['output_ids'][0]
        out[i] = dict(a=d['answer'] == g(dn['z']), b=d['output_ids'][0] == donor_first, c=d['output_ids'] == t['output_ids'],
                      cp=own, d1=(d['answer'] == p1) if p1 != rec['answer'] else None,
                      d2=(d['answer'] == p2) if p2 != rec['answer'] else None, e=d['answer'] == rec['answer'],
                      sl=d['output_ids'] == by[i, 'same_leading_different_value']['output_ids'],
                      noise=by[i, 'isotropic_delta']['output_ids'][0] == donor_first)
    return out


def prop(values, groups):
    keep = [(v, g) for v, g in zip(values, groups) if v is not None]
    return S.proportion([v for v, _ in keep], [g for _, g in keep])


def main():
    S.RESAMPLES = RESAMPLES
    rows, base, controls, actual, first_token, competence, sources = A.load('development')
    by_site = {}
    for r in controls:
        by_site.setdefault((r['layer'], r['role']), {})[r['id'], r['condition']] = r
    flags = {s: site_flags(rows, base, by_site[s]) for s in A.SITES}
    groups = sorted({r['group'] for r in rows})
    members = {g: [r['id'] for r in rows if r['group'] == g] for g in groups}
    rng = np.random.default_rng(SEED)
    counts = Counter()
    s23 = (23, 'prompt_end')
    for _ in range(REPLICATES):
        pick = rng.choice(groups, size=24, replace=True)

        def col(site, key):
            v, gr = [], []
            for j, g in enumerate(pick):
                for i in members[g]:
                    v.append(flags[site][i][key]); gr.append(j)
            return v, gr
        a = [prop(*col(s, 'a')) for s in A.SITES]
        counts['P1', 'supported' if all(p['upper'] < .2 for p in a) else
               'refuted' if any(p['rate'] >= .2 for p in a) else 'inconclusive'] += 1
        v, gr = col(s23, 'b'); p = prop(v, gr); dn = S.paired_difference(v, col(s23, 'noise')[0], gr)
        counts['P2', 'supported' if p['lower'] > .5 and dn['cluster95'][0] > 0 else
               'refuted' if p['upper'] < .5 or dn['cluster95'][1] <= 0 else 'inconclusive'] += 1
        for key, name in [('c', 'P3'), ('cp', 'P3s')]:
            p = prop(*col(s23, key))
            counts[name, 'supported' if p['rate'] >= .8 and p['lower'] >= .65 else
                   'refuted' if p['upper'] < .8 else 'inconclusive'] += 1
        verdicts = []
        for site, key in [((20, 'a_end'), 'd1'), ((9, 'a_end'), 'd1'), ((10, 'a_end'), 'd1'), ((14, 'b_end'), 'd2')]:
            p = prop(*col(site, key))
            verdicts.append('supported' if p['lower'] >= .25 else 'refuted' if p['upper'] < .25 else 'inconclusive')
        counts['P4', 'supported' if all(x == 'supported' for x in verdicts) else
               'refuted' if all(x == 'refuted' for x in verdicts) else 'supported at fewer than 4 sites'] += 1
        p = prop(*col((20, 'first_stage_end'), 'e'))
        counts['P5', 'supported' if p['lower'] >= .8 else 'refuted' if p['upper'] < .8 else 'inconclusive'] += 1
        p = prop(*col(s23, 'sl'))
        counts['P6', 'supported' if p['lower'] > .5 else 'refuted' if p['upper'] < .5 else 'inconclusive'] += 1
    result = dict(method=__doc__.strip().split('\n\n')[1].replace('\n', ' '), replicates=REPLICATES,
                  bootstrap_resamples=RESAMPLES, seed=SEED, note='P3s uses the 28 development pairs where it is derivable.',
                  verdict_fractions={rule: {v: counts[r, v] / REPLICATES for (r, v) in sorted(counts) if r == rule}
                                     for rule in ['P1', 'P2', 'P3', 'P3s', 'P4', 'P5', 'P6']})
    (STUDY / 'results/operating-characteristics.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['verdict_fractions'], indent=1))


if __name__ == '__main__':
    main()
