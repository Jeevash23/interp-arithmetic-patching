"""Written after the test run (17 October 2026). Two parts, written to results/posthoc.json:

1. The descriptive expectations of PREDICTIONS.md (items 1 to 5), stated before the test and evaluated here on the
   test records. They enter no decision rule.
2. Post hoc checks, chosen after the test outcomes were seen: whether the span sources' "own counterfactual" answers
   are pairs where that counterfactual equals the recipient's unchanged answer; the noise write that produced the
   counterfactual; and the intermediate leading-digit splice by the recipient's second operation.

Reads results/test-analysis.json and the raw records through code/analyze.py's loader (unchanged since the freeze).

  python -B code/posthoc.py
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
import analyze as A  # noqa: E402  frozen analysis code, used read-only


def main():
    s = A.Split('test', ['test'])
    t = json.loads((STUDY / 'results/test-analysis.json').read_text())
    sw, sites = t['sweeps'], t['sites']
    sel = json.loads((STUDY / 'results/development-selection.json').read_text())
    n = t['pairs']

    post = sw['span']['post_operand']
    full = sw['span']['first_step']
    fse = sw['direction']['first_stage_end/1']
    pe = {r['layer']: r for r in sw['direction']['prompt_end/1']}
    band = [pe[L] for L in range(25, 31)]
    exp = {
        '1_post_operand_unchanged_90': dict(min_unchanged=min(r['unchanged'] for r in post), n=n,
                                            holds=all(r['unchanged'] >= 0.9 * n for r in post)),
        '2a_first_step_first_op_70_blocks_0_11': dict(min_first_op=min(r['first_op'] for r in full if r['layer'] <= 11), n=n,
                                                      holds=all(r['first_op'] >= 0.7 * n for r in full if r['layer'] <= 11)),
        '2b_first_step_first_op_10_from_18': dict(max_first_op=max(r['first_op'] for r in full if r['layer'] >= 18), n=n,
                                                  holds=all(r['first_op'] <= 0.1 * n for r in full if r['layer'] >= 18)),
        '2c_first_step_unchanged_90_from_18': dict(min_unchanged=min(r['unchanged'] for r in full if r['layer'] >= 18), n=n,
                                                   first_block_meeting_it=min(r['layer'] for r in full if r['layer'] >= 18 and
                                                                              all(x['unchanged'] >= 0.9 * n for x in full if x['layer'] >= r['layer'])),
                                                   holds=all(r['unchanged'] >= 0.9 * n for r in full if r['layer'] >= 18)),
        '3_first_stage_end_unchanged_90': dict(min_unchanged=min(r['unchanged'] for r in fse), n=n,
                                               holds=all(r['unchanged'] >= 0.9 * n for r in fse)),
        '4_answer_prefix_blocks_25_30': dict(changed=[n - r['unchanged'] for r in band], z_splice=[r['z_splice'] for r in band],
                                             cf=[r['cf'] for r in band], answer_splice=[r['answer_splice'] for r in band], n=n,
                                             holds=all(n - r['unchanged'] > n / 2 and r['z_splice'] > r['cf'] and r['z_splice'] > r['answer_splice']
                                                       for r in band)),
    }
    pe_site = sites['direction']['prompt_end']
    exp['5_gurnee_flips_exceed_generated'] = dict(flips=pe_site['gurnee']['flips']['k'], generated_cf=pe_site['donor']['k'], n=n,
                                                  holds=pe_site['gurnee']['flips']['k'] > pe_site['donor']['k'])

    coincident = {}
    for x in sel['span_sites']:
        L, span = x['layer'], x['span']
        for src in ['same_z_decomposition', 'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value',
                    'random_source']:
            hits = [i for i in s.ids if s.index[f'{i}/span/{L}/{span}/{src}']['answer'] == s.target(s.by_id[i], src)]
            same = [i for i in hits if s.target(s.by_id[i], src) == s.unpatched[i]]
            coincident[f'{span}/{src}'] = dict(own_counterfactual=len(hits), equal_to_recipient_answer=len(same))
    d = [d for d in sel['direction_sites_test'] if d['role'] == 'prompt_end'][0]
    noise_hits = [dict(id=i, answer=s.index[f"{i}/direction-noise/{d['layer']}/prompt_end/{d['alpha']}"]['answer'], unpatched=s.unpatched[i])
                  for i in s.ids if s.index[f"{i}/direction-noise/{d['layer']}/prompt_end/{d['alpha']}"]['answer'] == s.target(s.by_id[i], 'noise')]
    by_op = {}
    for op in ['add', 'sub', 'mul']:
        ids = [i for i in s.ids if s.by_id[i]['recipient']['remaining'] == op]
        by_op[op] = dict(n=len(ids), z_splice_by_block={L: sum(s.classes(s.index[f'{i}/direction/{L}/prompt_end/1/donor'])['z_splice'] for i in ids)
                                                         for L in range(23, 32)})
    r4 = []                                       # the blocks that kept R4 from finding no block (upper bound >= 0.20)
    for role in ('first_stage_end', 'prompt_end'):
        for L in range(36):
            p = s.rate(A.dkey(L, role, 1))
            if p and p['upper'] >= 0.2:
                r4.append(dict(kind='direction', where=role, layer=L, k=p['k'], n=p['n'], lower=p['lower'], upper=p['upper']))
    for span in ('post_operand', 'first_step'):
        for L in range(36):
            p = s.rate(A.skey(L, span))
            if p and p['upper'] >= 0.2:
                r4.append(dict(kind='span', where=span, layer=L, k=p['k'], n=p['n'], lower=p['lower'], upper=p['upper']))
    out = dict(note=__doc__.strip().splitlines()[0], descriptive_expectations=exp, r4_blocks_with_upper_at_least_020=r4,
               posthoc=dict(coincident_source_matches=coincident, answer_prefix_noise_counterfactuals=noise_hits,
                            answer_prefix_z_splice_by_operation=by_op))
    (STUDY / 'results/posthoc.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
