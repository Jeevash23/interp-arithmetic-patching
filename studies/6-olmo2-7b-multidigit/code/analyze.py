"""Prespecified outcomes and decision rules of Study 4 (PROTOCOL.md sections 10 to 12) from the saved journals.

  python -B code/analyze.py <candidate> development   # development estimates (they set PREDICTIONS.md)
  python -B code/analyze.py <candidate> test          # the frozen test
  python -B code/analyze.py <candidate> summary       # results/summary.json

Reads only saved records; NumPy and the standard library.
"""
import json
import sys
from collections import Counter
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
from stats import proportion, paired_difference  # noqa: E402

CLASSES = ['unchanged', 'copy', 'counterfactual', 'splice', 'intermediate_splice', 'reverse_splice', 'partial', 'invalid', 'other']
H2_CLASSES = ('splice', 'intermediate_splice', 'partial')
CONDITIONS = ['donor', 'unpatched', 'same_example', 'isotropic_delta', 'random_source', 'same_z_decomposition',
              'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value', 'text_first_answer_digit']


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def calc(op, a, b):
    return {'add': a + b, 'sub': a - b, 'mul': a * b}[op]


def informative(row):
    """A leading-digit splice is a distinct prediction (PROTOCOL.md section 10, amendment 1)."""
    yd, yr, cf = row['donor']['y'], row['recipient']['y'], row['predictions']['counterfactual']
    splice, reverse = int(str(yd)[0] + str(yr)[1:]), int(str(yr)[0] + str(yd)[1:])
    return (len(str(yd)) == len(str(yr)) and str(yd)[0] != str(yr)[0]
            and splice not in (yd, yr, cf) and reverse not in (yd, yr, cf))


def classify(answer, row, unpatched_answer, donor_run_answer):
    """The answer classes of PROTOCOL.md section 10, taken in order; (s), (r), (m) on informative pairs only."""
    yd, yr = row['donor']['y'], row['recipient']['y']
    splice = int(str(yd)[0] + str(yr)[1:])
    reverse = int(str(yr)[0] + str(yd)[1:])
    if answer is None:
        return 'invalid'
    if answer == unpatched_answer:
        return 'unchanged'
    if answer == donor_run_answer:
        return 'copy'
    if answer == row['predictions']['counterfactual']:
        return 'counterfactual'
    if informative(row):
        if answer == splice:
            return 'splice'
        if answer == row['digit_alternatives']['intermediate_leading_digit_splice']:
            return 'intermediate_splice'
        if answer == reverse:
            return 'reverse_splice'
        if str(answer)[0] == str(yd)[0] and len(str(answer)) == len(str(yd)):
            return 'partial'
    return 'other'


def load(candidate, split):
    res = STUDY / 'results' / candidate / split
    rows = rd(STUDY / 'data' / f'{split}.jsonl')
    base = {r['key']: r for r in rd(res / 'baselines.jsonl')}
    controls = rd(res / 'controls.jsonl')
    text = rd(res / 'text-controls.jsonl')
    sweep = rd(res / 'sweep.jsonl')
    sites = [(s['layer'], s['role']) for s in json.loads((res / 'screen-summary.json').read_text())['selected']]
    competence = json.loads((res / 'competence.json').read_text())
    return rows, base, controls, text, sweep, sites, competence


def answer_prefix_outcomes(rows, base, by, actual, own, groups, tok_single_digit=None):
    """Outcomes at one answer-prefix site for the donor write (by: (id, condition) -> record)."""
    flags = {k: [] for k in ['unchanged', 'copy', 'copy_correct_donor', 'counterfactual', 'splice', 'intermediate_splice',
                             'reverse_splice', 'partial', 'invalid', 'other', 'splice_or_partial', 'changed', 'first_token_follows_donor',
                             'first_token_single_digit', 'equals_first_digit_text_edit', 'equals_donor_token_text_edit',
                             'equals_own_first_token_text_edit', 'identical_changed_output_same_leading',
                             'same_leading_identical_output', 'noise_first_token_follows_donor']}
    classes = []
    for row in rows:
        i = row['id']
        d, t, sl, noise = by[i, 'donor'], by[i, 'text_first_answer_digit'], by[i, 'same_leading_different_value'], by[i, 'isotropic_delta']
        rec = base[i + '/recipient_full']; donor_run = base[i + '/donor_full']
        c = classify(d['answer'], row, rec['answer'], donor_run['answer'])
        classes.append(c)
        for k in CLASSES:
            flags[k].append(c == k)
        flags['copy_correct_donor'].append(d['answer'] == row['donor']['y'])
        flags['splice_or_partial'].append(c in H2_CLASSES if informative(row) else None)
        flags['changed'].append(c != 'unchanged')
        flags['first_token_follows_donor'].append(d['output_ids'][0] == donor_run['output_ids'][0])
        flags['first_token_single_digit'].append(tok_single_digit(d['output_ids'][0]) if tok_single_digit else None)
        flags['equals_first_digit_text_edit'].append(d['output_ids'] == t['output_ids'])
        flags['equals_donor_token_text_edit'].append(d['output_ids'] == actual[i]['output_ids'])
        flags['equals_own_first_token_text_edit'].append(d['output_ids'] == own[i]['output_ids'])
        flags['identical_changed_output_same_leading'].append(d['output_ids'] == sl['output_ids'] and d['answer'] != rec['answer'])
        flags['same_leading_identical_output'].append(d['output_ids'] == sl['output_ids'])
        flags['noise_first_token_follows_donor'].append(noise['output_ids'][0] == donor_run['output_ids'][0])
    out = {k: proportion([x for x in v if x is not None], [g for x, g in zip(v, groups) if x is not None])
           for k, v in flags.items() if any(x is not None for x in v)}
    out['class_counts'] = dict(Counter(classes))
    out['changed_class_counts'] = dict(Counter(c for c in classes if c != 'unchanged'))
    inf = [j for j, row in enumerate(rows) if informative(row)]
    out['informative_pairs'] = dict(n=len(inf), groups=len({groups[j] for j in inf}))
    for k in ['splice', 'intermediate_splice', 'reverse_splice', 'partial']:
        out[k + '_informative'] = proportion([flags[k][j] for j in inf], [groups[j] for j in inf])
    out['splice_minus_reverse'] = paired_difference([flags['splice'][j] for j in inf], [flags['reverse_splice'][j] for j in inf],
                                                    [groups[j] for j in inf])
    out['first_token_follows_donor_minus_noise'] = paired_difference(flags['first_token_follows_donor'],
                                                                     flags['noise_first_token_follows_donor'], groups)
    out['by_operation'] = {op: dict(Counter(c for c, row in zip(classes, rows) if row['recipient']['remaining'] == op))
                           for op in ['add', 'sub', 'mul']}
    out['_classes'] = classes
    return out


def operand_outcomes(rows, base, by, which, groups):
    flags, gr, raw, unchanged, cf = [], [], [], [], []
    for row, g in zip(rows, groups):
        i, d, r = row['id'], row['donor'], row['recipient']
        f = lambda z: calc(r['remaining'], z, r['k'])
        pred = f(calc(r['first'], d['a'], r['b'])) if which == 'first' else f(calc(r['first'], r['a'], d['b']))
        a = by[i, 'donor']['answer']; rec = base[i + '/recipient_full']['answer']
        raw.append(a == pred); unchanged.append(a == rec); cf.append(a == row['predictions']['counterfactual'])
        if pred != rec:
            flags.append(a == pred); gr.append(g)
    return dict(operand_substitution=proportion(flags, gr), operand_substitution_raw=proportion(raw, groups),
                unchanged=proportion(unchanged, groups), complete_counterfactual=proportion(cf, groups))


def decisions(ap, sweep_blocks, operand):
    out = {}
    changed = ap['changed']
    out['R0'] = dict(holds=changed['rate'] >= 0.20, changed_rate=changed['rate'])
    sm = ap['splice_or_partial']
    r1 = 'H1' if sm['upper'] < 0.15 else 'H2' if (sm['rate'] >= 0.15 and sm['lower'] > 0.05) else 'inconclusive'
    out['R1'] = dict(favours=r1, rate=sm['rate'], lower=sm['lower'], upper=sm['upper'])
    i = ap['identical_changed_output_same_leading']
    r2 = 'H1' if i['upper'] < 0.20 else 'H2' if (i['rate'] >= 0.30 and i['lower'] > 0.10) else 'inconclusive'
    out['R2'] = dict(favours=r2, rate=i['rate'], lower=i['lower'], upper=i['upper'])
    a = ap['counterfactual']
    out['R3'] = dict(applies=a['rate'] >= 0.20, rate=a['rate'], lower=a['lower'], upper=a['upper'])
    cc = ap['changed_class_counts']; copies = cc.get('copy', 0)
    most = all(copies >= n for k, n in cc.items() if k != 'copy')
    out['R4'] = dict(holds=bool(copies) and most and ap['copy']['lower'] > 0.10, copy_rate=ap['copy']['rate'],
                     copy_lower=ap['copy']['lower'], copies_at_least_each_other_class=most and bool(copies),
                     changed_class_counts=cc)
    if out['R3']['applies']:
        verdict = 'complete transfer (R3): neither account'
    elif out['R0']['holds'] and out['R4']['holds'] and r1 == 'H1' and r2 == 'H1':
        verdict = 'H1 supported'
    elif out['R0']['holds'] and 'H2' in (r1, r2) and 'H1' not in (r1, r2):
        verdict = 'H2 supported'
    elif 'H1' in (r1, r2) and 'H2' in (r1, r2):
        verdict = 'inconclusive (mixed: R1 and R2 point in opposite directions)'
    else:
        reasons = []
        if not out['R0']['holds']:
            reasons.append('R0 fails: the write changes the answer in fewer than 20% of pairs')
        if r1 == 'inconclusive':
            reasons.append('R1 inconclusive')
        if r2 == 'inconclusive':
            reasons.append('R2 inconclusive')
        if r1 == 'H1' and r2 == 'H1' and not out['R4']['holds']:
            reasons.append('R4 fails: copies are not the leading changed class with a lower bound above 0.10')
        verdict = 'inconclusive (' + '; '.join(reasons) + ')'
    out['verdict'] = verdict
    sr = ap['splice_minus_reverse']
    out['splice_note'] = (r1 == 'H2' and sr['cluster95'][0] <= 0)
    # Layer sweep (secondary).
    h2_blocks = [b for b, x in sweep_blocks.items() if x['splice_or_partial']['rate'] >= 0.25 and x['splice_or_partial']['lower'] > 0.10]
    active = [b for b, x in sweep_blocks.items() if x['changed']['rate'] >= 0.20]
    h1_blocks = [b for b in active if sweep_blocks[b]['splice_or_partial']['upper'] < 0.15]
    out['sweep'] = dict(representational_support=bool(h2_blocks), h2_blocks=h2_blocks, active_blocks=active,
                        h1_blocks=h1_blocks, active_blocks_not_h1=[b for b in active if b not in h1_blocks])
    out['operand_control'] = {k: dict(lower=v['operand_substitution']['lower'], rate=v['operand_substitution']['rate'],
                                      k=v['operand_substitution']['k'], n=v['operand_substitution']['n'],
                                      success=v['operand_substitution']['n'] > 0 and v['operand_substitution']['lower'] >= 0.25)
                              for k, v in operand.items()}
    out['operand_control_failed_both'] = not any(v['success'] for v in out['operand_control'].values())
    return out


def analyse(candidate, split):
    rows, base, controls, text, sweep, sites, competence = load(candidate, split)
    groups = [r['group'] for r in rows]
    by_site = {}
    for r in controls:
        by_site.setdefault((r['layer'], r['role']), {})[r['id'], r['condition']] = r
    actual = {r['id']: r for r in text if r['condition'] == 'text_first_generated_donor_digit'}
    own_by_site = {}
    for r in text:
        if r['condition'] == 'text_activation_first_token':
            own_by_site.setdefault((r['layer'], r['role']), {})[r['id']] = r
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(STUDY / 'tokenizers' / candidate, local_files_only=True)
        single = lambda t: tok.decode([t]).strip().isdigit() and len(tok.decode([t]).strip()) == 1
    except Exception:
        single = None
    site = {s[1]: s for s in sites}
    ap_site = site['prompt_end']
    ap = answer_prefix_outcomes(rows, base, by_site[ap_site], actual, own_by_site[ap_site], groups, single)
    classes = ap.pop('_classes')
    operand = {'first_operand': operand_outcomes(rows, base, by_site[site['a_end']], 'first', groups),
               'second_operand': operand_outcomes(rows, base, by_site[site['b_end']], 'second', groups)}
    fs = by_site[site['first_stage_end']]
    first_step_end = dict(unchanged=proportion([fs[r['id'], 'donor']['answer'] == base[r['id'] + '/recipient_full']['answer'] for r in rows], groups),
                          complete_counterfactual=proportion([fs[r['id'], 'donor']['counterfactual_success'] for r in rows], groups))
    # Layer sweep.
    sw = {}
    for r in sweep:
        sw.setdefault(r['layer'], {})[r['id'], r['condition']] = r
    blocks = {}
    for layer in sorted(sw):
        cls, sl_same = [], []
        for row in rows:
            i = row['id']; rec = base[i + '/recipient_full']['answer']; donor_run = base[i + '/donor_full']['answer']
            d, s = sw[layer][i, 'donor'], sw[layer][i, 'same_leading_different_value']
            cls.append(classify(d['answer'], row, rec, donor_run))
            sl_same.append(d['output_ids'] == s['output_ids'] and d['answer'] != rec)
        inf = [j for j, row in enumerate(rows) if informative(row)]
        blocks[layer] = dict(changed=proportion([c != 'unchanged' for c in cls], groups),
                             splice_or_partial=proportion([cls[j] in H2_CLASSES for j in inf], [groups[j] for j in inf]),
                             copy=proportion([c == 'copy' for c in cls], groups),
                             counterfactual=proportion([c == 'counterfactual' for c in cls], groups),
                             identical_changed_output_same_leading=proportion(sl_same, groups),
                             class_counts=dict(Counter(cls)))
    # Consistency: the sweep at the selected block repeats the control records.
    layer = ap_site[0]
    sweep_matches_controls = sum(sw[layer][r['id'], 'donor']['output_ids'] == by_site[ap_site][r['id'], 'donor']['output_ids']
                                 for r in rows) if layer in sw else None
    conditions = {}
    for s, by in by_site.items():
        conditions[f'{s[0]}/{s[1]}'] = {c: dict(n=len(rows), counterfactual=sum(by[r['id'], c]['counterfactual_success'] for r in rows),
                                                unchanged=sum(by[r['id'], c]['answer'] == base[r['id'] + '/recipient_full']['answer'] for r in rows),
                                                invalid=sum(by[r['id'], c]['answer'] is None for r in rows))
                                        for c in CONDITIONS}
    identity = {f'{s[0]}/{s[1]}': dict(unpatched=sum(by[r['id'], 'unpatched']['output_ids'] == base[r['id'] + '/recipient_full']['output_ids'] for r in rows),
                                       self_write=sum(by[r['id'], 'same_example']['output_ids'] == base[r['id'] + '/recipient_full']['output_ids'] for r in rows),
                                       n=len(rows)) for s, by in by_site.items()}
    result = dict(candidate=candidate, split=split, pairs=len(rows), donor_groups=len(set(groups)),
                  sites={k: dict(layer=v[0], role=v[1]) for k, v in site.items()}, competence=competence,
                  answer_prefix=ap, answer_prefix_classes={r['id']: c for r, c in zip(rows, classes)},
                  operand_sites=operand, first_step_end=first_step_end, sweep=blocks,
                  sweep_matches_controls_at_selected_block=sweep_matches_controls, conditions=conditions,
                  identity_checks=identity)
    result['decisions'] = decisions(ap, blocks, operand)
    out = STUDY / 'results' / candidate / split / 'analysis.json'
    out.write_text(json.dumps(result, indent=1) + '\n')
    return result


def summary(candidate):
    data = {}
    for split in ['development', 'test']:
        p = STUDY / 'results' / candidate / split / 'analysis.json'
        if p.exists():
            data[split] = json.loads(p.read_text())
    out = dict(study='4-multidigit-tokens', candidate=candidate,
               candidates=json.loads((STUDY / 'results/candidates.json').read_text()),
               tokenizer_check={k: v['pass_check'] for k, v in json.loads((STUDY / 'results/tokenizer-check.json').read_text())['candidates'].items()},
               gate=json.loads((STUDY / 'results' / candidate / 'calibration/gate.json').read_text()),
               splits={s: {k: v for k, v in d.items() if k not in ('answer_prefix_classes', 'conditions')} for s, d in data.items()})
    (STUDY / 'results/summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print('summary written')


if __name__ == '__main__':
    cand, which = sys.argv[1], sys.argv[2]
    if which == 'summary':
        summary(cand)
    else:
        res = analyse(cand, which)
        print(json.dumps(res['decisions'], indent=1))
