"""Prespecified outcomes of Study 3 from the saved journals, with intervals and the PREDICTIONS.md decision rules.

  python -B code/analyze.py development   # recomputes the development reference values from the frozen AI-02 journals
  python -B code/analyze.py box           # primary test run (reserve numbers in the development wording)
  python -B code/analyze.py machine       # secondary test run (reserve in its own declared wording)
  python -B code/analyze.py summary       # writes results/summary.json from the saved analyses

Reads only saved records; needs NumPy (cluster bootstrap) and the standard library.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
HV = PROJECT / '06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer'
sys.path.insert(0, str(STUDY / 'code'))
from stats import proportion, paired_difference  # noqa: E402

SITES = [(23, 'prompt_end'), (20, 'a_end'), (9, 'a_end'), (10, 'a_end'), (14, 'b_end'), (20, 'first_stage_end')]
SITE_KEY = {(23, 'prompt_end'): 'block23_answer_prefix', (20, 'a_end'): 'block20_first_operand',
            (9, 'a_end'): 'block9_first_operand', (10, 'a_end'): 'block10_first_operand',
            (14, 'b_end'): 'block14_second_operand', (20, 'first_stage_end'): 'block20_first_step_end'}
# Outcome (d): at each operand site, the operand that site holds; estimated on the informative pairs, where the
# substitution prediction differs from the recipient's own unpatched answer (PROTOCOL.md section 8).
OPERAND_SITES = {(20, 'a_end'): 'first_operand_substitution_informative', (9, 'a_end'): 'first_operand_substitution_informative',
                 (10, 'a_end'): 'first_operand_substitution_informative', (14, 'b_end'): 'second_operand_substitution_informative'}
# Outcomes also broken down by the recipient's remaining operation.
BY_OPERATION = ['complete_counterfactual', 'first_token_follows_donor', 'equals_first_digit_text_edit',
                'equals_own_first_token_text_edit', 'first_operand_substitution_informative',
                'second_operand_substitution_informative', 'recipient_answer_unchanged']
CONDITIONS = ['donor', 'unpatched', 'same_example', 'isotropic_delta', 'random_source', 'same_z_decomposition',
              'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value', 'text_first_answer_digit']
SOURCE_CONDITIONS = ['donor', 'random_source', 'same_z_decomposition', 'same_z_different_answer',
                     'different_z_same_answer', 'same_leading_different_value']


def calc(op, a, b):
    return {'add': a + b, 'sub': a - b, 'mul': a * b}[op]


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def load(run):
    if run == 'development':
        v3 = HV / 'competence-v3'
        rows = rd(v3 / 'development-with-source-controls.jsonl')
        base = {r['key']: r for r in rd(v3 / 'results/qwen4b-fewshot/baselines.jsonl')}
        controls = rd(v3 / 'results/qwen4b-fewshot/controls.jsonl') + rd(v3 / 'results/eligible-sites/controls.jsonl')
        # The actual-donor-token text control is text only and the same at every site; one record per pair suffices.
        actual = {r['id']: r for r in rd(v3 / 'results/qwen4b-fewshot/actual-donor-digit-control.jsonl')}
        first_token = None
        competence = json.loads((v3 / 'results/qwen4b-fewshot/competence.json').read_text())
        sources = {r['key']: r for r in rd(v3 / 'results/qwen4b-fewshot/source-baselines.jsonl')}
    else:
        res = STUDY / 'results' / run
        rows = rd(STUDY / 'data' / f'test-{run}-with-source-controls.jsonl')
        base = {r['key']: r for r in rd(res / 'baselines.jsonl')}
        controls = rd(res / 'controls.jsonl')
        text = rd(res / 'text-controls.jsonl')
        actual = {r['id']: r for r in text if r['condition'] == 'text_first_generated_donor_digit'}
        first_token = {(r['id'], r['layer'], r['role']): r for r in text if r['condition'] == 'text_activation_first_token'}
        competence = json.loads((res / 'competence.json').read_text())
        sources = {r['key']: r for r in rd(res / 'source-baselines.jsonl')}
    return rows, base, controls, actual, first_token, competence, sources


def site_outcomes(rows, base, by, actual, first_token, site, competent):
    layer, role = site
    ids = [r['id'] for r in rows]
    groups = [r['group'] for r in rows]
    flags = {k: [] for k in [
        'complete_counterfactual', 'first_token_follows_donor', 'first_token_is_correct_donor_digit',
        'equals_first_digit_text_edit', 'equals_actual_donor_token_text_edit', 'equals_own_first_token_text_edit',
        'first_operand_substitution', 'second_operand_substitution', 'both_operands_substitution',
        'first_operand_substitution_distinct', 'second_operand_substitution_distinct',
        'first_operand_substitution_informative', 'second_operand_substitution_informative',
        'recipient_answer_unchanged', 'recipient_correct_answer', 'donor_answer_copy', 'answer_leading_digit_splice',
        'answer_leading_digit_splice_distinct', 'intermediate_leading_digit_splice', 'invalid',
        'same_leading_source_identical_output', 'same_leading_joint_success', 'same_z_joint_success',
        'noise_first_token_follows_donor', 'unpatched_first_token_follows_donor', 'first_token_unchanged',
        'diverges_after_donor_first_token']}
    own_first_derivable = []
    for row in rows:
        i = row['id']; r = row['recipient']; dn = row['donor']
        g = lambda z: calc(r['remaining'], z, r['k'])
        d, t, un, sl = by[i, 'donor'], by[i, 'text_first_answer_digit'], by[i, 'unpatched'], by[i, 'same_leading_different_value']
        noise = by[i, 'isotropic_delta']
        donor_first = base[i + '/donor_full']['output_ids'][0]
        rec = base[i + '/recipient_full']
        a = d['answer']
        f = flags
        f['complete_counterfactual'].append(a == g(dn['z']))
        f['first_token_follows_donor'].append(d['output_ids'][0] == donor_first)
        f['first_token_is_correct_donor_digit'].append(d['output_ids'][0] == t['forced_output_ids'][0])
        f['equals_first_digit_text_edit'].append(d['output_ids'] == t['output_ids'])
        f['equals_actual_donor_token_text_edit'].append(d['output_ids'] == actual[i]['output_ids'])
        if first_token is not None:
            own = first_token[i, layer, role]
            assert own['forced_output_ids'] == d['output_ids'][:1]
            f['equals_own_first_token_text_edit'].append(d['output_ids'] == own['output_ids'])
            own_first_derivable.append(True)
        else:
            # Development only: derivable when the write's first token equals the forced donor digit
            # (that text edit) or the recipient's own first token (then the edit equals unpatched generation).
            if d['output_ids'][0] == t['forced_output_ids'][0]:
                f['equals_own_first_token_text_edit'].append(d['output_ids'] == t['output_ids']); own_first_derivable.append(True)
            elif d['output_ids'][0] == rec['output_ids'][0]:
                f['equals_own_first_token_text_edit'].append(d['output_ids'] == un['output_ids']); own_first_derivable.append(True)
            else:
                f['equals_own_first_token_text_edit'].append(None); own_first_derivable.append(False)
        f['first_operand_substitution'].append(a == g(calc(r['first'], dn['a'], r['b'])))
        f['second_operand_substitution'].append(a == g(calc(r['first'], r['a'], dn['b'])))
        f['both_operands_substitution'].append(a == g(calc(r['first'], dn['a'], dn['b'])))
        # Outcome (d) on the informative pairs: the substitution prediction differs from the recipient's own
        # unpatched answer (None marks an uninformative pair, which leaves the denominator). The '_distinct'
        # versions count the same events over all pairs, uninformative pairs counting as non-events.
        p1, p2 = g(calc(r['first'], dn['a'], r['b'])), g(calc(r['first'], r['a'], dn['b']))
        f['first_operand_substitution_distinct'].append(a == p1 and p1 != rec['answer'])
        f['second_operand_substitution_distinct'].append(a == p2 and p2 != rec['answer'])
        f['first_operand_substitution_informative'].append(a == p1 if p1 != rec['answer'] else None)
        f['second_operand_substitution_informative'].append(a == p2 if p2 != rec['answer'] else None)
        f['recipient_answer_unchanged'].append(a == rec['answer'])
        f['recipient_correct_answer'].append(a == r['y'])
        f['donor_answer_copy'].append(a == dn['y'])
        splice = row['digit_alternatives']['donor_answer_leading_digit_splice']
        f['answer_leading_digit_splice'].append(a == splice)
        f['answer_leading_digit_splice_distinct'].append(a == splice and splice not in (r['y'], dn['y'], rec['answer']))
        f['intermediate_leading_digit_splice'].append(a == row['digit_alternatives']['intermediate_leading_digit_splice'])
        f['invalid'].append(a is None)
        f['same_leading_source_identical_output'].append(d['output_ids'] == sl['output_ids'])
        f['same_leading_joint_success'].append(d['counterfactual_success'] and sl['actual_source_success'])
        f['same_z_joint_success'].append(all(by[i, c]['counterfactual_success'] for c in ['donor', 'same_z_decomposition', 'same_z_different_answer']))
        f['noise_first_token_follows_donor'].append(noise['output_ids'][0] == donor_first)
        f['unpatched_first_token_follows_donor'].append(un['output_ids'][0] == donor_first)
        f['first_token_unchanged'].append(d['output_ids'][0] == rec['output_ids'][0])
        # Later divergence is judged against the text edit that forces the same first token (condition 11).
        f['diverges_after_donor_first_token'].append(d['output_ids'][0] == donor_first and d['output_ids'] != actual[i]['output_ids'])
    out = {}
    for name, values in flags.items():
        keep = [j for j, v in enumerate(values) if v is not None]
        out[name] = proportion([values[j] for j in keep], [groups[j] for j in keep])
        if len(keep) < len(values):
            out[name]['note'] = ('development: derivable pairs only' if name == 'equals_own_first_token_text_edit'
                                 else 'informative pairs only: the prediction differs from the recipient\'s unpatched answer')
    # Among pairs whose first token follows the donor, how many diverge from the first-digit text edit later.
    follow = [j for j, x in enumerate(flags['first_token_follows_donor']) if x]
    out['diverges_given_first_token_follows'] = proportion(
        [flags['diverges_after_donor_first_token'][j] for j in follow], [groups[j] for j in follow])
    out['first_token_follows_donor_minus_noise'] = paired_difference(
        flags['first_token_follows_donor'], flags['noise_first_token_follows_donor'], groups)
    out['complete_counterfactual_by_operation'] = {
        op: sum(x for x, row in zip(flags['complete_counterfactual'], rows) if row['recipient']['remaining'] == op)
        for op in ['add', 'sub', 'mul']}
    out['by_operation'] = {
        name: {op: dict(k=sum(bool(x) for x, row in zip(flags[name], rows) if row['recipient']['remaining'] == op and x is not None),
                        n=sum(1 for x, row in zip(flags[name], rows) if row['recipient']['remaining'] == op and x is not None))
               for op in ['add', 'sub', 'mul']}
        for name in BY_OPERATION}
    comp = [j for j, i in enumerate(ids) if i in competent]
    out['triple_competent_subset'] = {
        name: proportion([flags[name][j] for j in comp if flags[name][j] is not None],
                         [groups[j] for j in comp if flags[name][j] is not None])
        for name in ['complete_counterfactual', 'first_token_follows_donor', 'equals_first_digit_text_edit',
                     'equals_own_first_token_text_edit', 'first_operand_substitution_informative',
                     'second_operand_substitution_informative', 'recipient_answer_unchanged',
                     'same_leading_source_identical_output']}
    conditions = {}
    for c in CONDITIONS:
        recs = [by[i, c] for i in ids]
        conditions[c] = dict(
            n=len(recs),
            counterfactual=sum(x['counterfactual_success'] for x in recs),
            actual_source_counterfactual=sum(x['actual_source_success'] for x in recs) if c in SOURCE_CONDITIONS else None,
            unchanged=sum(x['answer'] == base[x['id'] + '/recipient_full']['answer'] for x in recs),
            invalid=sum(x['answer'] is None for x in recs),
            identical_to_donor_write=sum(x['output_ids'] == by[x['id'], 'donor']['output_ids'] for x in recs))
    out['conditions'] = conditions
    out['identity_checks'] = dict(
        unpatched_equals_baseline=sum(by[i, 'unpatched']['output_ids'] == base[i + '/recipient_full']['output_ids'] for i in ids),
        self_write_equals_baseline=sum(by[i, 'same_example']['output_ids'] == base[i + '/recipient_full']['output_ids'] for i in ids),
        n=len(ids))
    return out


RULE_TEXT = {
    'P1': 'Complete counterfactual transfer is rare at every tested site: supported if the upper 95% bound '
          '(larger of exact and donor-cluster) is below 0.20 at all six sites; refuted if any site has a rate of '
          '0.20 or more; otherwise inconclusive.',
    'P2': 'At the answer prefix (block 23) the first answer token follows the donor in most pairs, more often than '
          'under norm-matched noise: supported if the lower 95% bound (smaller of exact and donor-cluster) exceeds '
          '0.50 and the donor-cluster lower bound of the donor-minus-noise difference exceeds 0; refuted if the upper '
          'bound is below 0.50 or the upper bound of that difference is at most 0; otherwise inconclusive.',
    'P3': 'The 0.6B one-token account explains 4B answer-prefix writes (activation output identical to the '
          'first-digit text-edit output): supported if the rate is at least 0.80 and the lower bound at least '
          '0.65; refuted if the upper bound is below 0.80; otherwise inconclusive.',
    'P3s': 'Secondary: the output equals the text edit that forces the write\'s own first token; same thresholds as P3.',
    'P4': 'Writes at operand positions act as operand substitution (output equals the recipient\'s program run on '
          'the donor\'s operand at that position, on the pairs where that prediction differs from the recipient\'s '
          'unpatched answer): at each of the four operand sites, supported if the lower bound is at least 0.25 and '
          'refuted if the upper bound is below 0.25; the claim is supported if all four sites are supported and '
          'refuted if all four are refuted.',
    'P5': 'A write at the end of the first step (block 20) leaves the recipient\'s answer unchanged: supported if '
          'the lower bound is at least 0.80; refuted if the upper bound is below 0.80.',
    'P6': 'At the answer prefix, a source that shares the donor\'s leading digit but differs in value gives the '
          'identical output in most pairs: supported if the lower bound exceeds 0.50; refuted if the upper bound '
          'is below 0.50.',
}


def decisions(sites):
    s23 = sites['block23_answer_prefix']
    out = {}
    p1_sites = {k: v['complete_counterfactual'] for k, v in sites.items()}
    if all(v['upper'] < 0.20 for v in p1_sites.values()):
        p1 = 'supported'
    elif any(v['rate'] >= 0.20 for v in p1_sites.values()):
        p1 = 'refuted'
    else:
        p1 = 'inconclusive'
    out['P1'] = dict(verdict=p1, rates={k: v['rate'] for k, v in p1_sites.items()},
                     uppers={k: v['upper'] for k, v in p1_sites.items()})
    b = s23['first_token_follows_donor']
    dn = s23['first_token_follows_donor_minus_noise']
    out['P2'] = dict(verdict='supported' if b['lower'] > 0.5 and dn['cluster95'][0] > 0 else
                     'refuted' if b['upper'] < 0.5 or dn['cluster95'][1] <= 0 else 'inconclusive',
                     rate=b['rate'], lower=b['lower'], upper=b['upper'],
                     minus_noise=dn['difference'], minus_noise_cluster95=dn['cluster95'])
    for key, name in [('P3', 'equals_first_digit_text_edit'), ('P3s', 'equals_own_first_token_text_edit')]:
        c = s23[name]
        v = 'supported' if c['rate'] >= 0.80 and c['lower'] >= 0.65 else 'refuted' if c['upper'] < 0.80 else 'inconclusive'
        out[key] = dict(verdict=v, rate=c['rate'], lower=c['lower'], upper=c['upper'])
    p4 = {}
    for site, outcome in OPERAND_SITES.items():
        x = sites[SITE_KEY[site]][outcome]
        verdict = ('not estimable' if not x['n'] else 'supported' if x['lower'] >= 0.25 else
                   'refuted' if x['upper'] < 0.25 else 'inconclusive')
        p4[SITE_KEY[site]] = dict(outcome=outcome, k=x['k'], n=x['n'], rate=x['rate'], lower=x['lower'],
                                  upper=x['upper'], verdict=verdict)
    verdicts = [v['verdict'] for v in p4.values()]
    out['P4'] = dict(verdict='supported' if all(v == 'supported' for v in verdicts) else
                     'refuted' if all(v == 'refuted' for v in verdicts) else
                     f"supported at {verdicts.count('supported')} of 4 sites", sites=p4)
    e = sites['block20_first_step_end']['recipient_answer_unchanged']
    out['P5'] = dict(verdict='supported' if e['lower'] >= 0.80 else 'refuted' if e['upper'] < 0.80 else 'inconclusive',
                     rate=e['rate'], lower=e['lower'], upper=e['upper'])
    sl = s23['same_leading_source_identical_output']
    out['P6'] = dict(verdict='supported' if sl['lower'] > 0.5 else 'refuted' if sl['upper'] < 0.5 else 'inconclusive',
                     rate=sl['rate'], lower=sl['lower'], upper=sl['upper'],
                     joint_successes=s23['same_leading_joint_success']['k'])
    for k in out:
        out[k]['rule'] = RULE_TEXT[k]
    return out


def analyse(run):
    rows, base, controls, actual, first_token, competence, sources = load(run)
    by_site = {}
    for r in controls:
        by_site.setdefault((r['layer'], r['role']), {})[r['id'], r['condition']] = r
    competent = set(competence['triple_competent'])
    sites = {SITE_KEY[s]: site_outcomes(rows, base, by_site[s], actual, first_token, s, competent) for s in SITES}
    source_competence = {}
    for cond in ['same_z_decomposition', 'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value']:
        source_competence[cond] = {kind: dict(n=sum(1 for r in sources.values() if r['source_condition'] == cond and r['condition'] == kind),
                                              correct=sum(r['correct'] for r in sources.values() if r['source_condition'] == cond and r['condition'] == kind))
                                   for kind in ['full', 'first', 'target_remaining']}
    result = dict(run=run, pairs=len(rows), donor_groups=len({r['group'] for r in rows}),
                  wording=sorted({r['family'] for r in rows}), competence=competence,
                  source_competence=source_competence, sites=sites, decisions=decisions(sites),
                  control_records=len(controls))
    if run == 'development':
        result['note'] = ('Development reference recomputed from the frozen AI-02 journals (36 pairs, 12 donor groups, '
                          'box wording). The own-first-token text edit was not run in development; it is derivable '
                          'only where the write\'s first token equals the forced donor digit or the recipient\'s own first token.')
        out = STUDY / 'results/development-reference.json'
    else:
        out = STUDY / 'results' / run / 'analysis.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    return result


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f'{prefix}.{k}' if prefix else str(k), v, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value
    elif isinstance(value, list) and all(isinstance(x, (int, float)) or x is None for x in value):
        out[prefix] = value


def summary():
    data = {}
    for run in ['development', 'box', 'machine']:
        path = STUDY / ('results/development-reference.json' if run == 'development' else f'results/{run}/analysis.json')
        if path.exists():
            data[run] = json.loads(path.read_text())
    numbers = {}
    for run, res in data.items():
        flatten(run, dict(competence={k: v for k, v in res['competence'].items() if k != 'triple_competent'},
                          sites=res['sites'], decisions={k: {kk: vv for kk, vv in v.items() if kk != 'rule'}
                                                         for k, v in res['decisions'].items()}), numbers)
    summary = dict(
        study='3-heldout-4b-confirmation', primary_run='box', secondary_run='machine',
        model=json.loads((STUDY / 'code/frozen/competence-v2/model-manifest-qwen4b.json').read_text()),
        reserve_checks={k: v for k, v in json.loads((STUDY / 'data/reserve-checks.json').read_text()).items()
                        if not isinstance(v, list) or len(v) <= 3},
        decisions={run: {k: {kk: vv for kk, vv in v.items()} for k, v in res['decisions'].items()} for run, res in data.items()},
        numbers=numbers)
    for extra in ['instrumentation-summary', 'run-times', 'dev-reproduction']:
        p = STUDY / 'results' / f'{extra}.json'
        if p.exists():
            summary[extra.replace('-', '_')] = json.loads(p.read_text())
    (STUDY / 'results/summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('numbers:', len(numbers))


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'summary':
        summary()
    else:
        res = analyse(which)
        s23 = res['sites']['block23_answer_prefix']
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rule'} for k, v in res['decisions'].items()}, indent=1))
