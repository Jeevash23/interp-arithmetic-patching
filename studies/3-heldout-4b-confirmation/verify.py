"""Independent verification of Study 3 from the raw journals (written after the freeze, as PROTOCOL.md section 10 says).

It shares no code with code/analyze.py or code/stats.py. It checks:
  1. the freeze: every file in FREEZE.json is unchanged, and every run chunk started after the freeze time;
  2. completeness: every expected record exists once (baselines, source baselines, controls, text controls);
  3. parsing: every recorded answer equals a fresh strict parse of the recorded text;
  4. writes: every vector write hit its target (zero write error, no change at other positions), at the position
     the tokenizer gives for that role, with a saved vector whose SHA-256 equals the recorded hash;
  5. identities: unpatched and self-write outputs equal the recipient baseline;
  6. outcomes: every count and every interval in results/<wording>/analysis.json, recomputed here from the raw
     generations with separately written code (exact intervals from the beta distribution, the donor-cluster bootstrap
     re-implemented), and every verdict re-derived from the PREDICTIONS.md rules.
Writes results/verification.json. Needs torch (CPU, to read the saved vectors), transformers (tokenizer only) and NumPy.
The exact interval is computed here from the regularized incomplete beta function (continued fraction), not by the
binomial-sum bisection of code/stats.py.

  python -B verify.py
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
SITES = [(23, 'prompt_end'), (20, 'a_end'), (9, 'a_end'), (10, 'a_end'), (14, 'b_end'), (20, 'first_stage_end')]
NAMES = {(23, 'prompt_end'): 'block23_answer_prefix', (20, 'a_end'): 'block20_first_operand', (9, 'a_end'): 'block9_first_operand',
         (10, 'a_end'): 'block10_first_operand', (14, 'b_end'): 'block14_second_operand', (20, 'first_stage_end'): 'block20_first_step_end'}
CONDITIONS = ['donor', 'unpatched', 'same_example', 'isotropic_delta', 'random_source', 'same_z_decomposition',
              'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value', 'text_first_answer_digit']
VECTOR_CONDITIONS = ['donor', 'same_example', 'isotropic_delta', 'random_source', 'same_z_decomposition',
                     'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value']
checks = {}


def check(name, ok, detail=None):
    checks[name] = dict(ok=bool(ok), detail=detail)
    if not ok:
        print('FAIL', name, detail if detail is not None else '', flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def strict_parse(text):
    body = text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    return int(body) if re.fullmatch(r'-?[0-9]+', body) else None


def op(name, x, y):
    return x + y if name == 'add' else x - y if name == 'sub' else x * y


# Exact binomial interval, written independently: regularized incomplete beta by Lentz continued fraction.
def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(c) > 1e-300 else 1 + aa / 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(c) > 1e-300 else 1 + aa / 1e-300
        delta = d * c; h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h


def ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * betacf(a, b, x) / a
    return 1 - math.exp(lbt) * betacf(b, a, 1 - x) / b


def beta_quantile(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if ibeta(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact(k, n):
    lower = 0.0 if k == 0 else beta_quantile(0.025, k, n - k + 1)
    upper = 1.0 if k == n else beta_quantile(0.975, k + 1, n - k)
    return lower, upper


def cluster(values, groups, resamples=10000, seed=20261009):
    keys = sorted(set(groups))
    pos = {g: j for j, g in enumerate(keys)}
    s = np.zeros(len(keys)); c = np.zeros(len(keys))
    for v, g in zip(values, groups):
        s[pos[g]] += float(v); c[pos[g]] += 1
    draws = np.random.default_rng(seed).integers(0, len(keys), (resamples, len(keys)))
    m = s[draws].sum(1) / c[draws].sum(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def estimate(flags, groups):
    keep = [(bool(f), g) for f, g in zip(flags, groups) if f is not None]
    k, n = sum(f for f, _ in keep), len(keep)
    lo, hi = exact(k, n)
    cl, ch = cluster([f for f, _ in keep], [g for _, g in keep])
    return dict(k=k, n=n, lower=min(lo, cl), upper=max(hi, ch), exact=(lo, hi), cluster=(cl, ch))


def verify_run(wording, freeze, tok):
    res = STUDY / 'results' / wording
    rows = lines(STUDY / 'data' / f'test-{wording}-with-source-controls.jsonl')
    ids = [r['id'] for r in rows]; groups = [r['group'] for r in rows]
    base = {r['key']: r for r in lines(res / 'baselines.jsonl')}
    src = {r['key']: r for r in lines(res / 'source-baselines.jsonl')}
    ctl = lines(res / 'controls.jsonl')
    txt = {r['key']: r for r in lines(res / 'text-controls.jsonl')}
    by = {(r['id'], r['layer'], r['role'], r['condition']): r for r in ctl}
    # 1. freeze timing and runtime records
    chunks = lines(res / 'chunks.jsonl')
    check(f'{wording}: chunks started after the freeze', all(c['started_utc'] > freeze['frozen_at_utc'] for c in chunks), len(chunks))
    for phase in ['baseline', 'controls']:
        rt = json.loads((res / f'runtime-{phase}.json').read_text())
        check(f'{wording}: runtime-{phase} freeze hash', rt.get('freeze_sha256') == sha(STUDY / 'FREEZE.json'))
    # 2. completeness
    base_keys = {f'{i}/{c}' for i in ids for c in ['donor_full', 'recipient_full', 'donor_first', 'recipient_first',
                                                    'recipient_remaining', 'textual_counterfactual']}
    check(f'{wording}: baselines complete', set(base) == base_keys, len(base))
    src_keys = {f'{r["id"]}/{name}/{c}' for r in rows for name in r['controls'] for c in ['full', 'first', 'target_remaining']}
    check(f'{wording}: source baselines complete', set(src) == src_keys, len(src))
    ctl_keys = {(i, l, r, c) for i in ids for l, r in SITES for c in CONDITIONS}
    check(f'{wording}: controls complete and unique', set(by) == ctl_keys and len(ctl) == len(ctl_keys), len(ctl))
    txt_keys = {f'{i}/text_first_generated_donor_digit' for i in ids} | {f'{i}/{l}/{r}/text_activation_first_token' for i in ids for l, r in SITES}
    check(f'{wording}: text controls complete', set(txt) == txt_keys, len(txt))
    # 3. parsing
    all_recs = list(base.values()) + list(src.values()) + ctl + list(txt.values())
    bad = [r['key'] for r in all_recs if strict_parse(r['text']) != r['answer']]
    check(f'{wording}: answers equal a fresh strict parse', not bad, bad[:5])
    # 4. writes, positions and vectors
    import torch
    vectors = torch.load(res / 'control-vectors.pt', weights_only=True, map_location='cpu')
    sys.path.insert(0, str(STUDY / 'code/frozen'))
    from tasks import prompt as task_prompt  # frozen prompt text and character spans (no model code)
    pos_bad, write_bad, hash_bad = [], [], []
    positions = {}
    for row in rows:
        text, spans = task_prompt(row['recipient'], row['family'])
        serial = by[row['id'], 23, 'prompt_end', 'donor']['serialized_prompt']
        enc = tok(serial, add_special_tokens=False, return_offsets_mapping=True)
        start = serial.rindex(text)
        p = {'prompt_end': len(enc['input_ids']) - 1}
        for name, end in spans.items():
            p[name] = max(j for j, (a, b) in enumerate(enc['offset_mapping']) if start < b <= start + end)
        positions[row['id']] = p
        if enc['input_ids'] != by[row['id'], 23, 'prompt_end', 'donor']['prompt_ids']:
            pos_bad.append(row['id'] + ':ids')
    for r in ctl:
        if r['positions'] != positions[r['id']]:
            pos_bad.append(r['key'])
        if r['condition'] in VECTOR_CONDITIONS:
            w = r['write']
            if not (w.get('write_error') == 0 and w.get('other_positions_max_change') == 0 and w.get('layer') == r['layer']
                    and w.get('position') == positions[r['id']][r['role']] and w.get('delta_norm', 0) >= 0):
                write_bad.append(r['key'])
            v = vectors[r['key']]
            if hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != r['source_hash']:
                hash_bad.append(r['key'])
        elif r['write']:
            write_bad.append(r['key'] + ':unexpected write')
    check(f'{wording}: write positions equal the tokenizer positions', not pos_bad, pos_bad[:5])
    check(f'{wording}: every vector write hit only its target', not write_bad, write_bad[:5])
    check(f'{wording}: saved vectors match recorded hashes', not hash_bad, dict(n_bad=len(hash_bad), n=len(vectors)))
    # A donor write changes its node unless the donor and recipient prompts are identical up to that position (for
    # example a shared first operand at the first-operand position), where the two vectors must be identical.
    moved_ok, same_prefix = 0, 0
    for i in ids:
        dfull = base[f'{i}/donor_full']
        for l, r in SITES:
            x = by[i, l, r, 'donor']
            p_d, p_r = dfull['positions'][r], x['positions'][r]
            identical = p_d == p_r and dfull['prompt_ids'][:p_d + 1] == x['prompt_ids'][:p_r + 1]
            same_prefix += identical
            moved_ok += (x['write']['delta_norm'] == 0) if identical else (x['write']['delta_norm'] > 0)
    check(f'{wording}: donor writes change their node exactly where the prompts differ', moved_ok == len(ids) * len(SITES),
          dict(consistent=moved_ok, identical_prefix_writes=same_prefix))
    # 5. identities
    ident = sum(by[i, l, r, c]['output_ids'] == base[f'{i}/recipient_full']['output_ids']
                for i in ids for l, r in SITES for c in ['unpatched', 'same_example'])
    check(f'{wording}: unpatched and self writes equal the baseline', ident == len(ids) * len(SITES) * 2, ident)
    # Text controls force what they claim to force.
    tf = all(txt[f'{i}/text_first_generated_donor_digit']['forced_output_ids'] == base[f'{i}/donor_full']['output_ids'][:1]
             and txt[f'{i}/text_first_generated_donor_digit']['output_ids'][:1] == base[f'{i}/donor_full']['output_ids'][:1] for i in ids)
    to = all(txt[f'{i}/{l}/{r}/text_activation_first_token']['forced_output_ids'] == by[i, l, r, 'donor']['output_ids'][:1]
             for i in ids for l, r in SITES)
    check(f'{wording}: text controls force the intended first token', tf and to)
    # 6. outcomes
    analysis = json.loads((res / 'analysis.json').read_text())
    mismatches = []
    mine = {}
    for site in SITES:
        l, rl = site
        a = analysis['sites'][NAMES[site]]
        out = {}
        flags = {k: [] for k in ['cf', 'follow', 'text', 'text11', 'own', 'sub1', 'sub2', 'unch', 'same_leading', 'noise', 'diverge_given_follow']}
        for row in rows:
            i, d, r = row['id'], row['donor'], row['recipient']
            g = lambda z: op(r['remaining'], z, r['k'])
            x = by[i, l, rl, 'donor']; rec = base[f'{i}/recipient_full']['answer']
            first = base[f'{i}/donor_full']['output_ids'][0]
            flags['cf'].append(x['answer'] == g(d['z']))
            flags['follow'].append(x['output_ids'][0] == first)
            flags['text'].append(x['output_ids'] == by[i, l, rl, 'text_first_answer_digit']['output_ids'])
            flags['text11'].append(x['output_ids'] == txt[f'{i}/text_first_generated_donor_digit']['output_ids'])
            flags['own'].append(x['output_ids'] == txt[f'{i}/{l}/{rl}/text_activation_first_token']['output_ids'])
            p1, p2 = g(op(r['first'], d['a'], r['b'])), g(op(r['first'], r['a'], d['b']))
            flags['sub1'].append(x['answer'] == p1 if p1 != rec else None)
            flags['sub2'].append(x['answer'] == p2 if p2 != rec else None)
            flags['unch'].append(x['answer'] == rec)
            flags['same_leading'].append(x['output_ids'] == by[i, l, rl, 'same_leading_different_value']['output_ids'])
            flags['noise'].append(by[i, l, rl, 'isotropic_delta']['output_ids'][0] == first)
            flags['diverge_given_follow'].append((x['output_ids'] != txt[f'{i}/text_first_generated_donor_digit']['output_ids'])
                                                 if x['output_ids'][0] == first else None)
        names = dict(cf='complete_counterfactual', follow='first_token_follows_donor', text='equals_first_digit_text_edit',
                     text11='equals_actual_donor_token_text_edit', own='equals_own_first_token_text_edit',
                     sub1='first_operand_substitution_informative', sub2='second_operand_substitution_informative',
                     unch='recipient_answer_unchanged', same_leading='same_leading_source_identical_output',
                     noise='noise_first_token_follows_donor', diverge_given_follow='diverges_given_first_token_follows')
        for key, name in names.items():
            e = estimate(flags[key], groups)
            out[name] = e
            ref = a[name]
            if (e['k'], e['n']) != (ref['k'], ref['n']) or abs(e['lower'] - ref['lower']) > 1e-6 or abs(e['upper'] - ref['upper']) > 1e-6:
                mismatches.append((NAMES[site], name, (e['k'], e['n'], round(e['lower'], 6), round(e['upper'], 6)),
                                   (ref['k'], ref['n'], round(ref['lower'], 6), round(ref['upper'], 6))))
        diff = [int(f) - int(n) for f, n in zip(flags['follow'], flags['noise'])]
        out['minus_noise'] = cluster(diff, groups)
        mine[NAMES[site]] = out
    check(f'{wording}: outcome counts and intervals recomputed', not mismatches, mismatches[:6])
    # Verdicts from the PREDICTIONS.md rules.
    s23 = mine['block23_answer_prefix']
    v = {}
    cfs = [mine[NAMES[s]]['complete_counterfactual'] for s in SITES]
    v['P1'] = 'supported' if all(x['upper'] < 0.2 for x in cfs) else 'refuted' if any(x['k'] / x['n'] >= 0.2 for x in cfs) else 'inconclusive'
    b = s23['first_token_follows_donor']; mn = s23['minus_noise']
    v['P2'] = 'supported' if b['lower'] > 0.5 and mn[0] > 0 else 'refuted' if b['upper'] < 0.5 or mn[1] <= 0 else 'inconclusive'
    for key, name in [('P3', 'equals_first_digit_text_edit'), ('P3s', 'equals_own_first_token_text_edit')]:
        c = s23[name]
        v[key] = 'supported' if c['k'] / c['n'] >= 0.8 and c['lower'] >= 0.65 else 'refuted' if c['upper'] < 0.8 else 'inconclusive'
    p4 = []
    for s, name in [((20, 'a_end'), 'first_operand_substitution_informative'), ((9, 'a_end'), 'first_operand_substitution_informative'),
                    ((10, 'a_end'), 'first_operand_substitution_informative'), ((14, 'b_end'), 'second_operand_substitution_informative')]:
        x = mine[NAMES[s]][name]
        p4.append('supported' if x['lower'] >= 0.25 else 'refuted' if x['upper'] < 0.25 else 'inconclusive')
    v['P4'] = ('supported' if all(x == 'supported' for x in p4) else 'refuted' if all(x == 'refuted' for x in p4)
               else f"supported at {p4.count('supported')} of 4 sites")
    e = mine['block20_first_step_end']['recipient_answer_unchanged']
    v['P5'] = 'supported' if e['lower'] >= 0.8 else 'refuted' if e['upper'] < 0.8 else 'inconclusive'
    sl = s23['same_leading_source_identical_output']
    v['P6'] = 'supported' if sl['lower'] > 0.5 else 'refuted' if sl['upper'] < 0.5 else 'inconclusive'
    recorded = {k: x['verdict'] for k, x in analysis['decisions'].items()}
    check(f'{wording}: verdicts re-derived', v == recorded, dict(recomputed=v, recorded=recorded))
    # Competence gate recomputed.
    comp = json.loads((res / 'competence.json').read_text())
    rates = {c: sum(base[f'{i}/{c}']['correct'] for i in ids) / len(ids) for c in ['donor_full', 'recipient_full', 'textual_counterfactual']}
    correct_again = all(base[f'{i}/{c}']['correct'] == (base[f'{i}/{c}']['answer'] == base[f'{i}/{c}']['target'])
                        for i in ids for c in ['donor_full', 'recipient_full', 'textual_counterfactual'])
    triple = [i for i in ids if all(base[f'{i}/{c}']['correct'] for c in ['donor_full', 'recipient_full', 'textual_counterfactual'])]
    gate = all(x >= 0.9 for x in rates.values()) and len(triple) / len(ids) >= 0.8
    check(f'{wording}: competence gate recomputed', correct_again and gate == comp['pass_check'] and sorted(triple) == sorted(comp['triple_competent']),
          dict(rates=rates, triple=len(triple), gate=gate))
    return dict(verdicts=v, competence_rates=rates, triple_competent=len(triple))


def main():
    freeze = json.loads((STUDY / 'FREEZE.json').read_text())
    changed = [rel for rel, d in freeze['sha256'].items() if sha(STUDY / rel) != d]
    check('freeze: all frozen files unchanged', not changed, changed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDY / 'models/qwen4b', local_files_only=True)
    runs = {}
    for wording in ['box', 'machine']:
        if (STUDY / 'results' / wording / 'completed-controls.json').exists():
            runs[wording] = verify_run(wording, freeze, tok)
    result = dict(all_passed=all(c['ok'] for c in checks.values()), checks=checks, runs=runs)
    (STUDY / 'results/verification.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print('all passed' if result['all_passed'] else 'FAILURES', sum(c['ok'] for c in checks.values()), '/', len(checks))


if __name__ == '__main__':
    main()
