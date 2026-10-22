"""Study 6 ended at the competence gate (PROTOCOL.md section 2). This writes results/summary.json from the gate records:
the gate rates, the instrumentation result, and the error profile over distinct calibration problems, counted as for
Study 4's candidates (by the problem's second operation). Standard library only.

  python -B code/gate_summary.py
"""
import json
from collections import Counter
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results/olmo2-7b/calibration'


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    rows = {r['id']: r for r in rd(STUDY / 'data/calibration.jsonl')}
    base = rd(RES / 'baselines.jsonl')
    problems = {}
    for b in base:
        if b['condition'] not in ('donor_full', 'recipient_full'):
            continue
        t = rows[b['id']]['donor' if b['condition'] == 'donor_full' else 'recipient']
        sig = (t['a'], t['b'], t['first'], t['remaining'], t['k'])
        prev = problems.setdefault(sig, dict(task=t, answer=b['answer'], correct=b['correct']))
        assert prev['answer'] == b['answer']
    wrong = {s: p for s, p in problems.items() if not p['correct']}
    by_op = {op: dict(wrong=sum(1 for p in wrong.values() if p['task']['remaining'] == op),
                      distinct=sum(1 for p in problems.values() if p['task']['remaining'] == op)) for op in ['mul', 'add', 'sub']}
    kinds = Counter()
    for p in wrong.values():
        t, a = p['task'], p['answer']
        if a is None:
            kinds['invalid'] += 1
        elif a == t['z']:
            kinds['the intermediate'] += 1
        elif abs(a - t['y']) <= 10:
            kinds['within 10 of the answer'] += 1
        else:
            kinds['other'] += 1
    gate = json.loads((RES / 'gate.json').read_text())
    comp = json.loads((RES / 'competence.json').read_text())
    inst = json.loads((RES / 'instrumentation.json').read_text())
    summary = dict(
        study='6-olmo2-7b-multidigit',
        status='ended at the competence gate: OLMo-2-1124-7B-Instruct passed the tokenizer check, loading and the '
               'instrumentation check and failed the gate; no development screen, predictions, freeze or test inference',
        candidate=json.loads((STUDY / 'results/candidates.json').read_text())['candidates']['olmo2-7b'],
        tokenizer_check=json.loads((STUDY / 'results/tokenizer-check.json').read_text())['candidates']['olmo2-7b']['pass_check'],
        weights_check=all(v['ok'] for v in json.loads((STUDY / 'results/weights-check-olmo2-7b.json').read_text()).values()),
        instrumentation=dict(pass_check=inst['pass_check'],
                             largest_null_write_logit_change=max(c['max_logit_abs'] for case in inst['cases'] for c in case['null_checks']),
                             null_checks=sum(len(case['null_checks']) for case in inst['cases']),
                             native_equal=sum(case['native_equal'] for case in inst['cases'])),
        gate=dict(pass_check=gate['pass_check'], counts={c: dict(correct=v['correct'], n=v['n'], invalid=v['invalid'])
                                                         for c, v in comp['conditions'].items()},
                  triple_competent=len(comp['triple_competent']), pairs=comp['n_pairs'], thresholds=gate['thresholds']),
        error_profile=dict(distinct_problems=len(problems), distinct_wrong=len(wrong), by_second_operation=by_op,
                           wrong_answer_kinds=dict(kinds),
                           note='the wrong-answer kinds are descriptive and were defined after the gate result'),
        gpu=json.loads((RES / 'runtime-gate.json').read_text())['gpu'],
        gate_seconds=json.loads((RES / 'chunks.jsonl').read_text().splitlines()[0])['seconds'])
    (STUDY / 'results/summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('gate', 'error_profile', 'instrumentation')}, indent=1))


if __name__ == '__main__':
    main()
