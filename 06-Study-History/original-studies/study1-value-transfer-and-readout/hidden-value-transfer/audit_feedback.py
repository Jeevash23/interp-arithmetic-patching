"""Recompute review claims from immutable September journals; no inference."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
from tasks import HERE, ROOT, calc, dump, sha


def read(rel):
    return [json.loads(line) for line in (ROOT/rel).read_text().splitlines()]


def main():
    data = {r['id']:r for r in read('followup-readout/data.jsonl')}
    all_rows = []
    inputs = ['followup-readout/data.jsonl','data/development.jsonl',
              'results/v3/development/controls-n100/controls.jsonl']
    for seed in [20262909,20263909]:
        rel = f'followup-readout/results/seed-{seed}/interventions.jsonl'
        inputs.append(rel)
        all_rows.extend(read(rel))
    lookup = {(r['seed'],r['id'],r['condition']):r for r in all_rows}
    assert len(lookup)==len(all_rows)
    stages, counts = [], collections.defaultdict(collections.Counter)
    activation = [r for r in all_rows if r['condition']=='activation']
    for r in all_rows:
        if r['condition'] not in ['activation','text_full_value']:
            continue
        d = data[r['id']]
        parsed = r.get('parsed',{})
        z, y = parsed.get('intermediate'), parsed.get('answer')
        expected_splice = int(str(d['donor']['z'])[0]+str(d['recipient']['z'])[1:])
        expected_y = calc(d['recipient']['remaining'], z, d['recipient']['k']) if z is not None else None
        c = counts[r['condition']]
        c['planned'] += 1
        c['valid_final'] += y is not None
        c['valid_intermediate'] += z is not None
        c['splice_pattern'] += z == expected_splice
        c['consistent_second_step'] += expected_y is not None and y == expected_y
        # Separate permissive label extraction from the study's strict answer parser.
        tail = r.get('text','').rsplit('</think>',1)[-1]
        label = re.findall(r'(?m)^Intermediate:\s*(-?\d+)\s*(?:$|<)',tail)
        c['numeric_intermediate_label_anywhere_after_last_close'] += bool(label)
        stages.append(dict(seed=r['seed'],id=r['id'],group=d['group'],condition=r['condition'],
            donor_z=d['donor']['z'],recipient_z=d['recipient']['z'],printed_z=z,final=y,
            predicted_splice=expected_splice,splice_matches=z==expected_splice,
            predicted_second_step=expected_y,second_step_matches=expected_y is not None and y==expected_y,
            strict_parser_valid=y is not None,has_closed_thinking='</think>' in r.get('text','')))
    exact = available = copy = 0
    clamps = {v:dict(planned=240,available=0,exact=0) for v in ['recipient','donor']}
    width_exact = 0
    for r in activation:
        c = lookup[r['seed'],r['id'],'text_first_digit']
        ok = r['status']=='generated' and c['status']=='generated'
        available += ok
        exact += ok and r['output_ids']==c['output_ids']
        copy += bool(r.get('fresh_output_ids')) and r['fresh_output_ids'][0]==r['source_next_token']
        if data[r['id']]['width']==1:
            sibling = r['id'].removesuffix('w1')+'w2'
            width_exact += r['output_ids']==lookup[r['seed'],sibling,'activation']['output_ids']
        for fixed in clamps:
            a = lookup[r['seed'],r['id'],f'activation_{fixed}_clamp']
            b = lookup[r['seed'],r['id'],'text_recipient_clamp' if fixed=='recipient' else 'text_full_value']
            ok = a['status']=='generated' and b['status']=='generated'
            clamps[fixed]['available'] += ok
            clamps[fixed]['exact'] += ok and a['output_ids']==b['output_ids']
    dev = read('data/development.jsonl')
    controls = {r['id']:r for r in read('results/v3/development/controls-n100/controls.jsonl') if r['condition']=='donor'}
    same = [r for r in dev if str(r['donor']['z'])[0]==str(r['recipient']['z'])[0]]
    remaining = [r for r in dev if len(str(r['donor']['z']))==2 and r not in same]
    summary = dict(date='2026-10-01',analysis='post-hoc descriptive audit of external feedback',
        input_sha256={rel:sha(ROOT/rel) for rel in inputs},two_stage=dict(counts),
        first_digit_sequence_agreement=dict(exact=exact,available=available,planned=240),
        paired_width_agreement=dict(exact=width_exact,planned=120),source_first_token_copy=dict(matches=copy,n=240),
        success={condition:dict(correct=sum(r.get('success',False) for r in all_rows if r['condition']==condition),planned=240) for condition in ['activation','text_full_value']},
        clamps=clamps,
        development=dict(same_first_digit=len(same),same_first_digit_unchanged=sum(controls[r['id']].get('parsed',{}).get('answer')==r['recipient']['y'] for r in same),
            two_digit_total=75,different_first_digit_two_digit=len(remaining),
            different_first_digit_two_digit_success=sum(controls[r['id']].get('success',False) for r in remaining)),
        cautions=['Splice equality is not causal identification of suffix copying.',
            'Arithmetic consistency is not proof that the printed number mediates the answer.',
            'Sequence agreement under a fixed sampling stream is not a mediation percentage.',
            'Two truncated activation rows have an Intermediate label inside unclosed thinking; the strict parser rejects them.'])
    out = HERE/'review'
    dump(out/'feedback-audit.json',summary)
    with (out/'two-stage-scoring.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','cautions']},indent=2))


if __name__=='__main__':
    main()
