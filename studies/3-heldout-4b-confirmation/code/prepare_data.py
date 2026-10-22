"""Build the held-out test files from the untouched 72-pair reserve.

1. Copies the reserve (hash-checked against the 1 October manifest) to data/reserve.jsonl.
2. Adds the same-leading-digit source and the two leading-digit splice predictions with the
   transformation of the frozen competence-v2/prepare_sources.py, applied unchanged. As a check,
   the same function applied to the 36 development pairs must reproduce the saved
   development-with-source-controls.jsonl byte for byte.
3. Writes the primary test file in the reserve's own wording family ('machine') and the secondary
   file with the development wording ('box'); the numbers, identifiers and sources are identical.

Run: python -B code/prepare_data.py   (standard library only)
"""
import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
HV = PROJECT / '06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer'
sys.path.insert(0, str(STUDY / 'code/frozen'))
from tasks import task, predictions, calc  # frozen copy


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_sources(rows):
    """Line-for-line the body of the frozen competence-v2/prepare_sources.py."""
    for row in rows:
        d = row['donor']; r = row['recipient']; z = d['z'] + 1 if d['z'] % 10 < 9 else d['z'] - 1
        variant = task(z, d['first'], d['b'], d['remaining'], d['k'])
        assert z != d['z'] and str(z)[0] == str(d['z'])[0]
        assert predictions(variant, r)['counterfactual'] != row['predictions']['counterfactual']
        row['controls']['same_leading_different_value'] = variant
        splice = int(str(d['z'])[0] + str(r['z'])[1:])
        row['digit_alternatives'] = {
            'intermediate_leading_digit_splice': calc(r['remaining'], splice, r['k']),
            'donor_answer_leading_digit_splice': int(str(d['y'])[0] + str(r['y'])[1:])}
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)


def load(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f'Refusing to overwrite {path}'
    else:
        path.write_text(text)


def main():
    manifest = json.loads((HV / 'data/manifest.json').read_text())
    assert sha(HV / 'data/reserve.jsonl') == manifest['reserve_sha256'] and manifest['reserve_evaluated'] is False
    # Equivalence check on the development pairs.
    dev_text = add_sources(load(HV / 'data/development.jsonl'))
    saved = (HV / 'competence-v3/development-with-source-controls.jsonl').read_text()
    assert dev_text == saved, 'Transformation differs from the frozen prepare_sources.py output'
    reserve_text = (HV / 'data/reserve.jsonl').read_text()
    write(STUDY / 'data/reserve.jsonl', reserve_text)
    machine = add_sources(load(HV / 'data/reserve.jsonl'))
    write(STUDY / 'data/test-machine-with-source-controls.jsonl', machine)
    rows = [json.loads(s) for s in machine.splitlines()]
    for r in rows:
        r['family'] = 'box'
    box = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
    write(STUDY / 'data/test-box-with-source-controls.jsonl', box)
    record = dict(
        source_reserve=str((HV / 'data/reserve.jsonl').relative_to(PROJECT)),
        source_reserve_sha256=manifest['reserve_sha256'],
        development_equivalence_check=dict(
            reproduced=str((HV / 'competence-v3/development-with-source-controls.jsonl').relative_to(PROJECT)),
            sha256=sha(HV / 'competence-v3/development-with-source-controls.jsonl'), identical=True),
        files={name: sha(STUDY / 'data' / name) for name in
               ['reserve.jsonl', 'test-machine-with-source-controls.jsonl', 'test-box-with-source-controls.jsonl']},
        pairs=len(rows), donor_groups=len({r['group'] for r in rows}),
        recipient_operations={op: sum(r['recipient']['remaining'] == op for r in rows) for op in ['add', 'sub', 'mul']},
        note='The box file differs from the machine file only in the family field, which selects the wording.')
    (STUDY / 'data/data-manifest.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
