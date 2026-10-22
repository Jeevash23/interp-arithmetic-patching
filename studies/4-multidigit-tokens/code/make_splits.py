"""Draw the calibration, development and test splits of Study 4 and check that they share no problem with each
other or with any earlier AI-02, AI-04 or Study 3 item.

The sampling is the frozen AI-02 generator (`code/frozen/tasks.py`, `make_groups`), repeated here line for line with
one added rejection: a group is redrawn if any of its problems (donor, the three recipients and the four constructed
sources) already occurs in the earlier data or in an earlier split. Problems are compared as
(a, b, first operation, second operation, k). The same-leading-digit source is added by the frozen
`prepare_sources.py` transformation.

  python -B code/make_splits.py      # writes data/{calibration,development,test}.jsonl and data/split-checks.json
"""
import hashlib
import json
import random
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
sys.path.insert(0, str(STUDY / 'code/frozen'))
from tasks import task, predictions, calc  # noqa: E402  frozen copy

SPLITS = [('calibration', 40, 20261009), ('development', 12, 20262009), ('test', 48, 20263009)]
OPS = {'+': 'add', '-': 'sub', '*': 'mul', 'add': 'add', 'sub': 'sub', 'mul': 'mul'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sig(t):
    return (int(t['a']), int(t['b']), OPS[t['first']], OPS[t.get('remaining', t.get('second'))], int(t['k']))


def walk(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)


def earlier_problems():
    """Every two-step problem recorded in a JSON or JSON-lines file of the project, outside this study."""
    found, files = set(), 0
    skip = {'models', '.cache', '.packages', '__pycache__', 'tokenizers'}
    for path in sorted(PROJECT.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.jsonl'}:
            continue
        parts = path.relative_to(PROJECT).parts
        if skip & set(parts) or parts[:2] == ('studies', '4-multidigit-tokens'):
            continue
        files += 1
        text = path.read_text(errors='replace')
        try:
            values = [json.loads(s) for s in text.splitlines() if s.strip()] if path.suffix == '.jsonl' else [json.loads(text)]
        except json.JSONDecodeError:
            continue
        for value in values:
            for d in walk(value):
                if {'a', 'b', 'first', 'k'} <= set(d) and ('remaining' in d or 'second' in d):
                    try:
                        found.add(sig(d))
                    except (KeyError, TypeError, ValueError):
                        pass
    return found, files


def same_leading_source(d):
    """The frozen prepare_sources.py transformation (z + 1, or z - 1 when z ends in 9)."""
    z = d['z'] + 1 if d['z'] % 10 < 9 else d['z'] - 1
    return task(z, d['first'], d['b'], d['remaining'], d['k'])


def make_groups_fresh(n, seed, split, forbidden):
    """Frozen tasks.make_groups, line for line, plus the rejection of any group containing a forbidden problem."""
    rng = random.Random(seed)
    rows, seen = [], set(forbidden)
    redrawn = 0
    for group in range(n):
        for attempt in range(100000):
            zd, zr = rng.sample(range(20, 80), 2)
            if str(zd)[0] == str(zr)[0]:
                continue
            first = ['add', 'sub'][group % 2]
            rd = 'sub' if first == 'add' else 'add'
            d = task(zd, first, rng.randint(2, 9), ['mul', 'add', 'sub'][group % 3], rng.randint(2, 6))
            rb = rng.randint(2, 9)
            recipients = [task(zr, rd, rb, op, rng.randint(2, 6)) for op in ['add', 'sub', 'mul']]
            ps = [predictions(d, r) for r in recipients]
            if any(any(v == p['counterfactual'] for k, v in p.items() if k != 'counterfactual') for p in ps):
                continue
            alt = task(zd, rd, rng.choice([x for x in range(2, 10) if x != d['b']]), d['remaining'], d['k'])
            alt_answer = task(zd, first, d['b'], 'add', 11)
            if alt_answer['y'] == d['y']:
                continue
            dz = d['y'] - 7
            if not 10 <= dz <= 199 or dz == zd:
                continue
            same_answer = task(dz, first, rng.randint(2, 9), 'add', 7)
            signature = (d['a'], d['b'], d['first'], d['remaining'], d['k'])
            if signature in seen:
                continue
            # Added rejection: no problem of the group may occur in earlier data or an earlier split.
            problems = [d, *recipients, alt, alt_answer, same_answer, same_leading_source(d)]
            if any(sig(t) in forbidden for t in problems):
                redrawn += 1
                continue
            seen.add(signature)
            for r, pred in zip(recipients, ps):
                rows.append(dict(id=f'{split}-{group:02d}-{r["remaining"]}', group=group, split=split, family='box',
                                 donor=d, recipient=r, predictions=pred,
                                 controls=dict(same_z_decomposition=alt, same_z_different_answer=alt_answer,
                                               different_z_same_answer=same_answer)))
            break
        else:
            raise RuntimeError('Could not generate a collision-free group')
    return rows, redrawn


def add_sources(rows):
    """Line for line the body of the frozen competence-v2/prepare_sources.py."""
    for row in rows:
        d = row['donor']; r = row['recipient']
        variant = same_leading_source(d)
        assert variant['z'] != d['z'] and str(variant['z'])[0] == str(d['z'])[0]
        assert predictions(variant, r)['counterfactual'] != row['predictions']['counterfactual']
        row['controls']['same_leading_different_value'] = variant
        splice = int(str(d['z'])[0] + str(r['z'])[1:])
        row['digit_alternatives'] = {
            'intermediate_leading_digit_splice': calc(r['remaining'], splice, r['k']),
            'donor_answer_leading_digit_splice': int(str(d['y'])[0] + str(r['y'])[1:])}
    return rows


def problems_of(row):
    return [row['donor'], row['recipient'], *row['controls'].values()]


def main():
    earlier, files = earlier_problems()
    forbidden = set(earlier)
    record = dict(earlier_files_scanned=files, earlier_problems=len(earlier), splits={})
    for split, n, seed in SPLITS:
        path = STUDY / 'data' / f'{split}.jsonl'
        rows, redrawn = make_groups_fresh(n, seed, split, forbidden)
        rows = add_sources(rows)
        text = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_text() == text, f'Refusing to overwrite {path}'
        else:
            path.write_text(text)
        mine = {sig(t) for row in rows for t in problems_of(row)}
        record['splits'][split] = dict(file=str(path.relative_to(STUDY)), sha256=sha(path), seed=seed, groups=n,
                                       pairs=len(rows), problems=len(mine), groups_redrawn_for_overlap=redrawn,
                                       overlap_with_earlier_data=len(mine & earlier),
                                       overlap_with_earlier_splits=len(mine & (forbidden - earlier)),
                                       recipient_operations={op: sum(r['recipient']['remaining'] == op for r in rows)
                                                             for op in ['add', 'sub', 'mul']})
        forbidden |= mine
    for name, info in record['splits'].items():
        assert info['overlap_with_earlier_data'] == 0 and info['overlap_with_earlier_splits'] == 0, name
    record['no_overlap'] = True
    (STUDY / 'data/split-checks.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
