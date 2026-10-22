"""Deterministic tasks and strict answer-only parsing for a new development branch."""
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 20261001
ROLES = ['a_end', 'b_end', 'first_stage_end', 'prompt_end']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def calc(op, a, b):
    return {'add': a+b, 'sub': a-b, 'mul': a*b}[op]


def task(z, first, b, remaining, k):
    a = z-b if first == 'add' else z+b
    return dict(a=a, b=b, first=first, z=z, remaining=remaining, k=k,
                y=calc(remaining, z, k))


def predictions(d, r):
    g = lambda z: calc(r['remaining'], z, r['k'])
    return dict(counterfactual=g(d['z']), recipient_answer=r['y'],
                donor_answer=d['y'], donor_z=d['z'],
                operand_transfer=g(calc(r['first'], d['a'], d['b'])),
                donor_answer_as_z=g(d['y']), first_operand_as_z=g(d['a']),
                second_operand_as_z=g(d['b']), recipient_z=r['z'])


def prompt(t, family='box', kind='full', z=None):
    """Return exact text and exclusive character boundaries for semantic sites."""
    if kind in ['remaining', 'counterfactual']:
        value = t['z'] if z is None else z
        text = f"Start with {value}. "
        text += {'add': f"Add {t['k']}.", 'sub': f"Subtract {t['k']}.",
                 'mul': f"Multiply by {t['k']}."}[t['remaining']]
        return text + ' Return only the final integer, with no explanation.', {}
    lead = 'A box starts with ' if family == 'box' else 'A machine initially holds '
    text = lead + str(t['a'])
    spans = {'a_end': len(text)}
    text += ' counters. ' if family == 'box' else ' units. '
    text += 'Add ' if t['first'] == 'add' else 'Subtract '
    text += str(t['b'])
    spans['b_end'] = len(text)
    text += ' to finish the first step.' if t['first'] == 'add' else ' to finish the first step.'
    spans['first_stage_end'] = len(text)
    if kind == 'full':
        text += ' Then ' + {'add': f"add {t['k']} to that result.",
                           'sub': f"subtract {t['k']} from that result.",
                           'mul': f"multiply that result by {t['k']}."}[t['remaining']]
    text += ' Return only the final integer, with no explanation.'
    return text, spans


def parse(text):
    text = re.sub(r'<\|(?:im_end|endoftext)\|>', '', text).strip()
    return int(text) if re.fullmatch(r'-?\d+', text) else None


def make_groups(n, seed, split, family, forbidden=None):
    rng = random.Random(seed)
    rows, seen = [], set(forbidden or [])
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
            # Opposite decomposition changes operands and first operation but preserves z.
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
            seen.add(signature)
            for i, (r, pred) in enumerate(zip(recipients, ps)):
                rows.append(dict(id=f'{split}-{group:02d}-{r["remaining"]}', group=group,
                                 split=split, family=family, donor=d, recipient=r,
                                 predictions=pred, controls=dict(same_z_decomposition=alt,
                                 same_z_different_answer=alt_answer,
                                 different_z_same_answer=same_answer)))
            break
        else:
            raise RuntimeError('Could not generate collision-free group')
    return rows, seen


def prepare():
    dev, seen = make_groups(12, SEED, 'development', 'box')
    val, _ = make_groups(24, SEED+1000, 'reserve', 'machine', seen)
    for name, rows in [('development', dev), ('reserve', val)]:
        path = HERE / 'data' / f'{name}.jsonl'
        text = ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != text:
            raise RuntimeError('Refusing to overwrite existing data')
        path.write_text(text)
    dump(HERE/'data/manifest.json', dict(seed=SEED, development_pairs=len(dev),
         reserve_pairs=len(val), development_sha256=sha(HERE/'data/development.jsonl'),
         reserve_sha256=sha(HERE/'data/reserve.jsonl'), reserve_evaluated=False))


if __name__ == '__main__':
    prepare()
