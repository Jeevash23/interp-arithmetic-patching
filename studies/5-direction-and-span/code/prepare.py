"""Study 5 inputs (PROTOCOL.md sections 3 and 4). Runs on the local machine's CPU; no model is run.

  python -B code/prepare.py

1. Copies the development pairs (the 36 pairs of the Qwen3-4B development study, Study 1) and the test pairs (the 72
   held-out pairs of Study 3 in the development wording) byte for byte into data/, checking their SHA-256.
2. Draws the fitting problems for the mean-difference directions: for every intermediate value v from 19 to 80,
   24 problems with z = v, four in each cell of (first operation) x (second operation), distinct second operands,
   seeded; no fitting problem equals any problem (donor, recipient or constructed source) of the two splits.
3. Extracts the reproduction reference: the saved Study 3 records of donor group 0 (box wording) that the first cloud
   stage regenerates (baselines, the donor write at the six sites, the unpatched run at block 23), and the 18 saved
   donor vectors of those writes (data/reproduction-vectors.pt).
Refuses to overwrite a file whose content would change.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
S3 = PROJECT / 'studies/3-heldout-4b-confirmation'
DEV = PROJECT / ('06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/'
                 'competence-v3/development-with-source-controls.jsonl')
SOURCES = {'development.jsonl': (DEV, '0e993773e473e67b9ebdf87edda1c1ba4953d4673cfb2a7a80d59acb5c96f2f1'),
           'test.jsonl': (S3 / 'data/test-box-with-source-controls.jsonl',
                          'e0f057d97d73f51c95215b27d67b2f73a872f6a3ea60922d0263def79af5a464')}
SITES = [(23, 'prompt_end'), (20, 'a_end'), (9, 'a_end'), (10, 'a_end'), (14, 'b_end'), (20, 'first_stage_end')]
FIT_VALUES = range(19, 81)
FIT_PER_CELL = 4
FIT_SEED = 20261017
sys.path.insert(0, str(STUDY / 'code/frozen'))
from tasks import task  # noqa: E402  frozen generator helper: task(z, first, b, remaining, k)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_once(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f'refusing to overwrite {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def signature(t):
    return (t['a'], t['b'], t['first'], t['remaining'], t['k'])


def fitting(forbidden):
    rng = random.Random(FIT_SEED)
    rows = []
    for v in FIT_VALUES:
        for first in ['add', 'sub']:
            for remaining in ['add', 'sub', 'mul']:
                bs = rng.sample(range(2, 10), 8)
                chosen = 0
                for b in bs:
                    t = task(v, first, b, remaining, rng.randint(2, 6))
                    if signature(t) in forbidden or not 10 <= t['a'] <= 99:
                        continue
                    rows.append(dict(id=f'fit-{v}-{first}-{remaining}-{chosen}', z=v, task=t))
                    chosen += 1
                    if chosen == FIT_PER_CELL:
                        break
                assert chosen == FIT_PER_CELL, (v, first, remaining)
    return rows


def main():
    manifest = {}
    for name, (src, digest) in SOURCES.items():
        assert sha(src) == digest, src
        write_once(STUDY / 'data' / name, src.read_text())
        manifest[name] = dict(source=str(src.relative_to(PROJECT)), sha256=digest, rows=len(rd(src)))
    splits = {n: rd(STUDY / 'data' / n) for n in SOURCES}
    forbidden = set()
    for rows in splits.values():
        for r in rows:
            for t in [r['donor'], r['recipient'], *r['controls'].values()]:
                forbidden.add(signature(t))
    fit = fitting(forbidden)
    text = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in fit)
    write_once(STUDY / 'data/fitting.jsonl', text)
    manifest['fitting.jsonl'] = dict(sha256=sha(STUDY / 'data/fitting.jsonl'), rows=len(fit), values=[min(FIT_VALUES), max(FIT_VALUES)],
                                     per_value=6 * FIT_PER_CELL, seed=FIT_SEED, overlap_with_splits=0)
    # Values every direction needs: donor, recipient and same-leading-digit source intermediates.
    needed = set()
    for rows in splits.values():
        for r in rows:
            needed |= {r['donor']['z'], r['recipient']['z'], r['controls']['same_leading_different_value']['z']}
    assert needed <= set(FIT_VALUES), sorted(needed - set(FIT_VALUES))
    manifest['direction_values_needed'] = [min(needed), max(needed)]

    # Reproduction reference (Study 3, box wording, donor group 0).
    res = S3 / 'results/box'
    base = {r['key']: r for r in rd(res / 'baselines.jsonl')}
    ref, keys = [], []
    ids = [r['id'] for r in splits['test.jsonl'] if r['group'] == 0]
    controls = {}
    with (res / 'controls.jsonl').open() as fh:
        for line in fh:
            r = json.loads(line)
            if r['id'] in ids and (r['condition'] == 'donor' or (r['condition'] == 'unpatched' and r['layer'] == 23)):
                controls[r['key']] = r
    for i in ids:
        for c in ['donor_full', 'recipient_full']:
            b = base[f'{i}/{c}']
            ref.append(dict(key=b['key'], id=i, kind='baseline', condition=c, prompt_ids=b['prompt_ids'],
                            output_ids=b['output_ids'], answer=b['answer']))
        for layer, role in SITES + [(23, 'unpatched')]:
            key = f'{i}/23/prompt_end/unpatched' if role == 'unpatched' else f'{i}/{layer}/{role}/donor'
            c = controls[key]
            ref.append(dict(key=key, id=i, kind='control', layer=c['layer'], role=c['role'], condition=c['condition'],
                            prompt_ids=c['prompt_ids'], output_ids=c['output_ids'], answer=c['answer'],
                            source_hash=c['source_hash']))
            if c['condition'] == 'donor':
                keys.append(key)
    write_once(STUDY / 'data/reproduction-reference.json',
               json.dumps(dict(source='studies/3-heldout-4b-confirmation/results/box (baselines.jsonl, controls.jsonl, '
                                      'control-vectors.pt)', records=ref), indent=1) + '\n')
    vec_path = STUDY / 'data/reproduction-vectors.pt'
    if not vec_path.exists():
        import torch
        saved = torch.load(res / 'control-vectors.pt', weights_only=True)
        sub = {k: saved[k].clone() for k in keys}
        for k in keys:
            digest = hashlib.sha256(sub[k].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            assert digest == controls[k]['source_hash'], k
        torch.save(sub, vec_path)
    manifest['reproduction-reference.json'] = dict(sha256=sha(STUDY / 'data/reproduction-reference.json'), records=len(ref))
    manifest['reproduction-vectors.pt'] = dict(sha256=sha(vec_path), vectors=len(keys))
    write_once(STUDY / 'data/data-manifest.json', json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
