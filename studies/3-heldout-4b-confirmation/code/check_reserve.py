"""Check that the 72-pair reserve was never evaluated and does not overlap the development pairs.

Reads only. Scans every JSON and JSON-lines record in the AI-02-04 project (the original study trees,
the merged results and data copies) for reserve identifiers, reserve prompts (in both wordings) and
reserve arithmetic problems, and compares the reserve with the 36 development pairs, the three
demonstrations and every other task set of the project. Writes data/reserve-checks.json.

Run: python -B code/check_reserve.py   (standard library only)
"""
import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
HV = PROJECT / '06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer'
sys.path.insert(0, str(STUDY / 'code/frozen'))
import ast
from tasks import prompt  # frozen copy, byte-identical to the original (see code/frozen/COPIED-FROM.json)
# The demonstrations, read from the frozen prompt_formats.py without importing it (it needs torch).
EXAMPLES = next(ast.literal_eval(node.value) for node in ast.parse((STUDY / 'code/frozen/prompt_formats.py').read_text()).body
                if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'EXAMPLES')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def sig(t):
    return (t['a'], t['b'], t['first'], t['remaining'], t['k'])


def tasks_of(row):
    out = {'donor': row['donor'], 'recipient': row['recipient']}
    out.update({'control:' + k: v for k, v in row['controls'].items()})
    return out


def walk(value):
    """Yield every dict nested in a JSON value."""
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)


def main():
    manifest = json.loads((HV / 'data/manifest.json').read_text())
    reserve_path, dev_path = HV / 'data/reserve.jsonl', HV / 'data/development.jsonl'
    reserve, dev = rows(reserve_path), rows(dev_path)
    result = dict(
        reserve_file=str(reserve_path.relative_to(PROJECT)), reserve_sha256=sha(reserve_path),
        manifest_reserve_sha256=manifest['reserve_sha256'], manifest_reserve_evaluated=manifest['reserve_evaluated'],
        reserve_pairs=len(reserve), reserve_groups=len({r['group'] for r in reserve}),
        reserve_families=sorted({r['family'] for r in reserve}),
        development_sha256=sha(dev_path), manifest_development_sha256=manifest['development_sha256'])
    # 1. Overlap with the development pairs and the demonstrations.
    rsig = {}
    for r in reserve:
        for role, t in tasks_of(r).items():
            rsig.setdefault(sig(t), []).append(f"{r['id']}:{role}")
    dsig = {sig(t) for r in dev for t in tasks_of(r).values()}
    rpairs = {(sig(r['donor']), sig(r['recipient'])) for r in reserve}
    dpairs = {(sig(r['donor']), sig(r['recipient'])) for r in dev}
    result['pair_overlap_with_development'] = len(rpairs & dpairs)
    result['task_overlap_with_development'] = sorted(map(list, set(rsig) & dsig))
    result['donor_overlap_with_development'] = len({sig(r['donor']) for r in reserve} & {sig(r['donor']) for r in dev})
    result['recipient_overlap_with_development'] = len({sig(r['recipient']) for r in reserve} & {sig(r['recipient']) for r in dev})
    demo_texts = {inp for inp, _ in EXAMPLES}
    # 2. Reserve prompts in every wording the pipeline can emit.
    texts = {}
    for r in reserve:
        for role, t in tasks_of(r).items():
            for family in ['machine', 'box']:
                for kind in ['full', 'first']:
                    texts[prompt(t, family, kind)[0]] = f"{r['id']}:{role}:{family}:{kind}"
        rec = r['recipient']
        texts[prompt(rec, 'machine', 'remaining')[0]] = f"{r['id']}:recipient:remaining"
        texts[prompt(rec, 'machine', 'counterfactual', r['donor']['z'])[0]] = f"{r['id']}:counterfactual"
    result['demonstration_overlap'] = sorted(set(texts) & demo_texts)
    # The two short prompts (remaining step and supplied intermediate) use one wording for every family
    # and can coincide with development prompts; they are reported separately below.
    short = {t for t, label in texts.items() if label.endswith(':remaining') or label.endswith(':counterfactual')}
    # 3. Scan every JSON record in the project for reserve identifiers, prompts and problems.
    scanned, id_hits, prompt_hits, short_prompt_hits, task_hits, pair_hits = 0, [], [], [], [], []
    skip_dirs = {'models', '.cache', '.packages', '__pycache__', 'studies'}
    data_files = {reserve_path.resolve()} | {p.resolve() for p in PROJECT.glob('03-Data/**/reserve.jsonl')}
    for path in sorted(PROJECT.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.jsonl'}:
            continue
        if skip_dirs & set(path.relative_to(PROJECT).parts):
            continue
        if path.resolve() in data_files:
            continue
        scanned += 1
        text = path.read_text(errors='replace')
        rel = str(path.relative_to(PROJECT))
        if 'reserve-' in text:
            for line in text.splitlines():
                if '"reserve-' in line or "'reserve-" in line:
                    id_hits.append(rel)
                    break
        try:
            values = [json.loads(s) for s in text.splitlines() if s.strip()] if path.suffix == '.jsonl' else [json.loads(text)]
        except json.JSONDecodeError:
            continue
        for value in values:
            for d in walk(value):
                p = d.get('prompt')
                if isinstance(p, str) and p in texts:
                    (short_prompt_hits if p in short else prompt_hits).append(dict(file=rel, label=texts[p]))
                for dk, rk in [('donor', 'recipient'), ('source_task', 'recipient_task')]:
                    if isinstance(d.get(dk), dict) and isinstance(d.get(rk), dict):
                        try:
                            if (sig(d[dk]), sig(d[rk])) in rpairs:
                                pair_hits.append(dict(file=rel, donor=list(sig(d[dk])), recipient=list(sig(d[rk]))))
                        except (KeyError, TypeError):
                            pass
                if {'a', 'b', 'first', 'remaining', 'k'} <= set(d) and all(isinstance(d[x], (int, str)) for x in ['a', 'b', 'k']):
                    try:
                        s = sig(d)
                    except TypeError:
                        continue
                    if s in rsig:
                        task_hits.append(dict(file=rel, task=list(s), reserve_roles=rsig[s]))
    def uniq(items):
        seen, out = set(), []
        for it in items:
            key = json.dumps(it, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out
    prompt_hits = uniq(prompt_hits)
    result.update(
        json_files_scanned=scanned,
        reserve_identifier_files=sorted(set(id_hits)),
        reserve_pair_records=uniq(pair_hits),
        reserve_full_prompts_machine_wording=[h for h in prompt_hits if h['label'].endswith(':machine:full')],
        reserve_full_prompts_box_wording=[h for h in prompt_hits if h['label'].endswith(':box:full')],
        reserve_first_step_prompts=[h for h in prompt_hits if h['label'].endswith(':first')],
        reserve_short_prompt_hits=uniq(short_prompt_hits),
        reserve_problem_hits=uniq(task_hits),
        definitions=dict(
            never_evaluated='hash matches the 1 October manifest, the manifest records reserve_evaluated false, no record carries '
                            'a reserve identifier, no record pairs a reserve donor with its reserve recipient, and no record holds a '
                            'reserve full-problem prompt in the reserve wording (machine)',
            coincidences='first-step prompts depend only on (a, b, first) and the remaining-step and supplied-intermediate '
                         'prompts only on (z, operation, k), so independently generated items from the same small ranges '
                         'share them; complete problems can also coincide across independently generated sets'))
    result['never_evaluated'] = (result['reserve_sha256'] == manifest['reserve_sha256'] and manifest['reserve_evaluated'] is False
                                 and not id_hits and not pair_hits and not result['reserve_full_prompts_machine_wording'])
    result['no_overlap_with_development'] = result['pair_overlap_with_development'] == 0 and not result['task_overlap_with_development']
    out = STUDY / 'data/reserve-checks.json'
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 12 else f'{len(v)} items') for k, v in result.items()}, indent=1))


if __name__ == '__main__':
    main()
