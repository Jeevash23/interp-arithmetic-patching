"""Environment check before the freeze, on development pairs only.

Regenerates saved development records of the frozen AI-02 4B run (the first two donor groups: baselines,
donor writes at three sites and the first-digit text edit) and compares them token for token, and the
captured donor vectors byte for byte, with the saved development journals. No test pair is touched.

  flock $S/gpu.lock $PY -B code/reproduce_dev.py
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
HV = PROJECT / '06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer'
FROZEN = STUDY / 'code/frozen'
sys.path.insert(0, str(FROZEN / 'competence-v2'))
sys.path.insert(0, str(FROZEN))
import study as core  # noqa: E402  frozen competence-v2/study.py
from tasks import ROLES  # noqa: E402


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    out_dir = STUDY / 'results/dev-reproduction'
    out_dir.mkdir(parents=True, exist_ok=True)
    core.previous.RESULTS = out_dir
    engine = core.Engine('qwen4b', 'dev-reproduction')
    engine.format_name = 'fewshot'
    v3 = HV / 'competence-v3/results'
    rows = [r for r in rd(HV / 'competence-v3/development-with-source-controls.jsonl') if r['group'] < 2]
    base = {r['key']: r for r in rd(v3 / 'qwen4b-fewshot/baselines.jsonl')}
    saved = {}
    for name in ['qwen4b-fewshot/controls.jsonl', 'eligible-sites/controls.jsonl']:
        for r in rd(v3 / name):
            saved[r['id'], r['layer'], r['role'], r['condition']] = r
    checks = []
    for row in rows:
        i = row['id']
        for cond, side in [('donor_full', 'donor'), ('recipient_full', 'recipient')]:
            enc = engine.encode(*core.previous.prompt(row[side], row['family']))
            old = base[f'{i}/{cond}']
            assert enc['prompt_ids'] == old['prompt_ids']
            out = engine.generate(enc)
            checks.append(dict(record=old['key'], tokens_equal=out['output_ids'] == old['output_ids']))
        rec = engine.encode(*core.previous.prompt(row['recipient'], row['family']))
        donor = engine.capture(engine.encode(*core.previous.prompt(row['donor'], row['family'])))
        for layer, role in [(23, 'prompt_end'), (20, 'a_end'), (14, 'b_end')]:
            old = saved[i, layer, role, 'donor']
            vec = donor[layer, ROLES.index(role)]
            digest = hashlib.sha256(vec.contiguous().view(core.torch.uint8).numpy().tobytes()).hexdigest()
            out = engine.generate(rec, (layer, rec['positions'][role], vec, 'replace'))
            checks.append(dict(record=old['key'], tokens_equal=out['output_ids'] == old['output_ids'],
                               vector_bytes_equal=digest == old['source_hash']))
            text = saved[i, layer, role, 'text_first_answer_digit']
            out = engine.generate(rec, forced=text['forced_output_ids'])
            checks.append(dict(record=text['key'], tokens_equal=out['output_ids'] == text['output_ids']))
        print('Reproduced', i, flush=True)
    result = dict(stage='environment check before the freeze; development pairs only',
                  at_utc=datetime.now(timezone.utc).isoformat(), records=len(checks),
                  tokens_equal=sum(c['tokens_equal'] for c in checks),
                  vectors_checked=sum('vector_bytes_equal' in c for c in checks),
                  vectors_equal=sum(c.get('vector_bytes_equal', False) for c in checks),
                  checks=checks, runtime=f'results/dev-reproduction/runtime-dev-reproduction.json')
    result['all_equal'] = result['tokens_equal'] == result['records'] and result['vectors_equal'] == result['vectors_checked']
    (STUDY / 'results/dev-reproduction.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    main()
