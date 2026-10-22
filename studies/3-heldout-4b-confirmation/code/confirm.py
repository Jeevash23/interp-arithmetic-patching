"""Run the frozen AI-02 4B answer-only assay on the held-out reserve at the six fixed development sites.

  flock $S/gpu.lock $PY -B code/confirm.py --wording box --phase baseline
  flock $S/gpu.lock $PY -B code/confirm.py --wording box --phase controls --max-minutes 25   (repeat until exit 0)
  (then the same with --wording machine; code/run_test.sh runs the whole sequence)

baseline: the frozen instrumentation checks on the first four test pairs, then the frozen six behavioural
          baselines and competence gate (run.baselines).
controls: the frozen source baselines and the ten frozen conditions at the six sites (study.controls, which
          reads its site list from screen-summary.json), then the two added text controls (below).
All outputs go to results/<wording>/. The script stops unless every file listed in FREEZE.json is unchanged.

With --max-minutes, the controls phase ends after the first completed pair (vectors saved) once that many minutes
have passed, and exits with status 3; rerunning resumes from the journals, whose records are keyed and never
rewritten. Exit status 0 means the phase is complete.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FROZEN = STUDY / 'code/frozen'
SITES = [(23, 'prompt_end'), (20, 'a_end'), (9, 'a_end'), (10, 'a_end'), (14, 'b_end'), (20, 'first_stage_end')]
ORDER = ['box', 'machine']  # primary run first (PROTOCOL.md section 4)
CHUNK_EXIT = 3


class ChunkDone(Exception):
    """Raised at a pair boundary once the chunk's time budget is used."""


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def check_freeze():
    freeze = json.loads((STUDY / 'FREEZE.json').read_text())
    changed = [rel for rel, digest in freeze['sha256'].items() if sha(STUDY / rel) != digest]
    if changed:
        raise SystemExit(f'Frozen files changed, refusing to run: {changed}')
    return freeze


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def text_controls(core, engine, rows, results, deadline):
    """Condition 11 (donor run's own first token, once per pair) and condition 12 (the donor write's own first token, per site)."""
    journal = core.previous.Journal('text-controls')
    base = {r['key']: r for r in rd(results / 'baselines.jsonl')}
    controls = {r['key']: r for r in rd(results / 'controls.jsonl')}
    for n, row in enumerate(rows, 1):
        i = row['id']
        keys = [f'{i}/text_first_generated_donor_digit'] + [f'{i}/{layer}/{role}/text_activation_first_token' for layer, role in SITES]
        if all(k in journal.rows for k in keys):
            continue
        if deadline is not None and time.monotonic() > deadline:
            raise ChunkDone()
        encoded = engine.encode(*core.previous.prompt(row['recipient'], row['family']))
        assert encoded['prompt_ids'] == base[i + '/recipient_full']['prompt_ids']
        cf = row['predictions']['counterfactual']
        donor = base[i + '/donor_full']
        key = f'{i}/text_first_generated_donor_digit'
        if key not in journal.rows:
            forced = donor['output_ids'][:1]
            out = engine.generate(encoded, forced=forced)
            journal.put(key, dict(id=i, group=row['group'], layer=None, role=None, condition='text_first_generated_donor_digit',
                                  source_baseline_key=donor['key'], source_output_ids=donor['output_ids'], counterfactual=cf,
                                  counterfactual_success=out['answer'] == cf, **encoded, **out))
        for layer, role in SITES:
            key = f'{i}/{layer}/{role}/text_activation_first_token'
            if key in journal.rows:
                continue
            act = controls[f'{i}/{layer}/{role}/donor']
            forced = act['output_ids'][:1]
            out = engine.generate(encoded, forced=forced)
            journal.put(key, dict(id=i, group=row['group'], layer=layer, role=role, condition='text_activation_first_token',
                                  activation_key=act['key'], activation_output_ids=act['output_ids'], counterfactual=cf,
                                  counterfactual_success=out['answer'] == cf, **encoded, **out))
        print('Text controls', n, '/', len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wording', choices=ORDER, required=True)
    ap.add_argument('--phase', choices=['baseline', 'controls'], required=True)
    ap.add_argument('--max-minutes', type=float, default=None, help='controls phase: stop at a pair boundary after this long')
    args = ap.parse_args()
    freeze = check_freeze()
    started = datetime.now(timezone.utc).isoformat()
    assert started > freeze['frozen_at_utc']
    for earlier in ORDER[:ORDER.index(args.wording)]:
        assert (STUDY / f'results/{earlier}/completed-controls.json').exists(), f'The {earlier} run comes first'
    sys.path.insert(0, str(FROZEN / 'competence-v2'))
    sys.path.insert(0, str(FROZEN))
    import study as core  # frozen competence-v2/study.py: offline mode, kernel cache, package path, Engine
    from prompt_formats import EXAMPLES
    results = STUDY / 'results' / args.wording
    results.mkdir(parents=True, exist_ok=True)
    if (results / f'completed-{args.phase}.json').exists():
        print(f'{args.wording} {args.phase} already complete', flush=True)
        return 0
    core.previous.RESULTS = results
    data = STUDY / 'data' / f'test-{args.wording}-with-source-controls.jsonl'
    rows = rd(data)
    assert len(rows) == 72 and {r['family'] for r in rows} == {args.wording}
    if args.phase == 'controls':
        assert json.loads((results / 'instrumentation.json').read_text())['pass_check'], 'Instrumentation gate failed'
        assert (results / 'completed-baseline.json').exists(), 'The baseline phase comes first'
    t0 = time.monotonic()
    deadline = t0 + 60 * args.max_minutes if args.max_minutes else None
    engine = core.Engine('qwen4b', args.phase)
    engine.format_name = 'fewshot'
    runtime_path = results / f'runtime-{args.phase}.json'
    runtime = json.loads(runtime_path.read_text())
    runtime.update(calibration_format='fewshot', examples=EXAMPLES, wording=args.wording, data_file=str(data.relative_to(STUDY)),
                   data_sha256=sha(data), freeze_sha256=sha(STUDY / 'FREEZE.json'), started_utc=started)
    runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + '\n')
    chunk = dict(wording=args.wording, phase=args.phase, started_utc=started, max_minutes=args.max_minutes)
    status = 0
    try:
        if args.phase == 'baseline':
            if not core.instrumentation(engine, rows):
                (results / 'STOPPED.json').write_text(json.dumps(dict(reason='instrumentation gate failed', at=started)) + '\n')
                raise SystemExit('Instrumentation gate failed; the study stops (PROTOCOL.md section 7).')
            core.previous.baselines(engine, rows)
        else:
            sites = dict(selected=[dict(layer=l, role=r) for l, r in SITES],
                         selection='fixed by PROTOCOL.md section 5 from the development study; no screen was run on the test pairs. '
                                   'The frozen study.controls reads its sites from this file name.')
            path = results / 'screen-summary.json'
            if path.exists():
                assert json.loads(path.read_text()) == sites
            else:
                path.write_text(json.dumps(sites, indent=1) + '\n')
            core.source_baselines(engine, rows)
            if deadline is not None:
                # study.controls saves the vector file once per completed pair; stop right after such a save.
                save = core.torch.save

                def save_then_check(*a, **k):
                    save(*a, **k)
                    if time.monotonic() > deadline:
                        raise ChunkDone()
                core.torch.save = save_then_check
            core.controls(engine, rows)
            text_controls(core, engine, rows, results, deadline)
    except ChunkDone:
        status = CHUNK_EXIT
    chunk.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=round(time.monotonic() - t0, 1),
                 complete=status == 0, max_cuda_allocated=core.torch.cuda.max_memory_allocated())
    with (results / 'chunks.jsonl').open('a') as f:
        f.write(json.dumps(chunk) + '\n')
    if status == 0:
        record = dict(completed=True, wording=args.wording, phase=args.phase, finished_utc=chunk['finished_utc'],
                      freeze_sha256=sha(STUDY / 'FREEZE.json'), chunks=str((results / 'chunks.jsonl').relative_to(STUDY)),
                      reserve_evaluated=True)
        (results / f'completed-{args.phase}.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(chunk), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
