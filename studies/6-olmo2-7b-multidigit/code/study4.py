"""Study 4 runner (PROTOCOL.md sections 6 to 9). Every phase holds the GPU and should be wrapped in the shared lock.

  python -B code/study4.py gate    --candidate C   # instrumentation (first 4 calibration pairs), baselines, competence gate
  python -B code/study4.py screen  --candidate C   # development screen: all blocks x 4 positions, 5 complete answers scored
  python -B code/study4.py develop --candidate C   # development baselines, the 12 conditions at the 4 sites, layer sweep
  python -B code/study4.py test    --candidate C   # the same on the test split; refuses to run unless FREEZE.json holds

With --max-minutes, the screen, develop and test phases stop after a completed pair once that many minutes have
passed and exit with status 3; rerunning resumes from the keyed journals. Exit status 0 means the phase is complete.
Outputs go to results/<candidate>/<split>/.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
from candidates import CANDIDATES, ORDER  # noqa: E402

ROLES = ['a_end', 'b_end', 'first_stage_end', 'prompt_end']
CHUNK_EXIT = 3
SCREEN_CANDIDATES = ['counterfactual', 'recipient', 'first_operand_substitution', 'second_operand_substitution', 'donor_answer']
SELECTION = {'prompt_end': 'counterfactual', 'first_stage_end': 'counterfactual',
             'a_end': 'first_operand_substitution', 'b_end': 'second_operand_substitution'}


class ChunkDone(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def calc(op, a, b):
    return {'add': a + b, 'sub': a - b, 'mul': a * b}[op]


def screen_targets(row):
    d, r = row['donor'], row['recipient']
    g = lambda z: calc(r['remaining'], z, r['k'])
    return dict(counterfactual=row['predictions']['counterfactual'], recipient=r['y'],
                first_operand_substitution=g(calc(r['first'], d['a'], r['b'])),
                second_operand_substitution=g(calc(r['first'], r['a'], d['b'])), donor_answer=d['y'])


def update_candidates(candidate, **fields):
    path = STUDY / 'results/candidates.json'
    data = json.loads(path.read_text()) if path.exists() else dict(order=ORDER, candidates={})
    entry = data['candidates'].setdefault(candidate, dict(repository=CANDIDATES[candidate]['repository'],
                                                          revision=CANDIDATES[candidate]['revision']))
    entry.update(fields)
    path.write_text(json.dumps(data, indent=1) + '\n')


def check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise ChunkDone()


def gate(core, engine, rows, results):
    if not core.instrumentation(engine, rows):
        return dict(pass_check=False, reason='instrumentation failed')
    core.previous.baselines(engine, rows)
    comp = json.loads((results / 'competence.json').read_text())
    c = comp['conditions']
    probes = {k: c[k]['correct'] / c[k]['n'] for k in ['donor_first', 'recipient_first']}
    passed = comp['pass_check'] and all(v >= 0.9 for v in probes.values())
    return dict(pass_check=passed, frozen_gate=comp['pass_check'],
                rates={k: c[k]['correct'] / c[k]['n'] for k in c},
                triple_competent=len(comp['triple_competent']) / comp['n_pairs'], first_step_probes=probes,
                thresholds=dict(central=0.9, triple=0.8, first_step=0.9))


def screen(core, engine, rows, results, deadline):
    journal = core.previous.Journal('screen')
    layers = engine.model.config.num_hidden_layers
    for index, row in enumerate(rows):
        keys = [f'{row["id"]}/{layer}/{role}' for layer in range(layers) for role in ROLES]
        if all(k in journal.rows for k in keys):
            continue
        check_deadline(deadline)
        r = engine.encode(*core.previous.prompt(row['recipient'], row['family']))
        d = engine.encode(*core.previous.prompt(row['donor'], row['family']))
        source = engine.capture(d)
        targets = screen_targets(row)
        values = sorted(set(targets.values()))
        unpatched = engine.score_candidates(r, values)
        for layer in range(layers):
            for ri, role in enumerate(ROLES):
                key = f'{row["id"]}/{layer}/{role}'
                if key in journal.rows:
                    continue
                spec = (layer, r['positions'][role], source[layer, ri], 'replace')
                scores = engine.score_candidates(r, values, spec)
                journal.put(key, dict(id=row['id'], group=row['group'], layer=layer, role=role,
                                      recipient_operation=row['recipient']['remaining'], targets=targets,
                                      log_probability={name: scores[v]['log_probability'] for name, v in targets.items()},
                                      token_log_probabilities={name: scores[v]['token_log_probabilities'] for name, v in targets.items()},
                                      unpatched_log_probability={name: unpatched[v]['log_probability'] for name, v in targets.items()}))
        print('Screen pairs', index + 1, '/', len(rows), flush=True)
    sites = []
    for layer in range(layers):
        for role in ROLES:
            recs = [x for x in journal.rows.values() if x['layer'] == layer and x['role'] == role]
            sites.append(dict(layer=layer, role=role, n=len(recs), mean_probability={
                name: sum(math.exp(x['log_probability'][name]) for x in recs) / len(recs) for name in SCREEN_CANDIDATES}))
    selected = []
    for role in ['a_end', 'b_end', 'first_stage_end', 'prompt_end']:
        target = SELECTION[role]
        best = sorted([s for s in sites if s['role'] == role], key=lambda s: (-s['mean_probability'][target], s['layer']))[0]
        selected.append(dict(layer=best['layer'], role=role, selected_by=f'largest mean probability of the complete {target}',
                             mean_probability=best['mean_probability'][target]))
    core.dump(results / 'screen-summary.json', dict(sites=sites, selected=selected,
                                                    selection='development split; PROTOCOL.md section 8'))
    print('Selected', [(s['layer'], s['role']) for s in selected], flush=True)


def text_controls(core, engine, rows, results, sites, deadline):
    """Condition 11 (donor run's own first token, once per pair) and 12 (the donor write's own first token, per site)."""
    journal = core.previous.Journal('text-controls')
    base = {r['key']: r for r in rd(results / 'baselines.jsonl')}
    controls = {r['key']: r for r in rd(results / 'controls.jsonl')}
    for n, row in enumerate(rows, 1):
        i = row['id']
        keys = [f'{i}/text_first_generated_donor_digit'] + [f'{i}/{l}/{r}/text_activation_first_token' for l, r in sites]
        if all(k in journal.rows for k in keys):
            continue
        check_deadline(deadline)
        encoded = engine.encode(*core.previous.prompt(row['recipient'], row['family']))
        assert encoded['prompt_ids'] == base[i + '/recipient_full']['prompt_ids']
        cf = row['predictions']['counterfactual']
        donor = base[i + '/donor_full']
        if keys[0] not in journal.rows:
            out = engine.generate(encoded, forced=donor['output_ids'][:1])
            journal.put(keys[0], dict(id=i, group=row['group'], layer=None, role=None, condition='text_first_generated_donor_digit',
                                      source_baseline_key=donor['key'], source_output_ids=donor['output_ids'],
                                      counterfactual=cf, counterfactual_success=out['answer'] == cf, **encoded, **out))
        for layer, role in sites:
            key = f'{i}/{layer}/{role}/text_activation_first_token'
            if key in journal.rows:
                continue
            act = controls[f'{i}/{layer}/{role}/donor']
            out = engine.generate(encoded, forced=act['output_ids'][:1])
            journal.put(key, dict(id=i, group=row['group'], layer=layer, role=role, condition='text_activation_first_token',
                                  activation_key=act['key'], activation_output_ids=act['output_ids'], counterfactual=cf,
                                  counterfactual_success=out['answer'] == cf, **encoded, **out))
        print('Text controls', n, '/', len(rows), flush=True)


def sweep(core, engine, rows, results, deadline):
    """Answer-prefix layer sweep: donor write and same-leading-digit source at every block."""
    journal = core.previous.Journal('sweep')
    layers = engine.model.config.num_hidden_layers
    ri = ROLES.index('prompt_end')
    for n, row in enumerate(rows, 1):
        keys = [f'{row["id"]}/{layer}/{c}' for layer in range(layers) for c in ['donor', 'same_leading_different_value']]
        if all(k in journal.rows for k in keys):
            continue
        check_deadline(deadline)
        encoded = engine.encode(*core.previous.prompt(row['recipient'], row['family']))
        sources = dict(donor=row['donor'], same_leading_different_value=row['controls']['same_leading_different_value'])
        captures = {c: engine.capture(engine.encode(*core.previous.prompt(t, row['family']))) for c, t in sources.items()}
        for layer in range(layers):
            for c, t in sources.items():
                key = f'{row["id"]}/{layer}/{c}'
                if key in journal.rows:
                    continue
                vector = captures[c][layer, ri]
                out = engine.generate(encoded, (layer, encoded['positions']['prompt_end'], vector, 'replace'))
                digest = hashlib.sha256(vector.contiguous().view(core.torch.uint8).numpy().tobytes()).hexdigest()
                journal.put(key, dict(id=row['id'], group=row['group'], layer=layer, role='prompt_end', condition=c,
                                      source_task=t, recipient_task=row['recipient'], source_hash=digest,
                                      recipient_position=encoded['positions']['prompt_end'],
                                      prompt_ids=encoded['prompt_ids'], **out))
        print('Sweep pairs', n, '/', len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['gate', 'screen', 'develop', 'test'])
    ap.add_argument('--candidate', choices=ORDER, required=True)
    ap.add_argument('--max-minutes', type=float, default=None)
    args = ap.parse_args()
    split = {'gate': 'calibration', 'screen': 'development', 'develop': 'development', 'test': 'test'}[args.phase]
    results = STUDY / 'results' / args.candidate / split
    results.mkdir(parents=True, exist_ok=True)
    done = results / f'completed-{args.phase}.json'
    if done.exists():
        print(args.phase, 'already complete', flush=True)
        return 0
    started = now()
    if args.phase == 'test':
        freeze = json.loads((STUDY / 'FREEZE.json').read_text())
        changed = [rel for rel, digest in freeze['sha256'].items() if sha(STUDY / rel) != digest]
        if changed:
            raise SystemExit(f'Frozen files changed, refusing to run: {changed}')
        assert started > freeze['frozen_at_utc'] and freeze['candidate'] == args.candidate
    else:
        assert not (STUDY / 'FREEZE.json').exists() or args.phase == 'gate', 'Development phases run before the freeze'
    if args.phase in ('screen', 'develop', 'test'):
        gate_record = json.loads((STUDY / 'results' / args.candidate / 'calibration/gate.json').read_text())
        assert gate_record['pass_check'], 'The candidate did not pass the gate'
    if args.phase in ('develop', 'test'):
        assert (STUDY / 'results' / args.candidate / 'development/completed-screen.json').exists()
    import engine as E  # loads torch and the frozen code
    core = E.core
    core.previous.RESULTS = results
    rows = rd(STUDY / 'data' / f'{split}.jsonl')
    t0 = time.monotonic()
    deadline = t0 + 60 * args.max_minutes if args.max_minutes else None
    chunk = dict(phase=args.phase, candidate=args.candidate, started_utc=started, max_minutes=args.max_minutes)
    try:
        engine = E.Engine(args.candidate, args.phase)
    except Exception as exc:  # loading failure is a recorded gate outcome (PROTOCOL.md section 2)
        if args.phase != 'gate':
            raise
        record = dict(pass_check=False, reason='loading failed', error=repr(exc), traceback=traceback.format_exc()[-4000:],
                      at_utc=now())
        (results / 'gate.json').write_text(json.dumps(record, indent=1) + '\n')
        update_candidates(args.candidate, status='failed: loading', gate=record)
        print('Loading failed:', repr(exc), flush=True)
        return 2
    status = 0
    try:
        if args.phase == 'gate':
            record = gate(core, engine, rows, results)
            record.update(at_utc=now(), cuda_free_total_loaded=engine.runtime['cuda_free_total_loaded'],
                          dtype_counts=engine.runtime['dtype_counts'])
            (results / 'gate.json').write_text(json.dumps(record, indent=1) + '\n')
            update_candidates(args.candidate, status='passed' if record['pass_check'] else 'failed: ' + record.get('reason', 'competence gate'),
                              gate={k: v for k, v in record.items() if k != 'dtype_counts'})
        elif args.phase == 'screen':
            screen(core, engine, rows, results, deadline)
        else:
            if args.phase == 'test':
                dev_sites = json.loads((STUDY / 'results' / args.candidate / 'development/screen-summary.json').read_text())
                path = results / 'screen-summary.json'
                selected = dict(selected=[dict(layer=s['layer'], role=s['role']) for s in dev_sites['selected']],
                                selection='fixed on the development split (PROTOCOL.md section 8); no screen on the test pairs')
                if path.exists():
                    assert json.loads(path.read_text()) == selected
                else:
                    core.dump(path, selected)
            sites = [(s['layer'], s['role']) for s in json.loads((results / 'screen-summary.json').read_text())['selected']]
            core.previous.baselines(engine, rows)  # resumes from its journal
            core.source_baselines(engine, rows)
            if deadline is not None:
                save = core.torch.save

                def save_then_check(*a, **k):
                    save(*a, **k)
                    check_deadline(deadline)
                core.torch.save = save_then_check
            core.controls(engine, rows)
            text_controls(core, engine, rows, results, sites, deadline)
            sweep(core, engine, rows, results, deadline)
    except ChunkDone:
        status = CHUNK_EXIT
    chunk.update(finished_utc=now(), seconds=round(time.monotonic() - t0, 1), complete=status == 0,
                 max_cuda_allocated=core.torch.cuda.max_memory_allocated())
    with (results / 'chunks.jsonl').open('a') as f:
        f.write(json.dumps(chunk) + '\n')
    if status == 0:
        core.dump(done, dict(completed=True, phase=args.phase, candidate=args.candidate, finished_utc=chunk['finished_utc']))
    print(json.dumps(chunk), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
