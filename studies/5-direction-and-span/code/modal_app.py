"""Modal app for Study 5 (Qwen3-4B-Instruct-2507-FP8 on an L4). From the study folder:

    ~/.local/bin/modal run --detach code/modal_app.py::first      # instrumentation, reproduction check, class means, development
    ~/.local/bin/modal run code/modal_app.py::pull --stage first
    ~/.local/bin/modal run --detach code/modal_app.py::devsites   # development controls at the selected sites
    ~/.local/bin/modal run --detach code/modal_app.py::test       # after FREEZE.json: the held-out pairs
    ~/.local/bin/modal run code/modal_app.py::pull --stage test

Each stage runs inside Modal in orchestrate(), so a dropped local connection cannot stop it (the AI-03 Study 5 pattern).
The weights download inside the image build at the pinned revision with every SHA-256 asserted; they never reach the
local machine. Records go to the Volume `ai0204-study5`; only the small gzipped per-pair records and stage summaries come back.
Before each stage the billing report is read; no stage starts once this task's apps (`ai0204-*`) have spent $8 or the
workspace's month total has reached $27.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO, REVISION = 'Qwen/Qwen3-4B-Instruct-2507-FP8', '8591804019c8b22094c3b5b4454e0edc05dffc98'
REMOTE = '/root/s5'
MODEL_DIR = f'{REMOTE}/code/frozen/competence-v2/models/qwen4b'
DIGESTS = {'LICENSE': '832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e',
           'README.md': 'f95f1bd2f5ee5879c6dad8aad55c2c141cd9dfe2b7ec53c8b8822e515ba1fc5d',
           'config.json': '46c086c1975c333a0c1423371a711d9927179c01b54a7b7c308a8e90ac9030ee',
           'generation_config.json': '19d306dd769db12a9d710b44cf7f83b635efbe5166b84fb4358a08fb7d88bb53',
           'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
           'model.safetensors': 'b6154d74332140fd6dfbfbe70bbb3650dd6955861132bd59dda6789e6322b485',
           'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
           'tokenizer_config.json': 'a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3',
           'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
CODE = ['code/study5.py', 'code/plans.json', 'code/modal_app.py', 'code/prepare.py', 'code/freeze.py', 'code/stats.py',
        'code/analyze.py', 'code/cost_rule.py', 'code/COPIED-FROM-STUDY3.json']
DATA = ['data/development.jsonl', 'data/test.jsonl', 'data/fitting.jsonl', 'data/reproduction-reference.json',
        'data/reproduction-vectors.pt', 'data/data-manifest.json']
OPTIONAL = ['PROTOCOL.md', 'PREDICTIONS.md', 'FREEZE.json', 'PROTOCOL-FREEZE.json', 'AMENDMENTS.md',
            'results/development-selection.json']
STOP_TASK_USD, STOP_MONTH_USD = 8.0, 27.0
WORKERS = 4


def download_weights():
    from huggingface_hub import hf_hub_download
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    record = {}
    for f, want in DIGESTS.items():
        hf_hub_download(REPO, f, revision=REVISION, local_dir=MODEL_DIR)
        with open(Path(MODEL_DIR) / f, 'rb') as stream:
            got = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert got == want, (f, got)
        record[f] = dict(sha256=got, bytes=(Path(MODEL_DIR) / f).stat().st_size)
    cache = Path(MODEL_DIR) / '.cache'
    if cache.exists():                                   # hf_hub_download's local_dir metadata; not a model file
        import shutil
        shutil.rmtree(cache)
    (Path(MODEL_DIR).parent / 'build-record-qwen4b.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION, files=record), indent=2))


if modal.is_local():
    STUDY = Path(__file__).resolve().parents[1]
    AI01 = STUDY.parents[1]   # release: pyproject.toml and uv.lock are at the release root
else:
    STUDY = AI01 = Path('/nonexistent')

image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(uv_project_dir=str(AI01), extras=['experiment'], frozen=True)
         .add_local_dir(str(STUDY / 'code/frozen'), f'{REMOTE}/code/frozen', copy=True,
                        ignore=['**/__pycache__/**', 'competence-v2/.cache/triton/**', 'competence-v2/models/**'])
         .run_function(download_weights)
         .env({'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}))
for name in [n for n in CODE + DATA + OPTIONAL if (STUDY / n).exists()]:
    image = image.add_local_file(str(STUDY / name), f'{REMOTE}/{name}')
app = modal.App('ai0204-study5', image=image, tags={'project': 'AI-02-04', 'study': '5-direction-and-span'})
volume = modal.Volume.from_name('ai0204-study5', create_if_missing=True)
VOL = '/vol'
GPU_OPTS = dict(gpu='L4', cpu=2.0, memory=12288, timeout=4 * 3600, scaledown_window=10, volumes={VOL: volume})


# ======================================================================================== remote side
def _rd(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_freeze(freeze):
    """Every frozen file that is mounted in the container (code, data, protocol files) must match its hash; the
    local entry points check the rest (the development records) before a stage starts."""
    mounted = [rel for rel in freeze['sha256'] if rel.startswith(('code/', 'data/')) or rel in OPTIONAL]
    changed = [rel for rel in mounted if _sha(Path(REMOTE) / rel) != freeze['sha256'][rel]]
    if changed:
        raise RuntimeError(f'files differ from the freeze record: {changed}')


def _first(task):
    """Instrumentation, the reproduction check (stop if it fails), then the class means of the fitting problems."""
    sys.path.insert(0, f'{REMOTE}/code')
    import study5 as S
    out_dir = Path(VOL) / 'first'
    target = out_dir / 'checks.json'
    volume.reload()
    if target.exists():
        return target.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_freeze(task['protocol_freeze'])
    t0 = time.perf_counter()
    engine = S.Engine('first', out_dir)
    dev = _rd(f'{REMOTE}/data/development.jsonl')
    test0 = [r for r in _rd(f'{REMOTE}/data/test.jsonl') if r['group'] == 0]     # the reproduction pairs only
    result = dict(started_utc=S.utc(), runtime=json.loads((out_dir / 'runtime-first.json').read_text()))
    result['instrumentation'] = S.instrumentation(engine, dev)
    reference = json.loads(Path(f'{REMOTE}/data/reproduction-reference.json').read_text())
    vectors = S.torch.load(f'{REMOTE}/data/reproduction-vectors.pt', weights_only=True)
    result['reproduction'] = S.reproduction(engine, test0, reference, vectors)
    result['pass_check'] = result['instrumentation']['pass_check'] and result['reproduction']['pass_check']
    if result['pass_check']:
        values, means, counts = S.class_means(engine, _rd(f'{REMOTE}/data/fitting.jsonl'))
        S.torch.save(dict(values=values, means=means, counts=counts), out_dir / 'means.pt')
        result['means'] = dict(values=values, counts=counts.tolist(), sha256=_sha(out_dir / 'means.pt'),
                               shape=list(means.shape))
    result.update(finished_utc=S.utc(), seconds=time.perf_counter() - t0)
    target.write_text(json.dumps(result, indent=1))
    volume.commit()
    return target.read_bytes()


def _pairs(task):
    sys.path.insert(0, f'{REMOTE}/code')
    import study5 as S
    stage = task['stage']
    if stage == 'test':
        _check_freeze(task['freeze'])
    else:
        _check_freeze(task['protocol_freeze'])
    volume.reload()
    base = Path(VOL) / stage
    todo = [i for i in task['ids'] if not (base / f'{i}.jsonl.gz').exists()]
    if not todo:
        return json.dumps(dict(done=task['ids'], resumed=True)).encode()
    m = S.torch.load(Path(VOL) / 'first/means.pt', weights_only=True)
    assert _sha(Path(VOL) / 'first/means.pt') == task['means_sha256']
    means = m['means'].to('cuda')
    engine = S.Engine(stage, base / 'runtime' / todo[0])
    rows = _rd(f'{REMOTE}/data/{task["split"]}.jsonl')
    group_donor = {r['group']: r['donor'] for r in rows}
    n_layers = engine.model.config.num_hidden_layers
    by_id = {r['id']: r for r in rows}
    done = []
    for i in todo:
        t0 = time.perf_counter()
        p = S.Pair(engine, by_id[i], means, m['values'], group_donor)
        records = S.run_pair(p, task['plan'], n_layers)
        S.write_pair(base / f'{i}.jsonl.gz', records)
        volume.commit()
        done.append(i)
        print(json.dumps(dict(stage=stage, id=i, records=len(records), seconds=round(time.perf_counter() - t0, 1))), flush=True)
    return json.dumps(dict(done=done)).encode()


@app.function(**GPU_OPTS, max_containers=WORKERS)
def gpu_job(task: dict) -> bytes:
    return _first(task) if task['kind'] == 'first' else _pairs(task)


@app.function(cpu=0.25, memory=1024, timeout=8 * 3600, volumes={VOL: volume}, max_containers=1)
def orchestrate(task: dict) -> str:
    stage = task['stage']
    summary = dict(stage=stage, started_utc=datetime.now(timezone.utc).isoformat(), status='running')
    target = Path(VOL) / 'orchestration' / f'{stage}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    volume.commit()
    try:
        if stage == 'first':
            checks = json.loads(gpu_job.remote(dict(task, kind='first')))
            summary['first_pass'] = checks['pass_check']
            if not checks['pass_check']:
                raise RuntimeError('the first-stage checks failed; the study stops (PROTOCOL.md section 8)')
            task = dict(task, stage='development', means_sha256=checks['means']['sha256'])
        ids = task['ids']
        chunks = [ids[k::WORKERS] for k in range(WORKERS)]
        failures = []
        for res in gpu_job.map([dict(task, kind='pairs', ids=c) for c in chunks if c], return_exceptions=True):
            if isinstance(res, Exception):
                failures.append(repr(res))
        volume.reload()
        sub = task['stage']
        summary.update(pairs_done=sorted(p.name[:-9] for p in (Path(VOL) / sub).glob('*.jsonl.gz')), failures=failures)
        if failures:
            raise RuntimeError(f'workers failed: {failures}')
        summary['status'] = 'finished'
    except Exception as error:                                                   # noqa: BLE001
        summary.update(status='failed', error=repr(error))
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary['status']


# ======================================================================================== local side
def _billing():
    """This task's spend (every ai0204-* app) and the workspace's month total, from Modal's billing report."""
    for _ in range(6):
        proc = subprocess.run([str(Path.home() / '.local/bin/modal'), 'billing', 'report', '--for', 'this month', '--json'],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            break
        time.sleep(20)
    else:
        raise SystemExit(f'could not read the billing report: {proc.stderr[-300:]}')
    rows = json.loads(proc.stdout)
    task_usd = sum(float(r['cost']) for r in rows if r['description'].startswith('ai0204-'))
    month_usd = sum(float(r['cost']) for r in rows)
    entry = dict(utc=datetime.now(timezone.utc).isoformat(), task_usd=round(task_usd, 4), month_usd=round(month_usd, 4))
    (STUDY / 'results').mkdir(exist_ok=True)
    with (STUDY / 'results/billing-log.jsonl').open('a') as fh:
        fh.write(json.dumps(entry) + '\n')
    return task_usd, month_usd


def _guard(stage):
    task_usd, month_usd = _billing()
    print(f'[{stage}] task spend ${task_usd:.2f}, workspace month ${month_usd:.2f}', flush=True)
    if task_usd >= STOP_TASK_USD or month_usd >= STOP_MONTH_USD:
        raise SystemExit(f'stop: task ${task_usd:.2f} (limit {STOP_TASK_USD}), month ${month_usd:.2f} (limit {STOP_MONTH_USD})')


def _local_sha(rel):
    return hashlib.sha256((STUDY / rel).read_bytes()).hexdigest()


def _read_json(rel):
    return json.loads((STUDY / rel).read_text())


def _ids(split):
    return [json.loads(s)['id'] for s in (STUDY / f'data/{split}.jsonl').read_text().splitlines() if s.strip()]


def _start(stage, task):
    _guard(stage)
    call = orchestrate.spawn(dict(task, stage=stage))
    with (STUDY / 'results/stage-log.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(stage=stage, spawned_utc=datetime.now(timezone.utc).isoformat(), call=call.object_id)) + '\n')
    print(f'[{stage}] running in Modal as {call.object_id}', flush=True)


def _protocol_freeze():
    """PROTOCOL-FREEZE.json with the hashes recorded by later amendments (AMENDMENTS.md) applied in order."""
    freeze = _read_json('PROTOCOL-FREEZE.json')
    for path in sorted(STUDY.glob('PROTOCOL-FREEZE-amendment-*.json')):
        freeze['sha256'].update(_read_json(path.name)['sha256'])
    changed = [rel for rel, d in freeze['sha256'].items() if _local_sha(rel) != d]
    if changed:
        raise SystemExit(f'files differ from PROTOCOL-FREEZE.json: {changed}')
    return freeze


@app.local_entrypoint()
def first():
    freeze = _protocol_freeze()
    _start('first', dict(protocol_freeze=freeze, split='development', ids=_ids('development'), plan=_read_json('code/plans.json')['development']))


@app.local_entrypoint()
def devsites():
    freeze = _protocol_freeze()
    sel = _read_json('results/development-selection.json')
    checks = _read_json('results/first/checks.json')
    plan = dict(direction_sites=sel['direction_sites'], span_sites=sel['span_sites'])
    _start('devsites', dict(protocol_freeze=freeze, split='development', ids=_ids('development'), plan=plan,
                            means_sha256=checks['means']['sha256']))


@app.local_entrypoint()
def test():
    freeze = _read_json('FREEZE.json')
    changed = [rel for rel, d in freeze['sha256'].items() if _local_sha(rel) != d]
    if changed:
        raise SystemExit(f'files differ from FREEZE.json: {changed}')
    sel = _read_json('results/development-selection.json')
    checks = _read_json('results/first/checks.json')
    cost = _read_json('results/cost-decision.json')                  # PROTOCOL.md section 8, cost rule
    plan = dict(_read_json('code/plans.json')['test'], direction_sites=sel['direction_sites_test'], span_sites=sel['span_sites'],
                self_check=[[s['layer'], s['span']] for s in sel['span_sites']],
                direction_sweep_roles=cost['test_direction_sweep_roles'])
    _start('test', dict(freeze=freeze, split='test', ids=_ids('test'), plan=plan, means_sha256=checks['means']['sha256']))


def _read_volume(path):
    return b''.join(volume.read_file(path))


@app.local_entrypoint()
def pull(stage: str):
    """Copy a finished stage's small records to results/ (the means file stays on the Volume)."""
    status = json.loads(_read_volume(f'orchestration/{stage}.json'))
    print(json.dumps({k: status.get(k) for k in ('stage', 'status', 'started_utc', 'finished_utc', 'error')}), flush=True)
    out = STUDY / 'results'
    (out / 'orchestration').mkdir(parents=True, exist_ok=True)
    (out / 'orchestration' / f'{stage}.json').write_text(json.dumps(status, indent=2) + '\n')
    total = 0
    folders = {'first': ['first', 'development'], 'devsites': ['devsites'], 'test': ['test']}[stage]
    for folder in folders:
        try:
            entries = [e.path for e in volume.listdir(folder, recursive=True)]
        except Exception:                                                      # noqa: BLE001
            continue
        for path in entries:
            if path.endswith(('.json', '.jsonl.gz')):
                blob = _read_volume(path)
                target = out / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
                total += len(blob)
    print(f'pulled {total / 1e6:.3f} MB', flush=True)
    with (out / 'transfer-log.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, bytes_pulled=total)) + '\n')
    _billing()
