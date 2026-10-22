"""Modal app for Study 6 (OLMo-2-1124-7B-Instruct, bfloat16, one L4). From the study folder:

    ~/.local/bin/modal run --detach code/modal_app.py::gate       # weights check, loading, instrumentation, competence gate
    ~/.local/bin/modal run --detach code/modal_app.py::develop    # development screen, then the development conditions
    ~/.local/bin/modal run --detach code/modal_app.py::test       # after FREEZE.json
    ~/.local/bin/modal run code/modal_app.py::pull --stage gate

Study 4's runner, `code/study4.py`, runs unchanged in the container (in its 25-minute chunks, back to back); its
results folder is a link to the Modal Volume `ai0204-study6`, committed after every chunk. The weights download inside
the image build at the pinned revision, each file checked against the published hash; they never reach the local machine.
Only small records come back (journals gzipped). The billing guard is Study 5's: no stage starts once the ai0204-*
apps have spent $8 or the workspace's month total has reached $27.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

CANDIDATE = 'olmo2-7b'
REPO, REVISION = 'allenai/OLMo-2-1124-7B-Instruct', '470b1fba1ae01581f270116362ee4aa1b97f4c84'
REMOTE = '/root/s6'
MODEL_DIR = f'{REMOTE}/models/{CANDIDATE}'
WEIGHT_FILES = ['config.json', 'generation_config.json', 'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
                'model-00003-of-00003.safetensors', 'model.safetensors.index.json', 'special_tokens_map.json', 'tokenizer.json',
                'tokenizer_config.json', 'vocab.json', 'merges.txt']
STOP_TASK_USD, STOP_MONTH_USD = 8.0, 27.0
MOUNT = ['PROTOCOL.md', 'PROTOCOL-RECORD.json', 'STUDY4-PROTOCOL.md', 'STUDY4-AMENDMENTS.md', 'AMENDMENTS.md', 'PREDICTIONS.md',
         'FREEZE.json', 'verify.py.draft']


def download_weights(hub_files: dict):
    from huggingface_hub import hf_hub_download
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    record = {}
    for f in WEIGHT_FILES:
        hf_hub_download(REPO, f, revision=REVISION, local_dir=MODEL_DIR)
        data_path = Path(MODEL_DIR) / f
        want = hub_files[f]
        h256, h1 = hashlib.sha256(), hashlib.sha1()
        size = data_path.stat().st_size
        h1.update(b'blob %d\0' % size)
        with open(data_path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b''):
                h256.update(chunk)
                h1.update(chunk)
        ok = size == want['size'] and (h256.hexdigest() == want['lfs_sha256'] if want['lfs_sha256'] else h1.hexdigest() == want['git_oid'])
        assert ok, (f, size, h256.hexdigest(), h1.hexdigest())
        record[f] = dict(bytes=size, sha256=h256.hexdigest(), git_oid=h1.hexdigest(), ok=ok)
    cache = Path(MODEL_DIR) / '.cache'
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
    Path(f'/root/weights-check-{CANDIDATE}.json').write_text(json.dumps(record, indent=1))


if modal.is_local():
    STUDY = Path(__file__).resolve().parents[1]
    AI01 = STUDY.parents[1]   # release: pyproject.toml and uv.lock are at the release root
    HUB = json.loads((STUDY / 'results/hub-files.json').read_text())['candidates'][CANDIDATE]['files']
else:
    STUDY = AI01 = Path('/nonexistent')
    HUB = {}

image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(uv_project_dir=str(AI01), extras=['experiment'], frozen=True)
         .run_function(download_weights, kwargs=dict(hub_files=HUB))
         .env({'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
         .add_local_dir(str(STUDY / 'code'), f'{REMOTE}/code', ignore=['**/__pycache__/**', 'frozen/competence-v2/.cache/triton/**'])
         .add_local_dir(str(STUDY / 'data'), f'{REMOTE}/data')
         .add_local_dir(str(STUDY / 'tokenizers'), f'{REMOTE}/tokenizers'))
for name in MOUNT:
    if (STUDY / name).exists():
        image = image.add_local_file(str(STUDY / name), f'{REMOTE}/{name}')
app = modal.App('ai0204-study6', image=image, tags={'project': 'AI-02-04', 'study': '6-olmo2-7b-multidigit'})
volume = modal.Volume.from_name('ai0204-study6', create_if_missing=True)
VOL = '/vol'


# ======================================================================================== remote side
def _link_results():
    """study4.py writes to STUDY/results; make that the Volume (the protocol's local results files are copied in once)."""
    target = Path(VOL) / 'results'
    target.mkdir(parents=True, exist_ok=True)
    link = Path(REMOTE) / 'results'
    if not link.exists():
        os.symlink(target, link)
    weights = Path(f'/root/weights-check-{CANDIDATE}.json')
    dest = target / f'weights-check-{CANDIDATE}.json'
    if not dest.exists():
        dest.write_text(weights.read_text())


def _run_phase(phase, freeze=None):
    """study4.py <phase> in chunks; commits the Volume after every chunk. Returns the last exit status."""
    _link_results()
    for name, text in (freeze or {}).items():                  # small local records the runner reads (candidates.json etc.)
        p = Path(VOL) / 'results' / name
        if not p.exists():
            p.write_text(text)
    volume.commit()
    log = Path(VOL) / 'logs' / f'{phase}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    status = None
    for chunk in range(40):
        with log.open('a') as fh:
            fh.write(f'--- chunk {chunk} {datetime.now(timezone.utc).isoformat()}\n')
            fh.flush()
            proc = subprocess.run([sys.executable, '-B', f'{REMOTE}/code/study4.py', phase, '--candidate', CANDIDATE,
                                   '--max-minutes', '25'], cwd=REMOTE, stdout=fh, stderr=subprocess.STDOUT)
        status = proc.returncode
        volume.commit()
        if status != 3:
            break
    return status


@app.function(gpu='L4', cpu=2.0, memory=24576, timeout=6 * 3600, scaledown_window=10, volumes={VOL: volume}, max_containers=1)
def gpu_job(task: dict) -> str:
    statuses = {}
    for phase in task['phases']:
        statuses[phase] = _run_phase(phase, task.get('records'))
        if statuses[phase] != 0:
            break
        if phase == 'gate':
            gate = json.loads((Path(VOL) / 'results' / CANDIDATE / 'calibration/gate.json').read_text())
            if not gate['pass_check']:
                break
    return json.dumps(statuses)


@app.function(cpu=1.0, memory=4096, timeout=1800, volumes={VOL: volume}, max_containers=1)
def pack(folder: str) -> bytes:
    """Gzipped copies of a results folder's small files (JSON and JSON lines; no vector files)."""
    volume.reload()
    out = {}
    base = Path(VOL) / 'results'
    for p in sorted((base / folder).rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.jsonl'):
            out[str(p.relative_to(base))] = p.read_bytes()
    for p in sorted(base.glob('*.json')):
        out[p.name] = p.read_bytes()
    for p in sorted((Path(VOL) / 'logs').glob('*.log')):
        out['../logs-remote/' + p.name] = p.read_bytes()
    return gzip.compress(json.dumps({k: v.decode() for k, v in out.items()}).encode())


@app.function(cpu=0.25, memory=1024, timeout=8 * 3600, volumes={VOL: volume}, max_containers=1)
def orchestrate(task: dict) -> str:
    stage = task['stage']
    summary = dict(stage=stage, started_utc=datetime.now(timezone.utc).isoformat(), status='running')
    target = Path(VOL) / 'orchestration' / f'{stage}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    volume.commit()
    try:
        summary['phase_status'] = json.loads(gpu_job.remote(task))
        summary['status'] = 'finished'
    except Exception as error:                                                   # noqa: BLE001
        summary.update(status='failed', error=repr(error))
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary['status']


# ======================================================================================== local side
def _billing():
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
    with (STUDY / 'results/billing-log.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), task_usd=round(task_usd, 4), month_usd=round(month_usd, 4))) + '\n')
    return task_usd, month_usd


def _start(stage, task):
    task_usd, month_usd = _billing()
    print(f'[{stage}] task spend ${task_usd:.2f}, workspace month ${month_usd:.2f}', flush=True)
    if task_usd >= STOP_TASK_USD or month_usd >= STOP_MONTH_USD:
        raise SystemExit('stop: spending limit reached')
    call = orchestrate.spawn(dict(task, stage=stage))
    with (STUDY / 'results/stage-log.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(stage=stage, spawned_utc=datetime.now(timezone.utc).isoformat(), call=call.object_id)) + '\n')
    print(f'[{stage}] running in Modal as {call.object_id}', flush=True)


def _records():
    names = ['candidates.json', 'tokenizer-check.json', 'tokenizer-files-check.json', 'hub-files.json']
    return {n: (STUDY / 'results' / n).read_text() for n in names if (STUDY / 'results' / n).exists()}


def _check_record():
    rec = json.loads((STUDY / 'PROTOCOL-RECORD.json').read_text())
    got = hashlib.sha256((STUDY / 'PROTOCOL.md').read_bytes()).hexdigest()
    if got != rec['versions'][-1]['sha256']:
        raise SystemExit('PROTOCOL.md differs from its last record')


@app.local_entrypoint()
def gate():
    _check_record()
    _start('gate', dict(phases=['gate'], records=_records()))


@app.local_entrypoint()
def develop():
    _check_record()
    _start('develop', dict(phases=['screen', 'develop']))


@app.local_entrypoint()
def test():
    freeze = json.loads((STUDY / 'FREEZE.json').read_text())
    changed = [rel for rel, d in freeze['sha256'].items()
               if (STUDY / rel).exists() and hashlib.sha256((STUDY / rel).read_bytes()).hexdigest() != d]
    if changed:
        raise SystemExit(f'files differ from FREEZE.json: {changed}')
    _start('test', dict(phases=['test']))


@app.local_entrypoint()
def pull(stage: str):
    status = json.loads(b''.join(volume.read_file(f'orchestration/{stage}.json')))
    print(json.dumps(status), flush=True)
    folder = {'gate': f'{CANDIDATE}/calibration', 'develop': f'{CANDIDATE}/development', 'test': f'{CANDIDATE}/test'}[stage]
    blob = pack.remote(folder)
    files = json.loads(gzip.decompress(blob))
    for rel, text in files.items():
        target = (STUDY / 'results' / rel).resolve() if not rel.startswith('../') else (STUDY / rel[3:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (STUDY / 'results/orchestration').mkdir(parents=True, exist_ok=True)
    (STUDY / f'results/orchestration/{stage}.json').write_text(json.dumps(status, indent=2) + '\n')
    print(f'pulled {len(blob) / 1e6:.3f} MB (gzipped)', flush=True)
    with (STUDY / 'results/transfer-log.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, bytes_pulled=len(blob))) + '\n')
    _billing()
