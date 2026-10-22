"""Write FREEZE.json: SHA-256 of the protocol, predictions, code, test data, model files and kernel files, with the UTC time.

Refuses to overwrite an existing freeze. Before writing, it checks that
- the frozen pipeline copies are byte-identical to the original AI-02 files (code/frozen/COPIED-FROM.json);
- the model files match the 1 October model manifest;
- the kernel package and kernel source match the development dependency manifest;
- the development reproduction check passed and the test results folders are empty.

  python -B code/freeze.py
"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[1]
FROZEN = STUDY / 'code/frozen'
KERNEL = FROZEN / 'competence-v2/.cache/huggingface/kernels--kernels-community--finegrained-fp8/snapshots/8c178950e8710e26a2210e4e909cd02dc16c8715/build/torch-cuda'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return str(Path(p).relative_to(STUDY))


def main():
    target = STUDY / 'FREEZE.json'
    if target.exists():
        raise SystemExit('FREEZE.json exists; a frozen study is never refrozen. Record changes in AMENDMENTS.md.')
    for wording in ['machine', 'box']:
        d = STUDY / 'results' / wording
        assert not d.exists() or not any(d.iterdir()), f'Test results already exist in {d}'
    copied = json.loads((FROZEN / 'COPIED-FROM.json').read_text())
    for name, info in copied['files'].items():
        assert sha(FROZEN / name) == info['sha256'] == sha(PROJECT / info['source']), name
    manifest = json.loads((FROZEN / 'competence-v2/model-manifest-qwen4b.json').read_text())
    model_dir = STUDY / 'models/qwen4b'
    for f in manifest['files']:
        p = model_dir / f['name']
        assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256'], f['name']
    deps = json.loads((FROZEN / 'competence-v2/dependency-manifest.json').read_text())
    for f in deps['added_packages']:
        assert sha(FROZEN / 'competence-v2/.packages' / f['path']) == f['sha256'], f['path']
    for f in deps['kernel_source']:
        assert sha(KERNEL / f['path']) == f['sha256'], f['path']
    repro = json.loads((STUDY / 'results/dev-reproduction.json').read_text())
    assert repro['all_equal'], 'Development reproduction failed'
    assert not (STUDY / 'verify.py').exists(), 'verify.py is written after the runs'
    files = [STUDY / 'PROTOCOL.md', STUDY / 'PREDICTIONS.md']
    files += sorted((STUDY / 'code').glob('*.py')) + sorted((STUDY / 'code').glob('*.sh'))
    files += [FROZEN / name for name in copied['files']] + [FROZEN / 'COPIED-FROM.json']
    files += sorted(p for p in (FROZEN / 'competence-v2/.packages').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [KERNEL / f['path'] for f in deps['kernel_source']]
    files += sorted((STUDY / 'data').glob('*'))
    files += [model_dir / f['name'] for f in manifest['files']]
    files += [STUDY / 'results/development-reference.json', STUDY / 'results/dev-reproduction.json',
              STUDY / 'results/operating-characteristics.json']
    venv = Path(sys.executable)
    record = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='frozen before any model output on the test pairs',
        sha256={rel(p): sha(p) for p in files},
        model=dict(repository=manifest['repository'], revision=manifest['revision'],
                   weights_sha256=next(f['sha256'] for f in manifest['files'] if f['name'] == 'model.safetensors')),
        kernel=dict(repository='kernels-community/finegrained-fp8', revision=deps['kernel_revision'], kernels_package=deps['kernels_version']),
        environment=dict(python=platform.python_version(), executable=str(venv),
                         packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'accelerate', 'numpy', 'safetensors', 'tokenizers', 'triton']},
                         ai01_uv_lock_sha256=sha(PROJECT / 'uv.lock')   # release: the environment lock is this release's uv.lock),
        development_reproduction=dict(records=repro['records'], tokens_equal=repro['tokens_equal'],
                                      vectors_equal=repro['vectors_equal'], vectors_checked=repro['vectors_checked']))
    target.write_text(json.dumps(record, indent=1) + '\n')
    print('Frozen at', record['frozen_at_utc'], 'files', len(record['sha256']))


if __name__ == '__main__':
    main()
