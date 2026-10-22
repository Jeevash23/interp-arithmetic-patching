"""Write FREEZE.json for Study 4 before any test inference (PROTOCOL.md section 13).

Hashes the protocol, its record and amendments, the predictions, the code (with the frozen copies, the kernel package
and kernel source), the three splits and their checks, the study model's tokenizer and model files, and every result
of the tokenizer check, the gate, the screen and the development run. Refuses to overwrite a freeze and refuses if
any test output exists.

  python -B code/freeze.py <candidate>
"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FROZEN = STUDY / 'code/frozen'
KERNEL = FROZEN / 'competence-v2/.cache/huggingface/kernels--kernels-community--finegrained-fp8/snapshots/8c178950e8710e26a2210e4e909cd02dc16c8715/build/torch-cuda'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    candidate = sys.argv[1]
    target = STUDY / 'FREEZE.json'
    if target.exists():
        raise SystemExit('FREEZE.json exists; a frozen study is never refrozen. Record changes in AMENDMENTS.md.')
    test = STUDY / 'results' / candidate / 'test'
    assert not test.exists() or not any(test.iterdir()), 'Test outputs already exist'
    assert not (STUDY / 'verify.py').exists(), 'verify.py is written after the test run'
    assert (STUDY / 'PREDICTIONS.md').exists()
    gate = json.loads((STUDY / 'results' / candidate / 'calibration/gate.json').read_text())
    assert gate['pass_check'], 'The candidate did not pass the gate'
    dev = STUDY / 'results' / candidate / 'development'
    assert (dev / 'completed-develop.json').exists() and (dev / 'analysis.json').exists()
    copied = json.loads((FROZEN / 'COPIED-FROM.json').read_text())
    deps = json.loads((FROZEN / 'competence-v2/dependency-manifest.json').read_text())
    for f in deps['added_packages']:
        assert sha(FROZEN / 'competence-v2/.packages' / f['path']) == f['sha256'], f['path']
    for f in deps['kernel_source']:
        assert sha(KERNEL / f['path']) == f['sha256'], f['path']
    for name, info in copied['files'].items():
        assert sha(FROZEN / name) == info['sha256'], name
    weights = json.loads((STUDY / f'results/weights-check-{candidate}.json').read_text())
    assert all(v['ok'] for v in weights.values()), 'model files differ from the published files'
    files = [STUDY / 'PROTOCOL.md', STUDY / 'PROTOCOL-RECORD.json', STUDY / 'AMENDMENTS.md', STUDY / 'PREDICTIONS.md']
    files += sorted((STUDY / 'code').glob('*.py')) + sorted((STUDY / 'code').glob('*.sh'))
    files += [FROZEN / name for name in copied['files']] + [FROZEN / 'COPIED-FROM.json']
    files += sorted(p for p in (FROZEN / 'competence-v2/.packages').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [KERNEL / f['path'] for f in deps['kernel_source']]
    files += sorted((STUDY / 'data').glob('*'))
    files += sorted((STUDY / 'tokenizers' / candidate).glob('*'))
    files += sorted(p for p in (STUDY / 'models' / candidate).iterdir() if p.is_file())
    files += [STUDY / 'results' / n for n in ['tokenizer-check.json', 'tokenizer-files-check.json', 'hub-files.json',
                                              'candidates.json', f'weights-check-{candidate}.json']]
    files += sorted(p for p in (STUDY / 'results' / candidate / 'calibration').iterdir() if p.is_file())
    files += sorted(p for p in dev.iterdir() if p.is_file())
    record = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), candidate=candidate,
        stage='frozen before any model output on the test pairs',
        sha256={str(p.relative_to(STUDY)): sha(p) for p in files},
        environment=dict(python=platform.python_version(), executable=sys.executable,
                         packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'accelerate', 'numpy',
                                                                             'safetensors', 'tokenizers', 'triton']}))
    target.write_text(json.dumps(record, indent=1) + '\n')
    print('Frozen at', record['frozen_at_utc'], 'files', len(record['sha256']))


if __name__ == '__main__':
    main()
