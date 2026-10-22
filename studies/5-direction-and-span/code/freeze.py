"""Freeze records of Study 5 (PROTOCOL.md section 11). Standard library only.

  python -B code/freeze.py protocol   # PROTOCOL-FREEZE.json: protocol, run code, frozen copies, inputs; before the first cloud stage
  python -B code/freeze.py amendment-N FILE...   # PROTOCOL-FREEZE-amendment-N.json: current hashes of the files amendment N changed
  python -B code/freeze.py test       # FREEZE.json: the same plus PREDICTIONS.md, the analysis code and the development records;
                                      # before any test output
Each refuses to overwrite an existing record.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = ['PROTOCOL.md', 'code/study5.py', 'code/plans.json', 'code/modal_app.py', 'code/prepare.py', 'code/freeze.py',
                  'code/stats.py', 'code/COPIED-FROM-STUDY3.json', 'data/development.jsonl', 'data/test.jsonl', 'data/fitting.jsonl',
                  'data/reproduction-reference.json', 'data/reproduction-vectors.pt', 'data/data-manifest.json']
TEST_FILES = ['PREDICTIONS.md', 'code/analyze.py', 'code/cost_rule.py', 'results/development-selection.json',
              'results/first/checks.json', 'results/development-analysis.json', 'results/cost-decision.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_code():
    return sorted(str(p.relative_to(STUDY)) for p in (STUDY / 'code/frozen').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and '.cache/triton' not in str(p))


def main():
    kind = sys.argv[1]
    files = PROTOCOL_FILES + frozen_code()
    target = STUDY / ('PROTOCOL-FREEZE.json' if kind == 'protocol' else 'FREEZE.json')
    if kind == 'test':
        files += TEST_FILES + sorted(str(p.relative_to(STUDY)) for p in (STUDY / 'results').glob('development*/*.jsonl.gz'))
        files += sorted(str(p.relative_to(STUDY)) for p in (STUDY / 'results/devsites').glob('*.jsonl.gz'))
        if (STUDY / 'AMENDMENTS.md').exists():
            files.append('AMENDMENTS.md')
        files += sorted(p.name for p in STUDY.glob('PROTOCOL-FREEZE*.json'))
    if kind.startswith('amendment-'):
        # An amendment record: the current hashes of the files an amendment changed (given after the kind).
        target = STUDY / f'PROTOCOL-FREEZE-{kind}.json'
        files = sys.argv[2:]
    if target.exists():
        raise SystemExit(f'{target.name} exists; not overwritten')
    record = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), kind=kind,
                  stage={'protocol': 'before the first cloud stage', 'test': 'before any test output'}.get(kind, 'amendment record'),
                  sha256={f: sha(STUDY / f) for f in files})
    target.write_text(json.dumps(record, indent=1) + '\n')
    print(target.name, record['frozen_at_utc'], len(files), 'files')


if __name__ == '__main__':
    main()
