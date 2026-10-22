"""Record and check the published files of a candidate at its pinned revision (needs network for `record`).

  python -B code/hub_files.py record             # results/hub-files.json: size, git blob id and LFS SHA-256 of every file
  python -B code/hub_files.py check-tokenizers   # the tokenizer files in tokenizers/<candidate>/ against that record
  python -B code/hub_files.py check-weights CAND # every file in models/<candidate>/ against that record (SHA-256)
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
from candidates import CANDIDATES, ORDER  # noqa: E402

RECORD = STUDY / 'results/hub-files.json'


def git_blob_sha1(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def record():
    out = {}
    for name in ORDER:
        cfg = CANDIDATES[name]
        url = f"https://huggingface.co/api/models/{cfg['repository']}/tree/{cfg['revision']}?recursive=true"
        with urllib.request.urlopen(url, timeout=60) as r:
            tree = json.load(r)
        out[name] = dict(repository=cfg['repository'], revision=cfg['revision'], files={
            e['path']: dict(size=e.get('size'), git_oid=e.get('oid'), lfs_sha256=(e.get('lfs') or {}).get('oid'))
            for e in tree if e.get('type') == 'file'})
    RECORD.write_text(json.dumps(out, indent=1) + '\n')
    print({n: len(v['files']) for n, v in out.items()})


def check_file(entry, path):
    data_size = path.stat().st_size
    if entry['size'] is not None and data_size != entry['size']:
        return False
    if entry['lfs_sha256']:
        return sha256_file(path) == entry['lfs_sha256']
    return git_blob_sha1(path.read_bytes()) == entry['git_oid']


def check_tokenizers():
    rec = json.loads(RECORD.read_text())
    result = {}
    for name in ORDER:
        folder = STUDY / 'tokenizers' / name
        result[name] = {f: check_file(rec[name]['files'][f], folder / f) for f in CANDIDATES[name]['tokenizer_files']}
    print(json.dumps(result, indent=1))
    return result


def check_weights(name):
    rec = json.loads(RECORD.read_text())[name]
    folder = STUDY / 'models' / name
    result = {}
    for p in sorted(folder.iterdir()):
        if p.is_file():
            entry = rec['files'].get(p.name)
            result[p.name] = dict(ok=bool(entry) and check_file(entry, p), sha256=sha256_file(p), bytes=p.stat().st_size)
    print(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'record':
        record()
    elif cmd == 'check-tokenizers':
        res = check_tokenizers()
        (STUDY / 'results/tokenizer-files-check.json').write_text(json.dumps(res, indent=1) + '\n')
    elif cmd == 'check-weights':
        res = check_weights(sys.argv[2])
        (STUDY / f'results/weights-check-{sys.argv[2]}.json').write_text(json.dumps(res, indent=1) + '\n')
