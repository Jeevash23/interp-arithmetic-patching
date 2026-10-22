"""Fetch only pinned public files, with disk headroom and post-download hashes."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='0'
os.environ['HF_HUB_DISABLE_XET']='0'
os.environ['HF_XET_CACHE']=str(HERE/'.cache/xet')
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
os.environ['HF_HUB_CACHE']=str(HERE/'.cache/huggingface')
from huggingface_hub import HfApi,snapshot_download

CANDIDATES={
    'qwen4b':('Qwen/Qwen3-4B-Instruct-2507-FP8','8591804019c8b22094c3b5b4454e0edc05dffc98'),
    'qwen25':('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306')}
args=argparse.ArgumentParser();args.add_argument('candidate',choices=CANDIDATES);a=args.parse_args()
repo,revision=CANDIDATES[a.candidate]
if a.candidate=='qwen4b':
    assert json.loads((HERE/'kernel-inspection.json').read_text())['pass_check']
meta=HfApi(token=False).model_info(repo,revision=revision,files_metadata=True)
target=HERE/'models'/a.candidate
allowed=[f.rfilename for f in meta.siblings if '/' not in f.rfilename and (f.rfilename.endswith(('.json','.safetensors','.txt','.model')) or f.rfilename in ['README.md','LICENSE'])]
remaining=sum(f.size or 0 for f in meta.siblings if f.rfilename in allowed and not (target/f.rfilename).exists())
free=shutil.disk_usage(HERE).free
assert free-remaining>1024**3, f'Insufficient disk headroom: {free=} {remaining=}'
print(json.dumps(dict(repository=repo,revision=revision,download_bytes=remaining,free_disk_bytes=free)),flush=True)
snapshot_download(repo_id=repo,revision=revision,local_dir=target,allow_patterns=allowed,max_workers=2,token=False)
files=[]
for p in sorted(target.iterdir()):
    if p.is_file():
        with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        files.append(dict(name=p.name,bytes=p.stat().st_size,sha256=digest))
out=dict(repository=repo,revision=revision,files=files,client_date='2026-10-01')
(HERE/f'model-manifest-{a.candidate}.json').write_text(json.dumps(out,indent=2)+'\n')
print('Verified downloaded files:',len(files),flush=True)
