"""Fetch the public official checkpoint with revision and file hashes."""
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='0'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi,snapshot_download
HERE=Path(__file__).resolve().parent
repo='Qwen/Qwen3-1.7B'
meta=HfApi().model_info(repo,token=False)
model=HERE/'model'
snapshot_download(repo_id=repo,revision=meta.sha,local_dir=model,allow_patterns=['*.json','*.safetensors','*.txt','*.model','README.md','LICENSE'],max_workers=2,token=False)
files=[]
for p in sorted(model.iterdir()):
    if p.is_file():
        with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha})
(HERE/'model-manifest.json').write_text(json.dumps({'repository':repo,'revision':meta.sha,'utc':datetime.now(timezone.utc).isoformat(),'files':files},indent=2)+'\n')
print(json.dumps({'revision':meta.sha,'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
