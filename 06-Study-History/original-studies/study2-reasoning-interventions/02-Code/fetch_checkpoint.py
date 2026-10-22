from pathlib import Path
from urllib.request import urlopen,Request
import json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
repo='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
def get(u):return urlopen(Request(u,headers={'User-Agent':'AI04-research'}),timeout=120)
meta=json.load(get('https://huggingface.co/api/models/'+repo+'?blobs=true'));revision=meta['sha']
p=ROOT/'03-Data/raw/models/DeepSeek-R1-Distill-Qwen-1.5B';p.mkdir(parents=True,exist_ok=True)
records=[]
for item in meta['siblings']:
 name=item['rfilename']
 if '/' in name or not (name.endswith('.json') or name.endswith('.safetensors') or name in ['README.md','LICENSE','merges.txt','vocab.json']):continue
 f=p/name
 if not f.exists():
  with get('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name) as source, (p/(name+'.partial')).open('wb') as out:
   while chunk:=source.read(8*1024*1024):out.write(chunk)
  (p/(name+'.partial')).rename(f)
 h=hashlib.sha256()
 with f.open('rb') as inp:
  while chunk:=inp.read(8*1024*1024):h.update(chunk)
 sha=h.hexdigest();expected=item.get('lfs',{}).get('sha256')
 if expected and sha!=expected:raise RuntimeError('weight hash mismatch')
 records.append(dict(name=name,sha256=sha,bytes=f.stat().st_size,expected_lfs_sha256=expected));print(name,f.stat().st_size,flush=True)
(ROOT/'06-Study-History/environment/deepseek-model-files.json').write_text(json.dumps(dict(repo=repo,revision=revision,files=records),indent=2))
# Independently compare reused Qwen weight bytes with the recorded public revision.
q=json.load(get('https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/c1899de289a04d12100db370d81485cdf75e47ca?blobs=true'))
old=json.loads((ROOT/'06-Study-History/environment/model-files.json').read_text())
for item in q['siblings']:
 if item.get('lfs') and item['rfilename'] in old['files']:
  assert old['files'][item['rfilename']]['sha256']==item['lfs']['sha256']
(ROOT/'06-Study-History/environment/qwen-public-revision.json').write_text(json.dumps(q,indent=2))
print('Checkpoints verified')
