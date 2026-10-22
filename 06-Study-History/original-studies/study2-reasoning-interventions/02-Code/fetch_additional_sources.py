from fetch_sources import fetch,ROOT
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen,Request
import json,hashlib,datetime
sources={'readout-shortcut':'https://arxiv.org/html/2605.22870','future-probes':'https://arxiv.org/html/2606.11172v1','cache-evidence':'https://arxiv.org/html/2608.01631v1','deepseek-model':'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/README.md'}
res=list(ThreadPoolExecutor(max_workers=3).map(fetch,sources.items()))
for repo in ['interp-reasoning/thought-branches','YIDING4869/scit-emnlp-2026','ericb-goodfire/forking-fast']:
 try:
  base='https://api.github.com/repos/'+repo
  def get(u):return json.load(urlopen(Request(u,headers={'User-Agent':'AI04-public-research'}),timeout=30))
  r=get(base);commit=get(base+'/commits/'+r['default_branch'])['sha'];lic=r.get('license')
  res.append(dict(repo=repo,commit=commit,license=lic,default_branch=r['default_branch'],code_reproduced=False))
 except Exception as e:res.append(dict(repo=repo,error=repr(e)))
(ROOT/'01-Papers/sources/additional-manifest.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
