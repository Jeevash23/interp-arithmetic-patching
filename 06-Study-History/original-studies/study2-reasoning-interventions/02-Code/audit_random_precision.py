from pathlib import Path
import numpy as np,torch
from ai04.engine import Engine
from ai04.common import ROOT,readj,writej,digest,utc
out=ROOT/'04-Results/audits/random-operator-v2';out.mkdir(exist_ok=True)
checks=[]
for path in sorted((ROOT/'04-Results/audits/qwen06-fp32-v1').glob('*.npz')):
 with np.load(path) as f:
  a=torch.from_numpy(f['recipient']);b=torch.from_numpy(f['source'])
  for seed in range(20):
   w=Engine.random_source(None,a,b,943000+seed)
   want=np.linalg.norm(b.numpy().astype(float)-a.numpy());got=np.linalg.norm(w.numpy().astype(float)-a.numpy());err=abs(want-got)
   checks.append(dict(case=path.name,seed=943000+seed,target_norm=want,written_norm=got,error=err,passed=bool(err<=.0001)))
old=readj(ROOT/'04-Results/audits/qwen06-fp32-v1/checks.json')
r=dict(complete=True,passed=old['passed'] and all(c['passed'] for c in checks),original_gate='qwen06-fp32-v1',new_operator_checks=checks,max_error=max(c['error'] for c in checks),created_utc=utc(),engine_hash=digest((ROOT/'02-Code/ai04/engine.py').read_bytes()))
writej(out/'checks.json',r);print('random operator checks',len(checks),'max',r['max_error'],'passed',r['passed'])
