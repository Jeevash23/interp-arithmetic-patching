from ai04.common import ROOT,readj,writej,freeze,Journal,digest
from ai04.engine import Engine,DEFAULT
import torch,time,numpy as np
RUN='audits/qwen06-fp32-v1'
config=dict(engine=DEFAULT,seed=940101,groups=4,layers=[13,20,26],boundaries=['first_z','label_z'],tolerance=.0005,cache_steps=24)
out,h=freeze(RUN,config,['05-Notes/protocols/QUALIFICATION-v1.md','03-Data/generated/audit.json'])
j=Journal(out/'records.jsonl');e=Engine();checks=[]
def check(name,value,ok,**kw):
 r=dict(name=name,value=value,passed=bool(ok),**kw);checks.append(r);writej(out/'checks.json',dict(checks=checks,passed=all(c['passed'] for c in checks),complete=False));print(name,value,ok,flush=True)
for g in readj(ROOT/'03-Data/generated/audit.json'):
 baselines={}
 for role in ['recipient','donor']:
  key=g['id']+'/'+role;ids=e.encode(g[role+'_prompt'])
  if key in j.rows:r=j.rows[key]
  else:r=j.add(dict(key=key,group=g['id'],role=role,**e.generate(ids,config['seed'])))
  baselines[role]=r
  native=e.native(ids,config['seed']);check(key+'/native',native==r['output_ids'],native==r['output_ids'])
  raw=e.forward(ids,use_cache=False).logits
  for rule in ['noop','identity']:
   with e.patch(20,len(ids)-1,rule=rule):p=e.forward(ids,use_cache=False).logits
   err=(p-raw).abs().max().item();check(key+'/'+rule,err,err<=config['tolerance'])
 r=baselines['recipient'];d=baselines['donor'];rc=e.cuts(r,g['recipient']['z'],g['recipient']['y']);dc=e.cuts(d,g['donor']['z'],g['donor']['y'])
 writej(out/(g['id']+'-cuts.json'),dict(recipient=rc,donor=dc))
 check(g['id']+'/sites',list(rc),all(x in rc and x in dc for x in config['boundaries']) and 'final' in rc)
 for boundary in config['boundaries']:
  if boundary not in rc or boundary not in dc:continue
  prefix=rc[boundary]['prefix_ids'];position=len(prefix)-1;src=e.capture(dc[boundary]['prefix_ids'],config['layers']);base=e.capture(prefix,config['layers'])
  for rule in ['noop','identity']:
   v=e.generate(r['prompt_ids'],config['seed'],prefix,26,base[26],rule)
   jkey=f'{g["id"]}/{boundary}/{rule}'
   if jkey not in j.rows:j.add(dict(key=jkey,group=g['id'],**v))
   check(jkey,v['output_ids']==r['output_ids'],v['output_ids']==r['output_ids'])
  raw=e.forward(prefix,use_cache=False).logits
  for layer in config['layers']:
   stats={}
   with e.patch(layer,position,src[layer],stats=stats):p=e.forward(prefix,use_cache=False).logits
   err=(p-raw).abs().max().item();check(f'{g["id"]}/{boundary}/{layer}/positive',err,err>.001 and stats['delta_norm']>0 and stats['other_positions_max']==0 and stats['write_error']==0,write=stats)
   a,ta=e.generate(r['prompt_ids'],config['seed'],prefix,layer,src[layer],limit=config['cache_steps'],trace=True)
   b,tb=e.generate(r['prompt_ids'],config['seed'],prefix,layer,src[layer],cache=False,limit=config['cache_steps'],trace=True)
   same=a['output_ids']==b['output_ids'];err=max((x-y).abs().max().item() for x,y in zip(ta,tb)) if same else float('inf')
   check(f'{g["id"]}/{boundary}/{layer}/cache',err,same and err<=config['tolerance'],tokens_equal=same,cached_tokens=a['fresh_output_ids'],recomputed_tokens=b['fresh_output_ids'])
   if 'final' in rc:
    values=[g['recipient']['y'],g['hypotheses']['counterfactual']]
    s1=e.score(rc['final']['prefix_ids'],values,layer,src[layer],position,cached=False);s2=e.score(rc['final']['prefix_ids'],values,layer,src[layer],position,cached=True)
    err=max(abs(x-y) for k in s1 for x,y in zip(s1[k]['token_logprobs'],s2[k]['token_logprobs']))
    check(f'{g["id"]}/{boundary}/{layer}/score',err,err<=config['tolerance'],scores_full=s1,scores_cache=s2)
   np.savez(out/f'{g["id"]}-{boundary}-{layer}.npz',source=src[layer].numpy(),recipient=base[layer].numpy())
writej(out/'checks.json',dict(checks=checks,passed=all(c['passed'] for c in checks),complete=True,peak_memory_bytes=torch.cuda.max_memory_allocated(),freeze_hash=h))
print('FINAL',all(c['passed'] for c in checks),flush=True)
