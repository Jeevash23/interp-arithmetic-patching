import time,json,numpy as np
from ai04.common import ROOT,readj,writej,Journal,digest,utc
from second_checkpoint import SecondEngine
p=ROOT/'04-Results/audits/deepseek-cpu-v1';p.mkdir(exist_ok=True);j=Journal(p/'records.jsonl');seed=948101;checks=[]
inputs={'groups':['audit-000','audit-001'],'layers':[13,20,26],'boundaries':['first_z','label_z'],'steps':12,'seed':seed,'tolerance':.0005,'dtype':'float32','device':'cpu','source':{f:digest((ROOT/f).read_bytes()) for f in ['02-Code/second_checkpoint.py','02-Code/audit_second_checkpoint.py','02-Code/ai04/engine.py','05-Notes/protocols/SECOND-CHECKPOINT-QUALIFICATION-v2.md']},'model':readj(ROOT/'06-Study-History/environment/deepseek-model-files.json')}
freeze=p/'freeze.json'
if freeze.exists():assert readj(freeze)['identity']==inputs
else:writej(freeze,dict(identity=inputs,created_utc=utc()))
e=SecondEngine()
def check(name,value,ok,**kw):
 checks.append(dict(name=name,value=value,passed=bool(ok),**kw));writej(p/'checks.json',dict(complete=False,passed=all(c['passed'] for c in checks),checks=checks));print(name,value,ok,flush=True)
for g in readj(ROOT/'03-Data/generated/audit.json')[:2]:
 base={}
 for role in ['recipient','donor']:
  key=g['id']+'/'+role;ids=e.encode(g[role+'_prompt'])
  if key in j.rows:r=j.rows[key]
  else:r=j.add(dict(key=key,**e.generate(ids,seed)))
  base[role]=r;n=e.native(ids,seed);check(key+'/native',n==r['output_ids'],n==r['output_ids'])
  raw=e.forward(ids,use_cache=False).logits
  for rule in ['noop','identity']:
   with e.patch(20,len(ids)-1,rule=rule):a=e.forward(ids,use_cache=False).logits
   err=(raw-a).abs().max().item();check(key+'/'+rule,err,err<=.0005)
 r=base['recipient'];d=base['donor'];rc=e.cuts(r,g['recipient']['z'],g['recipient']['y']);dc=e.cuts(d,g['donor']['z'],g['donor']['y']);writej(p/(g['id']+'-cuts.json'),dict(recipient=rc,donor=dc));check(g['id']+'/sites',list(rc),all(x in rc and x in dc for x in ['first_z','label_z']) and 'final' in rc)
 for b in ['first_z','label_z']:
  if b not in rc or b not in dc:continue
  pre=rc[b]['prefix_ids'];pos=len(pre)-1;vs=e.capture(dc[b]['prefix_ids'],[13,20,26])
  for rule in ['noop','identity']:
   a=e.generate(r['prompt_ids'],seed,pre,26,rule=rule);check(g['id']+'/'+b+'/'+rule,a['output_ids']==r['output_ids'],a['output_ids']==r['output_ids'])
  for layer in [13,20,26]:
   a,ta=e.generate(r['prompt_ids'],seed,pre,layer,vs[layer],limit=12,trace=True);b0,tb=e.generate(r['prompt_ids'],seed,pre,layer,vs[layer],cache=False,limit=12,trace=True);same=a['output_ids']==b0['output_ids'];err=max((x-y).abs().max().item() for x,y in zip(ta,tb)) if same else 1e9
   check(f'{g["id"]}/{b}/{layer}/cache',err,same and err<=.0005,tokens_equal=same)
   check(f'{g["id"]}/{b}/{layer}/write',a['write'],a['write']['delta_norm']>0 and a['write']['other_positions_max']==0 and a['write']['write_error']==0)
   if 'final' in rc:
    vals=[g['recipient']['y'],g['hypotheses']['counterfactual']];s1=e.score(rc['final']['prefix_ids'],vals,layer,vs[layer],pos);s2=e.score(rc['final']['prefix_ids'],vals,layer,vs[layer],pos,cached=True);err=max(abs(x-y) for k in s1 for x,y in zip(s1[k]['token_logprobs'],s2[k]['token_logprobs']));check(f'{g["id"]}/{b}/{layer}/scores',err,err<=.0005)
writej(p/'checks.json',dict(complete=True,passed=all(c['passed'] for c in checks),checks=checks,finished_utc=utc()))
