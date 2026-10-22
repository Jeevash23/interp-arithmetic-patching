import shutil
from ai04.common import ROOT,readj,writej,Journal,digest,utc
from second_checkpoint import SecondEngine

SUFFIX=' After your thinking, give exactly two plain-text lines in this format: Intermediate: <integer> followed by Final: <integer>. Do not use boxes, equations, Markdown, or other text after thinking. The first line gives z and the second gives the final answer.'
p=ROOT/'04-Results/audits/deepseek-format-v3';p.mkdir(exist_ok=True)
files=['02-Code/audit_second_format.py','02-Code/second_checkpoint.py','02-Code/ai04/engine.py','02-Code/ai04/tasks.py','05-Notes/protocols/SECOND-CHECKPOINT-FORMAT-v3.md','03-Data/generated/audit.json']
identity=dict(suffix=SUFFIX,seed=948101,source={f:digest((ROOT/f).read_bytes()) for f in files},model=readj(ROOT/'06-Study-History/environment/deepseek-model-files.json'))
if (p/'freeze.json').exists():assert readj(p/'freeze.json')['identity']==identity
else:writej(p/'freeze.json',dict(identity=identity,created_utc=utc()))
for f in files:
 dest=p/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():shutil.copy2(ROOT/f,dest)
e=SecondEngine();j=Journal(p/'records.jsonl');groups=readj(ROOT/'03-Data/generated/audit.json')[:2];checks=[]
def check(name,value,ok,**kw):
 checks.append(dict(name=name,value=value,passed=bool(ok),**kw));writej(p/'checks.json',dict(complete=False,passed=all(c['passed'] for c in checks),checks=checks));print(name,value,ok,flush=True)
for g in groups:
 for role in ['recipient','donor','supplied']:
  key=g['id']+'/'+role
  if key not in j.rows:j.add(dict(key=key,group=g['id'],role=role,**e.generate(e.encode(g[role+'_prompt']+SUFFIX),948101)))
  r=j.rows[key];expected=g['hypotheses']['counterfactual'] if role=='supplied' else g[role]['y']
  check(key+'/correct',dict(parsed=r['parsed'],stop=r['stop_reason'],expected=expected),r['stop_reason']=='eos' and r['parsed']['final']==expected)
 for role in ['recipient','donor']:
  r=j.rows[g['id']+'/'+role];cuts=e.cuts(r,g[role]['z'],g[role]['y']);writej(p/(g['id']+'-'+role+'-cuts.json'),cuts)
  check(g['id']+'/'+role+'/sites',list(cuts),all(x in cuts for x in ['first_z','label_z','final']))
if not all(c['passed'] for c in checks):
 writej(p/'checks.json',dict(complete=True,passed=False,boundary='format_or_competence_preflight',checks=checks,finished_utc=utc()))
 raise SystemExit('Preflight failed; no mechanism inference on this checkpoint')
for g in groups:
 r=j.rows[g['id']+'/recipient'];d=j.rows[g['id']+'/donor'];rc=e.cuts(r,g['recipient']['z'],g['recipient']['y']);dc=e.cuts(d,g['donor']['z'],g['donor']['y'])
 for role in ['recipient','donor']:
  rr=j.rows[g['id']+'/'+role];n=e.native(rr['prompt_ids'],948101);check(g['id']+'/'+role+'/native',n==rr['output_ids'],n==rr['output_ids'])
 for boundary in ['first_z','label_z']:
  pre=rc[boundary]['prefix_ids'];pos=len(pre)-1;src=e.capture(dc[boundary]['prefix_ids'],[13,20,26]);raw=e.forward(pre,use_cache=False).logits
  for rule in ['noop','identity']:
   a=e.generate(r['prompt_ids'],948101,pre,26,rule=rule);check(g['id']+'/'+boundary+'/'+rule,a['output_ids']==r['output_ids'],a['output_ids']==r['output_ids'])
  for layer in [13,20,26]:
   stats={}
   with e.patch(layer,pos,src[layer],stats=stats):patched=e.forward(pre,use_cache=False).logits
   delta=(patched-raw).abs().max().item();check(f'{g["id"]}/{boundary}/{layer}/positive',delta,delta>.001 and stats['delta_norm']>0 and stats['other_positions_max']==0 and stats['write_error']==0,write=stats)
   a,ta=e.generate(r['prompt_ids'],948101,pre,layer,src[layer],limit=12,trace=True);b,tb=e.generate(r['prompt_ids'],948101,pre,layer,src[layer],cache=False,limit=12,trace=True)
   same=a['output_ids']==b['output_ids'];err=max((x-y).abs().max().item() for x,y in zip(ta,tb)) if same else 1e9
   check(f'{g["id"]}/{boundary}/{layer}/cache',err,same and err<=.0005)
   vals=[g['recipient']['y'],g['hypotheses']['counterfactual']];s1=e.score(rc['final']['prefix_ids'],vals,layer,src[layer],pos);s2=e.score(rc['final']['prefix_ids'],vals,layer,src[layer],pos,cached=True)
   err=max(abs(x-y) for k in s1 for x,y in zip(s1[k]['token_logprobs'],s2[k]['token_logprobs']));check(f'{g["id"]}/{boundary}/{layer}/scores',err,err<=.0005)
writej(p/'checks.json',dict(complete=True,passed=all(c['passed'] for c in checks),checks=checks,finished_utc=utc()))
