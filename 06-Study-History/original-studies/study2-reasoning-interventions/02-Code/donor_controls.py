from pathlib import Path
import numpy as np,shutil,torch
from ai04.common import ROOT,readj,writej,Journal,digest,utc
from ai04.engine import Engine
from ai04.tasks import program,prompt,op,classify
RUN='development/qwen06-donor-controls-v1';out=ROOT/'04-Results'/RUN;out.mkdir(exist_ok=True)
basepath=ROOT/'04-Results/development/qwen06-development-v2/records.jsonl';old=Journal(basepath).rows
assert readj(basepath.parent/'completion.json')['complete'] and readj(basepath.parent/'competence.json')['passed']
groups=readj(ROOT/'03-Data/generated/development.json')[:8];e=Engine();j=Journal(out/'records.jsonl');seed=944001
identity=dict(groups=[g['id'] for g in groups],seed=seed,sites=['first_z-L26','label_z-L26'],variants=['same_z_other_answer','same_z_decomposition','different_z_same_answer','same_leading_wide'],base_journal_hash=digest(basepath.read_bytes()),code={f:digest((ROOT/f).read_bytes()) for f in ['02-Code/donor_controls.py','02-Code/ai04/engine.py','02-Code/ai04/tasks.py','05-Notes/protocols/DONOR-CONTROLS-v1.md']})
identity['inputs']={f:digest((ROOT/f).read_bytes()) for f in ['02-Code/ai04/common.py','03-Data/generated/development.json','06-Study-History/environment/model-files.json','04-Results/audits/random-operator-v2/checks.json']}
if (out/'freeze.json').exists():assert readj(out/'freeze.json')['identity']==identity
else:writej(out/'freeze.json',dict(identity=identity,created_utc=utc()))
for f in dict(identity['code'],**identity['inputs']):
 dest=out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():shutil.copy2(ROOT/f,dest)
writej(out/'runtime.json',dict(engine=e.c,generation=e.genconfig.to_dict(),model_config=e.m.config.to_dict(),device=torch.cuda.get_device_name(),started_utc=utc()))
for g in groups:
 r=old[g['id']+'/baseline/recipient'];p=g['donor'];rc=e.cuts(r,g['recipient']['z'],g['recipient']['y'])
 alt=p['z']+1 if p['second']!='-' else p['z']+5
 altop='+' if p['y']>alt else '-';altk=abs(p['y']-alt)
 variants=dict(same_z_other_answer=program(p['z'],p['first'],p['b'],'+',p['k']+11),same_z_decomposition=program(p['z'],'-' if p['first']=='+' else '+',p['b']+3,p['second'],p['k']),different_z_same_answer=program(alt,p['first'],p['b'],altop,altk),same_leading_wide=program(10*p['z']+2,p['first'],p['b'],p['second'],p['k']))
 assert variants['same_z_other_answer']['z']==p['z'] and variants['same_z_other_answer']['y']!=p['y']
 assert variants['same_z_decomposition']['z']==p['z'] and variants['same_z_decomposition']['a']!=p['a']
 assert variants['different_z_same_answer']['z']!=p['z'] and variants['different_z_same_answer']['y']==p['y']
 assert str(variants['same_leading_wide']['z'])[0]==str(p['z'])[0] and len(str(variants['same_leading_wide']['z']))!=len(str(p['z']))
 for variant,d in variants.items():
  key=g['id']+'/'+variant+'/baseline'
  if key in j.rows:dr=j.rows[key]
  else:
   dr=j.add(dict(key=key,group=g['id'],variant=variant,kind='source_baseline',task=d,**e.generate(e.encode(prompt(d,g['template'])),seed)))
  correct=dr['parsed']['final']==d['y'] and dr['stop_reason']=='eos';dc=e.cuts(dr,d['z'],d['y'])
  for b in ['first_z','label_z']:
   key=g['id']+'/'+variant+'/'+b
   if key in j.rows:continue
   common=dict(key=key,group=g['id'],variant=variant,boundary=b,layer=26,kind='intervention',source_task=d,source_correct=correct,recipient_correct=old[g['id']+'/baseline/recipient']['correct'],original_counterfactual=g['hypotheses']['counterfactual'],source_counterfactual=op(d['z'],g['recipient']['second'],g['recipient']['k']))
   if b not in rc or b not in dc:
    j.add(dict(**common,available=False,own_cf=False,original_cf=False,reason='missing_site'));continue
   pre=rc[b]['prefix_ids'];src=e.capture(dc[b]['prefix_ids'],[26])[26];result=e.generate(r['prompt_ids'],seed,pre,26,src);f=out/(g['id']+'-'+variant+'-'+b+'.npz');np.savez(f,source=src.numpy())
   gg=g|dict(hypotheses=g['hypotheses']|dict(donor_answer=d['y'],source_counterfactual=common['source_counterfactual']))
   j.add(dict(**common,available=True,source_prefix_ids=dc[b]['prefix_ids'],source_hash=digest(src.numpy().tobytes()),activation_path=str(f.relative_to(ROOT)),own_cf=result['parsed']['final']==common['source_counterfactual'] and result['stop_reason']=='eos',original_cf=result['parsed']['final']==common['original_counterfactual'] and result['stop_reason']=='eos',matches=classify(result,gg)['matches'],**result));print(key,result['parsed'],correct,flush=True)
writej(out/'completion.json',dict(complete=True,groups=8,source_baselines=32,interventions=64,peak_memory_bytes=torch.cuda.max_memory_allocated(),finished_utc=utc()))
