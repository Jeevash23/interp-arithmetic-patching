import argparse,shutil,time
from pathlib import Path
import numpy as np,torch
from ai04.common import ROOT,readj,writej,freeze,Journal,digest,utc
from ai04.engine import Engine
from ai04.tasks import classify

a=argparse.ArgumentParser();a.add_argument('config');args=a.parse_args();c=readj(ROOT/args.config)
gate=readj(ROOT/c['gate'])
if not gate.get('complete') or not gate['passed']:raise RuntimeError('Qualification gate not passed')
extra=[args.config,c['data'],c['protocol'],c['gate']]+c.get('freeze_extra',[])
if c.get('selection'):extra.append(c['selection'])
out,h=freeze(c['run'],c,extra)
for f in readj(out/'freeze.json')['source']:
 dest=out/'source'/f
 if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
j=Journal(out/'records.jsonl');e=Engine(c['engine']);groups=readj(ROOT/c['data'])[:c['groups']]
writej(out/'runtime.json',dict(started_utc=utc(),model_config=e.m.config.to_dict(),template=e.t.chat_template,engine=e.c,generation=e.genconfig.to_dict(),device=torch.cuda.get_device_name() if e.c['device']=='cuda' else 'cpu',freeze_hash=h))
def add(r):
 r=j.add(r);print(r['key'],r.get('outcome',r.get('correct')),round(r.get('seconds',0),2),flush=True);return r
for g in groups:
 for role in ['donor','recipient','supplied']:
  key=g['id']+'/baseline/'+role
  if key in j.rows:continue
  prompt=e.encode(g[role+'_prompt'])
  if c.get('baseline_source') and key in Journal(ROOT/'04-Results'/c['baseline_source']/'records.jsonl').rows:
   old=Journal(ROOT/'04-Results'/c['baseline_source']/'records.jsonl').rows[key]
   assert old['prompt_ids']==prompt and old['seed']==c['baseline_seed']
   r={k:v for k,v in old.items() if k not in ['key','group','phase','kind','role','expected','correct','recorded_utc']}
   r.update(reused_from=c['baseline_source']+'/'+key,original_seconds=r['seconds'],seconds=0.,inference_fresh_tokens=0)
  else:r=e.generate(prompt,c['baseline_seed'])
  answer=g['donor']['y'] if role=='donor' else g['recipient']['y'] if role=='recipient' else g['hypotheses']['counterfactual']
  add(dict(key=key,group=g['id'],phase=c['phase'],kind='baseline',role=role,expected=answer,correct=r['parsed']['final']==answer and r['stop_reason']=='eos',**r))
competence={role:sum(j.rows[g['id']+'/baseline/'+role]['correct'] for g in groups)/len(groups) for role in ['donor','recipient','supplied']}
writej(out/'competence.json',dict(rates=competence,groups=len(groups),passed=all(v>=c['competence_gate'] for v in competence.values())))
if not all(v>=c['competence_gate'] for v in competence.values()):
 writej(out/'completion.json',dict(status='competence_boundary',complete=True,mechanism_run=False,competence=competence));raise SystemExit('Competence gate failed; all baselines retained')
if c.get('baselines_only',False):raise SystemExit('Baselines-only configuration complete')
for g in groups:
 r=j.rows[g['id']+'/baseline/recipient'];d=j.rows[g['id']+'/baseline/donor'];rc=e.cuts(r,g['recipient']['z'],g['recipient']['y']);dc=e.cuts(d,g['donor']['z'],g['donor']['y'])
 writej(out/(g['id']+'-boundaries.json'),dict(recipient=rc,donor=dc))
 competent=all(j.rows[g['id']+'/baseline/'+role]['correct'] for role in ['donor','recipient','supplied'])
 for site in c['sites']:
  boundary=site['boundary'];layer=site['layer'];sitename=boundary+'-L'+str(layer)
  for condition in c['conditions']:
   for si,seed in enumerate(c['continuation_seeds']):
    key=f'{g["id"]}/{sitename}/{condition}/s{si}'
    if key in j.rows:continue
    common=dict(key=key,group=g['id'],phase=c['phase'],kind='intervention',template=g['template'],boundary=boundary,layer=layer,site=sitename,condition=condition,stream=si,competent=competent,hypotheses=g['hypotheses'])
    if boundary not in rc or boundary not in dc:
     add(dict(**common,available=False,reason='missing_recipient_or_source_boundary',outcome=dict(category='unavailable',counterfactual=False,completed=False,valid=False),seconds=0));continue
    prefix=rc[boundary]['prefix_ids'];pos=len(prefix)-1;source=e.capture(dc[boundary]['prefix_ids'],[layer])[layer];before=e.capture(prefix,[layer])[layer]
    source_to_use=source;active_layer=layer;rule='replace';text_insert=None
    if condition=='unchanged':active_layer=None
    elif condition.startswith('random'):
     ri=int(condition.split('_')[-1]) if '_' in condition else 0
     source_to_use=e.random_source(before,source,c['random_seed']+int(g['id'].split('-')[-1])*1000+layer*10+ri)
    elif condition in ['text_full','text_digit','text_identity']:
     active_layer=None;value=g['recipient']['z'] if condition=='text_identity' else g['donor']['z']
     insert=e.t.encode(str(value)[0] if condition=='text_digit' else str(value),add_special_tokens=False)
     endindex=rc[boundary]['numeral_end_index']
     if condition!='text_digit' and endindex<len(r['output_ids']):insert+=r['output_ids'][endindex:endindex+1]
     prefix=prefix+insert;text_insert=dict(ids=insert,text=e.t.decode(insert),replaced_original_ids=r['output_ids'][rc[boundary]['output_index']:endindex+1])
    elif condition!='activation':raise ValueError(condition)
    vecpath=out/'activations'/f'{g["id"]}-{sitename}-{condition}.npz';vecpath.parent.mkdir(exist_ok=True)
    if not vecpath.exists():np.savez(vecpath,source=source.numpy(),recipient=before.numpy(),written=source_to_use.numpy())
    measurekey=f'{g["id"]}/{sitename}/{condition}/assays';measurefile=out/'assays'/f'{g["id"]}-{sitename}-{condition}.json'
    if measurefile.exists():assay=readj(measurefile)
    else:
     start=time.monotonic();baseA=e.immediate(rc[boundary]['prefix_ids'],g['donor']['z'],g['recipient']['z'])
     A=baseA if c.get('reuse_baseline_assays') and condition=='unchanged' else None if text_insert else e.immediate(prefix,g['donor']['z'],g['recipient']['z'],active_layer,source_to_use)
     scores=base_scores=None;replayprefix=None
     if 'final' in rc:
      replayprefix=list(rc['final']['prefix_ids'])
      if text_insert:
       oi=rc[boundary]['output_index'];end=rc[boundary]['numeral_end_index']+1
       if condition=='text_digit':end=oi+1
       replayprefix=r['prompt_ids']+r['output_ids'][:oi]+text_insert['ids']+r['output_ids'][end:rc['final']['output_index']]
      values=list(g['hypotheses'].values());shared=out/'shared-baseline-assays'/(g['id']+'.json')
      if c.get('reuse_baseline_assays') and shared.exists():
       saved=readj(shared);assert saved['prefix_ids']==rc['final']['prefix_ids'] and saved['values']==values;base_scores=saved['scores']
      else:
       base_scores=e.score(rc['final']['prefix_ids'],values)
       if c.get('reuse_baseline_assays'):writej(shared,dict(prefix_ids=rc['final']['prefix_ids'],values=values,scores=base_scores))
      scores=base_scores if c.get('reuse_baseline_assays') and condition=='unchanged' else e.score(replayprefix,values,active_layer,source_to_use,pos)
     cf=str(g['hypotheses']['counterfactual']);rec=str(g['recipient']['y'])
     assay=dict(A=A,A_baseline=baseA,A_delta=None if A is None else A['digit_margin']-baseA['digit_margin'],B=scores,B_baseline=base_scores,B_delta=None if scores is None else (scores[cf]['sum_logprob']-scores[rec]['sum_logprob'])-(base_scores[cf]['sum_logprob']-base_scores[rec]['sum_logprob']),B_margin=None if scores is None else scores[cf]['sum_logprob']-scores[rec]['sum_logprob'],replay_prefix_ids=replayprefix,seconds=time.monotonic()-start)
     writej(measurefile,assay)
    if len(prefix)-len(r['prompt_ids'])>=e.c['max_tokens']:
     add(dict(**common,available=False,reason='prefix_exhausts_budget',outcome=dict(category='unavailable',counterfactual=False,completed=False,valid=False),seconds=0));continue
    reused=next((x for x in j.rows.values() if c.get('reuse_identical_unchanged') and condition=='unchanged' and x.get('kind')=='intervention' and x.get('condition')=='unchanged' and x.get('available') and x['prefix_ids']==prefix and x['seed']==seed),None)
    if reused is not None:
     fields=['prompt_ids','prefix_ids','replayed_output_ids','fresh_output_ids','output_ids','text','parsed','stop_reason','fresh_tokens','total_output_tokens','seconds','seed','effective_seed','write','semantics']
     result={k:reused[k] for k in fields}
     result.update(seconds=0.,inference_fresh_tokens=0,execution_reused_from=reused['key'])
    else:result=e.generate(r['prompt_ids'],seed,prefix,active_layer,source_to_use,pos=pos)
    add(dict(**common,available=True,assays_path=str(measurefile.relative_to(ROOT)),A_delta=assay['A_delta'],B_delta=assay['B_delta'],B_margin=assay['B_margin'],activation_path=str(vecpath.relative_to(ROOT)),source_hash=digest(source.numpy().tobytes()),written_hash=digest(source_to_use.numpy().tobytes()),source_prefix_ids=dc[boundary]['prefix_ids'],timing=rc[boundary],text_insert=text_insert,outcome=classify(result,g),**result))
writej(out/'completion.json',dict(status='completed',complete=True,groups=len(groups),baseline_records=3*len(groups),planned_interventions=len(groups)*len(c['sites'])*len(c['conditions'])*len(c['continuation_seeds']),intervention_records=sum(r.get('kind')=='intervention' for r in j.rows.values()),freeze_hash=h,peak_memory_bytes=torch.cuda.max_memory_allocated() if e.c['device']=='cuda' else None,finished_utc=utc()))
