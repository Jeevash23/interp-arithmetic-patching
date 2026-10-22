"""Descriptive diagnostics. No additional confirmatory contrasts or p-values."""
import argparse,collections,json,re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from ai04.common import ROOT,readj,writej,digest
from analyze import loadrows,summarize
from ai04.tasks import op

def maybe_corr(x,y):
 if len(x)<3 or len(set(x))<2 or len(set(y))<2:return dict(rho=None,reason='constant or insufficient observations')
 return dict(rho=float(spearmanr(x,y).statistic),reason='descriptive rank association; no independence claim for sites')

def describe(run):
 p=ROOT/'04-Results'/run;c=readj(p/'freeze.json')['config'];done=readj(p/'completion.json');assert done['complete'];rows=loadrows(run);groups={g['id']:g for g in readj(ROOT/c['data'])[:c['groups']]}
 assert sum(r['kind']=='baseline' for r in rows)==3*c['groups']
 ints=[r for r in rows if r['kind']=='intervention'];out=dict(summary=summarize(rows),timing={},assays={},trajectories={},cost={},baseline_stage_audit={})
 for role in ['donor','recipient','supplied']:
  rr=[r for r in rows if r['kind']=='baseline' and r['role']==role]
  flags=[dict(key=r['key'],intermediate_correct=r['parsed']['intermediate']==groups[r['group']]['donor' if role=='supplied' else role]['z'],final_correct=r['correct']) for r in rr]
  out['baseline_stage_audit'][role]=dict(n=len(rr),intermediate_correct=sum(x['intermediate_correct'] for x in flags),both_correct=sum(x['intermediate_correct'] and x['final_correct'] for x in flags),exceptions=[x for x in flags if not x['intermediate_correct'] or not x['final_correct']])
 basekeys={r['key']:r for r in rows if r['kind']=='baseline'};clean_ids=[];all_stage_ids=[]
 for gid,g in groups.items():
  bs={role:basekeys[gid+'/baseline/'+role] for role in ['donor','recipient','supplied']}
  finals=all(x['correct'] for x in bs.values());source_rec=all(bs[role]['parsed']['intermediate']==g[role]['z'] for role in ['donor','recipient'])
  if finals and source_rec:clean_ids.append(gid)
  if finals and source_rec and bs['supplied']['parsed']['intermediate']==g['donor']['z']:all_stage_ids.append(gid)
 out['stage_clean_sensitivity']=dict(note='Descriptive post-freeze sensitivity; primary eligibility and all-intended results unchanged.',source_recipient_intermediates_and_all_finals=clean_ids,all_six_stage_labels=all_stage_ids,sites={})
 for site in sorted({r['site'] for r in ints}):
  rr=[r for r in ints if r['site']==site and r['condition']=='activation' and r['group'] in clean_ids]
  out['stage_clean_sensitivity']['sites'][site]=dict(n_groups=len(clean_ids),branches=len(rr),counterfactual=sum(r['outcome']['counterfactual'] for r in rr))
 if done.get('status')=='completed':assert len(ints)==done['planned_interventions']
 for site in sorted({r['site'] for r in ints}):
  rr=[r for r in ints if r['site']==site];available=[r for r in rr if r.get('available')];unique={r['group']:r['timing'] for r in available}
  out['timing'][site]=dict(groups_available=len(unique),groups_total=c['groups'],in_thinking=sum(t['in_thinking'] for t in unique.values()),prior_literal_z=sum(t['prior_z_mentions']>0 for t in unique.values()),prior_literal_y=sum(t['prior_y_mentions']>0 for t in unique.values()),median_output_index=float(np.median([t['output_index'] for t in unique.values()])) if unique else None)
  for condition in sorted({r['condition'] for r in rr}):
   rs=[r for r in rr if r['condition']==condition];avs=[r for r in rs if r.get('available')];u={r['group']:r for r in avs};diag=[]
   for r in avs:
    g=groups[r['group']];suffix=r['text'][r['timing']['char_start']:];m=re.match(r'\s*(-?\d+)',suffix);first=int(m.group(1)) if m else None;zr=g['recipient']['z'];zd=g['donor']['z'];later=bool(re.search(r'(?<!\d)'+str(zr)+r'(?!\d)',suffix[m.end():] if m else suffix));returned=r['stop_reason']=='eos' and r['parsed']['final']==g['recipient']['y'];baseline=next(b for b in rows if b['kind']=='baseline' and b['group']==r['group'] and b['role']=='recipient')
    original_first=int(r['timing']['observed_text']);changed=first is not None and first!=original_first
    consistent=bool(changed and 'second' in g['recipient'] and r['stop_reason']=='eos' and r['parsed']['final']==op(first,g['recipient']['second'],g['recipient']['k']))
    diag.append(dict(key=r['key'],group=r['group'],first_numeral=first,baseline_first_numeral=original_first,donor_first_digit=first is not None and str(first)[0]==str(zd)[0],complete_donor_numeral=first==zd,changed_numeral=changed,differs_from_oracle=first is not None and first!=zr,changed_later_original_recipient_final=changed and later and returned,final_consistent_with_changed_first_number=consistent,error_word=bool(re.search(r'\b(mistake|error|incorrect|wrong|miscalculat\w*)\b',suffix,re.I)),actual_baseline_answer=r['stop_reason']=='eos' and baseline['stop_reason']=='eos' and r['parsed']['final'] is not None and r['parsed']['final']==baseline['parsed']['final'],actual_baseline_sequence=r['output_ids']==baseline['output_ids']))
   keys=['donor_first_digit','complete_donor_numeral','changed_numeral','differs_from_oracle','changed_later_original_recipient_final','final_consistent_with_changed_first_number','error_word','actual_baseline_answer','actual_baseline_sequence'];out['trajectories'][site+'/'+condition]=dict(n=len(rs),available=len(avs),arithmetic_consistency_applicable=all('second' in g['recipient'] for g in groups.values()),**{k:sum(x[k] for x in diag) for k in keys},records=diag)
   A=[r['A_delta'] for r in u.values() if r.get('A_delta') is not None];B=[r['B_delta'] for r in u.values() if r.get('B_delta') is not None];M=[r['B_margin'] for r in u.values() if r.get('B_margin') is not None]
   C={g:np.mean([r['outcome']['counterfactual'] for r in rs if r['group']==g]) for g in groups}
   ab=[r for r in u.values() if r.get('A_delta') is not None and r.get('B_delta') is not None];ac=[r for r in u.values() if r.get('A_delta') is not None];bc=[r for r in u.values() if r.get('B_delta') is not None]
   out['assays'][site+'/'+condition]=dict(groups=len(u),A_groups=len(A),B_groups=len(B),positive_A=sum(a>.001 for a in A),B_increase_above_deadband=sum(b>.01 for b in B),B_decrease_below_deadband=sum(b<-.01 for b in B),B_within_deadband=sum(abs(b)<=.01 for b in B),B_prefers_cf=sum(m>0 for m in M),A_B=maybe_corr([r['A_delta'] for r in ab],[r['B_delta'] for r in ab]),A_C=maybe_corr([r['A_delta'] for r in ac],[C[r['group']] for r in ac]),B_C=maybe_corr([r['B_delta'] for r in bc],[C[r['group']] for r in bc]))
   raw_A=[readj(ROOT/r['assays_path'])['A'] for r in u.values()];raw_A=[x for x in raw_A if x is not None]
   out['assays'][site+'/'+condition].update(mean_raw_A_margin=float(np.mean([x['digit_margin'] for x in raw_A])) if raw_A else None,mean_raw_donor_digit_probability=float(np.mean([x['donor_digit_probability'] for x in raw_A])) if raw_A else None,A_raw_prefers_donor_digit=sum(x['digit_margin']>0 for x in raw_A))
 assay_files=list((p/'assays').glob('*.json'));out['cost']=dict(executed_generation_tokens=sum(r.get('inference_fresh_tokens',r.get('fresh_tokens',0)) for r in rows),executed_generation_seconds=sum(r.get('seconds',0) for r in rows),combined_A_B_assay_seconds=sum(readj(x)['seconds'] for x in assay_files),execution_reused_rows=sum('reused_from' in r or 'execution_reused_from' in r for r in rows),note='Wall times include contemporaneous contention; all saved assays include clean/edited A/B but not capture, loading or bookkeeping, including assays preceding a text-prefix budget rejection. Reused rows have zero new inference cost.')
 out['journal_hash']=digest((p/'records.jsonl').read_bytes());return out

def donor():
 run='development/qwen06-donor-controls-v1';p=ROOT/'04-Results'/run;assert readj(p/'completion.json')['complete'];rows=loadrows(run);dev=loadrows('development/qwen06-development-v2');base={r['key']:r for r in rows if r['kind']=='source_baseline'};out={}
 development_groups={g['id']:g for g in readj(ROOT/'03-Data/generated/development.json')}
 from ai04.engine import Engine
 # Tokenizer is sufficient to decode the saved inserted suffix; do not load weights.
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained(ROOT/'../study1-value-transfer-and-readout/models/Qwen3-0.6B',local_files_only=True)
 for variant,b in sorted({(r['variant'],r['boundary']) for r in rows if r['kind']=='intervention'}):
  rr=[r for r in rows if r['kind']=='intervention' and r['variant']==variant and r['boundary']==b];av=[r for r in rr if r['available']];diag=[]
  for r in av:
   original=next(x for x in dev if x.get('site')==b+'-L26' and x.get('condition')=='activation' and x['group']==r['group'] and x['stream']==0);m=re.match(r'\s*(-?\d+)',t.decode(r['fresh_output_ids']));z=int(m.group(1)) if m else None
   diag.append(dict(key=r['key'],same_sequence_as_original=original.get('available') and r['output_ids']==original['output_ids'],same_answer_as_original=original.get('available') and r['stop_reason']=='eos' and original['stop_reason']=='eos' and r['parsed']['final'] is not None and r['parsed']['final']==original['parsed']['final'],first_digit_copied=z is not None and str(z)[0]==str(r['source_task']['z'])[0],complete_source_value=z==r['source_task']['z']))
  out[variant+'/'+b]=dict(n=len(rr),available=len(av),source_correct=sum(r['source_correct'] for r in rr),own_cf=sum(r['own_cf'] for r in rr),original_cf=sum(r['original_cf'] for r in rr),natural=sum(r.get('stop_reason')=='eos' for r in rr),own_cf_competent=sum(r['own_cf'] and r['source_correct'] and r['recipient_correct'] for r in rr),competent_n=sum(r['source_correct'] and r['recipient_correct'] for r in rr),**{k:sum(x[k] for x in diag) for k in ['same_sequence_as_original','same_answer_as_original','first_digit_copied','complete_source_value']},records=diag)
 for key,cell in out.items():
  variant,b=key.split('/');rr=[r for r in rows if r['kind']=='intervention' and r['variant']==variant and r['boundary']==b]
  cell['source_leading_digit_changed']=sum(str(r['source_task']['z'])[0]!=str(development_groups[r['group']]['donor']['z'])[0] for r in rr)
 return dict(groups=8,source_baselines=len(base),source_baseline_audit=dict(n=len(base),correct_final=sum(r['stop_reason']=='eos' and r['parsed']['final']==r['task']['y'] for r in base.values()),correct_intermediate=sum(r['parsed']['intermediate']==r['task']['z'] for r in base.values()),natural=sum(r['stop_reason']=='eos' for r in base.values())),conditions=out,journal_hash=digest((p/'records.jsonl').read_bytes()),cost=dict(tokens=sum(r.get('fresh_tokens',0) for r in rows),generation_seconds=sum(r.get('seconds',0) for r in rows)))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();p=Path(args.output);p=p if p.is_absolute() else ROOT/p
 if p.exists():raise RuntimeError('Use a fresh output path')
 runs=['pilot/qwen06-pilot-v1','pilot/qwen06-random-repair-v2','development/qwen06-development-v2','final/qwen06-main-v1','final/qwen06-controls-v1','development/qwen06-symbolic-v1']
 result=dict(runs={run:describe(run) for run in runs},donor=donor(),interpretation='Secondary descriptive diagnostics. Leading-digit association does not identify value encoding. Trajectory signatures and error words need contextual review. Constant C makes correlation undefined, not zero.')
 dev=result['runs']['development/qwen06-development-v2']['summary']['conditions'];cells=[v for k,v in dev.items() if k.endswith('/activation')]
 result['development_site_ranks']=dict(B_C=maybe_corr([v['mean_B_delta'] for v in cells],[v['success']/v['n'] for v in cells]),A_C=maybe_corr([v['mean_A_delta'] for v in cells],[v['success']/v['n'] for v in cells]))
 writej(p/'secondary.json',result);writej(p/'provenance.json',dict(source_sha256=digest(Path(__file__).read_bytes())));print('Saved secondary analysis:',p)
