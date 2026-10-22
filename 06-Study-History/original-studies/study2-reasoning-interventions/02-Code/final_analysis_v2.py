from pathlib import Path
import argparse,json,collections
import numpy as np
from scipy.stats import spearmanr
from ai04.common import ROOT,readj,writej,digest
from analyze import loadrows,summarize

def stratified_interval(values,strata,B=10000,seed=942001):
 vals=np.asarray(values,float);s=np.asarray(strata);rng=np.random.default_rng(seed);draw=np.zeros(B)
 for v in np.unique(s):
  subset=vals[s==v];draw+=subset[rng.integers(0,len(subset),(B,len(subset)))].sum(1)
 draw/=len(vals)
 return dict(mean=float(vals.mean()),lo=float(np.quantile(draw,.025)),hi=float(np.quantile(draw,.975)),n_groups=len(vals),degenerate=bool(vals.std()==0))

def valid_answer_agreement(a,t):
 return bool(a.get('available') and t.get('available') and a['stop_reason']=='eos' and t['stop_reason']=='eos' and a['parsed']['final'] is not None and t['parsed']['final'] is not None and a['parsed']['final']==t['parsed']['final'])

def main(out):
 out=Path(out)
 if not out.is_absolute():out=ROOT/out
 if out.exists():raise ValueError('fresh analysis directory required')
 for run in ['final/qwen06-main-v1','final/qwen06-controls-v1']:
  done=readj(ROOT/'04-Results'/run/'completion.json')
  assert done['complete'] and done['status']=='completed' and done['intervention_records']==done['planned_interventions']
 selection=readj(ROOT/'05-Notes/protocols/frozen-selection-v1.json');main=loadrows('final/qwen06-main-v1');ctrl=loadrows('final/qwen06-controls-v1');gmap={r['id']:r for r in readj(ROOT/'03-Data/generated/final.json')};ids=sorted({r['group'] for r in main});strata=[gmap[g]['template'] for g in ids]
 assert len(ids)==64
 cells={}
 for policy in ['replay','continuation']:
  site=selection[policy]['site'];rs=[r for r in main if r.get('site')==site and r.get('condition')=='activation'];means=[np.mean([r['outcome']['counterfactual'] for r in rs if r['group']==g]) for g in ids];anys=[any(r['outcome']['counterfactual'] for r in rs if r['group']==g) for g in ids]
  cells[policy]=dict(site=site,success=sum(r['outcome']['counterfactual'] for r in rs),branches=len(rs),groups=len(ids),groups_with_any_success=sum(anys),interval=stratified_interval(means,strata),zero_group_one_sided95_upper=1-.05**(1/len(ids)) if not any(anys) else None,group_rates=means)
 diff=np.asarray(cells['continuation']['group_rates'])-np.asarray(cells['replay']['group_rates']);primary=stratified_interval(diff,strata)
 res=dict(selection=selection,policies=cells,primary_difference=primary,identical_policies=selection['identical_policies'],main=summarize(main),controls=summarize(ctrl))
 # Paired control comparisons share program and random stream; exact output vs text digit.
 paired=[]
 for site in sorted({r['site'] for r in ctrl if r.get('kind')=='intervention'}):
  for g in sorted({r['group'] for r in ctrl}):
   for stream in [0,1]:
    subset={r['condition']:r for r in ctrl if r.get('site')==site and r['group']==g and r.get('stream')==stream}
    a=subset['activation'];t=subset['text_digit'];f=subset['text_full'];ok=a.get('available') and t.get('available')
    paired.append(dict(site=site,group=g,stream=stream,available=bool(ok),exact_sequence=bool(ok and a['output_ids']==t['output_ids']),same_answer=valid_answer_agreement(a,t),raw_parsed_equality=bool(ok and a['parsed']['final']==t['parsed']['final']),joint_valid_completions=bool(ok and a['outcome']['valid'] and t['outcome']['valid'] and a['outcome']['completed'] and t['outcome']['completed']),activation_cf=a['outcome']['counterfactual'],digit_cf=t['outcome']['counterfactual'],full_cf=f['outcome']['counterfactual']))
 res['digit_pairs']=paired;res['digit_summary']={s:{k:sum(r[k] for r in paired if r['site']==s) for k in ['available','exact_sequence','same_answer','joint_valid_completions','raw_parsed_equality','activation_cf','digit_cf','full_cf']} for s in sorted({r['site'] for r in paired})}
 res['analysis_revision']='v2: semantic agreement requires two valid natural completions; primary estimator unchanged';writej(out/'results.json',res);writej(out/'provenance.json',dict(inputs={run:digest((ROOT/'04-Results'/run/'records.jsonl').read_bytes()) for run in ['final/qwen06-main-v1','final/qwen06-controls-v1']},analysis_source_hash=digest(Path(__file__).read_bytes())))
 print(json.dumps({k:res[k] for k in ['policies','primary_difference','identical_policies','digit_summary']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
