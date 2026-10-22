from pathlib import Path
import argparse,json,collections
import numpy as np
from scipy.stats import spearmanr
from ai04.common import ROOT,readj,writej,digest

def loadrows(run):return [json.loads(l) for l in (ROOT/'04-Results'/run/'records.jsonl').read_text().splitlines()]
def interval(values,seed=942001,B=10000):
 a=np.asarray(values,float);rng=np.random.default_rng(seed);draw=a[rng.integers(0,len(a),(B,len(a)))].mean(1)
 return dict(mean=float(a.mean()),lo=float(np.quantile(draw,.025)),hi=float(np.quantile(draw,.975)),n_groups=len(a),degenerate=bool(a.std()==0))
def summarize(rows):
 base=[r for r in rows if r['kind']=='baseline'];ints=[r for r in rows if r['kind']=='intervention'];ids=sorted({r['group'] for r in base});out=dict(groups=len(ids),baseline={},conditions={},rows=len(rows))
 for role in ['donor','recipient','supplied']:
  rr=[r for r in base if r['role']==role];out['baseline'][role]=dict(n=len(rr),correct=sum(r['correct'] for r in rr),natural=sum(r['stop_reason']=='eos' for r in rr),formats=dict(collections.Counter(r['parsed']['format'] for r in rr)),tokens=sum(r['fresh_tokens'] for r in rr),seconds=sum(r['seconds'] for r in rr))
 for site,condition in sorted({(r['site'],r['condition']) for r in ints}):
  rr=[r for r in ints if r['site']==site and r['condition']==condition];groupmeans=[np.mean([x['outcome']['counterfactual'] for x in rr if x['group']==g]) for g in ids]
  allA=[x['A_delta'] for x in rr if x.get('A_delta') is not None];allB=[x['B_delta'] for x in rr if x.get('B_delta') is not None];margins=[x['B_margin'] for x in rr if x.get('B_margin') is not None]
  out['conditions'][site+'/'+condition]=dict(n=len(rr),available=sum(r['available'] for r in rr),success=sum(r['outcome']['counterfactual'] for r in rr),categories=dict(collections.Counter(r['outcome']['category'] for r in rr)),cluster_interval=interval(groupmeans),competent_n=sum(r['competent'] for r in rr),competent_success=sum(r['competent'] and r['outcome']['counterfactual'] for r in rr),mean_A_delta=float(np.mean(allA)) if allA else None,mean_B_delta=float(np.mean(allB)) if allB else None,mean_B_margin=float(np.mean(margins)) if margins else None,positive_B_margin=sum(x>0 for x in margins),fresh_tokens=sum(r.get('fresh_tokens',0) for r in rr),seconds=sum(r['seconds'] for r in rr))
 out['total_generation_tokens']=sum(r.get('inference_fresh_tokens',r.get('fresh_tokens',0)) for r in rows);out['total_generation_seconds']=sum(r.get('seconds',0) for r in rows)
 return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--output',required=True);args=a.parse_args();p=Path(args.output)
 if not p.is_absolute():p=ROOT/p
 if p.exists():raise SystemExit('Refusing existing analysis output; use a fresh path')
 rows=loadrows(args.run);summary=summarize(rows);writej(p/'summary.json',summary);writej(p/'provenance.json',dict(input=args.run,input_sha256=digest((ROOT/'04-Results'/args.run/'records.jsonl').read_bytes())))
 print(json.dumps(summary,indent=2))
