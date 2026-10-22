from pathlib import Path
import json,numpy as np
from ai04.common import ROOT,readj,writej,digest,utc
run='development/qwen06-development-v2';p=ROOT/'04-Results'/run
assert readj(p/'completion.json')['complete']
assert readj(p/'competence.json')['passed']
c=readj(p/'freeze.json')['config'];rows=[json.loads(l) for l in (p/'records.jsonl').read_text().splitlines()]
ranking=[]
for order,s in enumerate(c['sites']):
 name=s['boundary']+'-L'+str(s['layer']);rs=[r for r in rows if r.get('site')==name and r.get('condition')=='activation'];gs=sorted({r['group'] for r in rs});bs=[np.mean([r['B_delta'] for r in rs if r['group']==g and r.get('B_delta') is not None]) for g in gs if any(r.get('B_delta') is not None for r in rs if r['group']==g)]
 ranking.append(dict(site=name,boundary=s['boundary'],layer=s['layer'],order=order,score_groups=len(bs),mean_B_delta=float(np.mean(bs)) if bs else None,success_rate=sum(r['outcome']['counterfactual'] for r in rs)/len(rs),successes=sum(r['outcome']['counterfactual'] for r in rs),failures=sum(r['outcome']['category'] in ['noncompletion','invalid','unavailable'] for r in rs),n=len(rs)))
eligible=[r for r in ranking if r['score_groups']>=.8*c['groups']]
if not eligible:raise RuntimeError('No replay-eligible site')
maximum=max(r['mean_B_delta'] for r in eligible);b=min([r for r in eligible if r['mean_B_delta']>=maximum-.01],key=lambda r:r['order']);g=min(ranking,key=lambda r:(-r['success_rate'],r['failures'],r['order']))
result=dict(selected_utc=utc(),development_run=run,journal_hash=digest((p/'records.jsonl').read_bytes()),ranking=ranking,replay=b,continuation=g,deadband_nats=.01,identical_policies=b['site']==g['site'])
dest=ROOT/'05-Notes/protocols/frozen-selection-v1.json'
if dest.exists():raise RuntimeError('selection already exists')
writej(dest,result)
base=readj(ROOT/'05-Notes/protocols/development-v2.json');sites=[]
for s in [b,g]:
 pair=dict(boundary=s['boundary'],layer=s['layer'])
 if pair not in sites:sites.append(pair)
base.update(run='final/qwen06-main-v1',phase='final',data='03-Data/generated/final.json',groups=64,baseline_seed=945001,continuation_seeds=[945001,945002],random_seed=945101,sites=sites,conditions=['unchanged','activation'],baseline_source=None,selection='05-Notes/protocols/frozen-selection-v1.json',reuse_baseline_assays=True,freeze_extra=['05-Notes/protocols/FINAL-EXECUTION-v3.md','04-Results/audits/baseline-score-identity-v1.json','05-Notes/protocols/DEVELOPMENT-v1.md','02-Code/select_sites.py','02-Code/final_analysis.py','02-Code/analyze.py','02-Code/trajectory_audit.py','02-Code/verify_run.py','06-Study-History/environment/model-files.json'])
writej(ROOT/'05-Notes/protocols/final-main-v1.json',base)
ctrl=base|dict(run='final/qwen06-controls-v1',phase='final_controls',groups=16,baseline_source=base['run'],sites=[dict(boundary=x,layer=26) for x in ['first_z','label_z']],conditions=['activation','text_digit','text_full','random_0','random_1','random_2'])
writej(ROOT/'05-Notes/protocols/final-controls-v1.json',ctrl)
print(json.dumps(result,indent=2))
