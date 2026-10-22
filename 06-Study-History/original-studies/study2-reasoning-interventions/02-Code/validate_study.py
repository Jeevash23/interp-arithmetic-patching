"""Independent count/provenance checks and fresh-path saved-data reproduction."""
import argparse,collections,hashlib,json,math,subprocess,sys
from pathlib import Path
from statistics import mean
from ai04.common import ROOT,readj,writej,digest,utc
from ai04.tasks import generate_groups
from ai04.tasks import parse
import numpy as np

a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();out=Path(args.output);out=out if out.is_absolute() else ROOT/out
if out.exists():raise RuntimeError('Fresh validation output required')
out.mkdir(parents=True);checks=[]
def check(name,value,passed):
 checks.append(dict(name=name,value=value,passed=bool(passed)))
 if not passed:print('FAILED:',name,value)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
runs=['pilot/qwen06-pilot-v1','pilot/qwen06-random-repair-v2','development/qwen06-development-v2','final/qwen06-main-v1','final/qwen06-controls-v1','development/qwen06-symbolic-v1']
panel=ROOT/'04-Results/development/deepseek-panel-v1/completion.json'
if panel.exists():runs.append('development/deepseek-panel-v1')
for run in runs:
 p=ROOT/'04-Results'/run;fr=readj(p/'freeze.json');done=readj(p/'completion.json');rs=[json.loads(l) for l in (p/'records.jsonl').read_text().splitlines()]
 check(run+'/complete',done.get('status'),done['complete']);check(run+'/unique_keys',len(rs),len(rs)==len({r['key'] for r in rs}))
 check(run+'/freeze',fr['hash'],digest({k:fr[k] for k in ['config','source']})==fr['hash'])
 data={g['id']:g for g in readj(ROOT/fr['config']['data'])}
 for row in rs:
  if row['kind']=='baseline':
   g=data[row['group']];expected=g['hypotheses']['counterfactual'] if row['role']=='supplied' else g[row['role']]['y'];correct=row['parsed']['final']==expected and row['stop_reason']=='eos'
   check(run+'/baseline_grade/'+row['key'],row['expected'],row['expected']==expected and row['correct']==correct)
 for path,h in fr['source'].items():check(run+'/snapshot/'+path,h,(p/'source'/path).exists() and sha(p/'source'/path)==h)
 if done['status']=='completed':check(run+'/planned_rows',len(rs),len(rs)==done['baseline_records']+done['planned_interventions'])
 cp=subprocess.run([sys.executable,str(ROOT/'02-Code/verify_run.py'),run,'--output',str(out/(run.replace('/','-')+'-integrity.json'))],cwd=ROOT,capture_output=True,text=True)
 check(run+'/verifier_process',cp.returncode,cp.returncode==0)
 v=readj(out/(run.replace('/','-')+'-integrity.json'))
 # The preserved v1 failure is expected, not silently converted into a pass.
 expected=['pilot-006/first_z-L26/random/s0/norm','pilot-006/first_z-L26/random/s1/norm'] if run=='pilot/qwen06-pilot-v1' else []
 check(run+'/integrity_expected_issues',v['issues'],sorted(v['issues'])==sorted(expected))
 for r in rs:
  oldrun=r.get('reused_from')
  if oldrun:
   oldrows=[json.loads(x) for x in (ROOT/'04-Results'/'/'.join(oldrun.split('/')[:2])/'records.jsonl').read_text().splitlines()];old=next(x for x in oldrows if oldrun.endswith('/'+x['key']))
   check(run+'/reused/'+r['key'],r['key'],r['output_ids']==old['output_ids'] and r['prompt_ids']==old['prompt_ids'] and r['seed']==old['seed'])
for split,n,seed in [('audit',4,940001),('pilot',8,940021),('development',24,940031),('final',80,940041)]:
 rows=readj(ROOT/f'03-Data/generated/{split}.json');m=readj(ROOT/f'03-Data/splits/{split}.json');check('generator/'+split,m['hash'],rows==generate_groups(split,n,seed) and digest(rows)==m['hash'])
dp=ROOT/'04-Results/development/qwen06-donor-controls-v1';dr=[json.loads(l) for l in (dp/'records.jsonl').read_text().splitlines()];df=readj(dp/'freeze.json')['identity']
check('donor_panel/complete',len(dr),readj(dp/'completion.json')['complete'] and len(dr)==96 and len({r['key'] for r in dr})==96)
check('donor_panel/base_journal',df['base_journal_hash'],df['base_journal_hash']==sha(ROOT/'04-Results/development/qwen06-development-v2/records.jsonl'))
for path,h in dict(df['code'],**df['inputs']).items():check('donor_panel/snapshot/'+path,h,sha(dp/'source'/path)==h)
for r in dr:
 if r['kind']=='source_baseline' or r.get('available'):
  check('donor_panel/parser/'+r['key'],r['parsed'],parse(r['text'])==r['parsed'])
 if r['kind']=='intervention' and r.get('available'):
  check('donor_panel/target/'+r['key'],r['own_cf'],r['own_cf']==(r['parsed']['final']==r['source_counterfactual'] and r['stop_reason']=='eos'))
  with np.load(ROOT/r['activation_path']) as v:check('donor_panel/vector/'+r['key'],r['source_hash'],digest(v['source'].tobytes())==r['source_hash'])
  check('donor_panel/locality/'+r['key'],r['write'],r['write']['write_error']==0 and r['write']['other_positions_max']==0)
selection=readj(ROOT/'05-Notes/protocols/frozen-selection-v1.json');p=ROOT/'04-Results/development/qwen06-development-v2';rs=[json.loads(l) for l in (p/'records.jsonl').read_text().splitlines()];ranking=[]
for order,site in enumerate(['first_z-L13','first_z-L20','first_z-L26','label_z-L13','label_z-L20','label_z-L26']):
 rows=[r for r in rs if r.get('site')==site and r.get('condition')=='activation'];bygroup=collections.defaultdict(list)
 for r in rows:
  if r.get('B_delta') is not None:bygroup[r['group']].append(r['B_delta'])
 ranking.append(dict(site=site,order=order,coverage=len(bygroup),B=mean(mean(x) for x in bygroup.values()) if bygroup else None,C=mean(r['outcome']['counterfactual'] for r in rows),failure=sum(r['outcome']['category'] in ['noncompletion','invalid','unavailable'] for r in rows)))
eligible=[r for r in ranking if r['coverage']>=.8*24];maximum=max(r['B'] for r in eligible);b=next(r for r in eligible if r['B']>=maximum-.01);c=min(ranking,key=lambda x:(-x['C'],x['failure'],x['order']))
check('independent_selector_recalculation',dict(B=b['site'],C=c['site']),b['site']==selection['replay']['site'] and c['site']==selection['continuation']['site'])
check('selection_journal_hash',selection['journal_hash'],selection['journal_hash']==sha(p/'records.jsonl'))
mr=[json.loads(l) for l in (ROOT/'04-Results/final/qwen06-main-v1/records.jsonl').read_text().splitlines()];result=readj(ROOT/'04-Results/analysis/final-v2/results.json');groups=sorted({r['group'] for r in mr});rates={}
original_analysis=readj(ROOT/'04-Results/analysis/final-v1/results.json')
for key in ['selection','policies','primary_difference','main','controls']:
 check('analysis_v1_v2_primary_unchanged/'+key,key,original_analysis[key]==result[key])
check('main_groups',len(groups),len(groups)==64 and groups==[f'final-{i:03d}' for i in range(64)])
reserved={f'final-{i:03d}' for i in range(64,80)};reserve_hits=[]
for journal in (ROOT/'04-Results').rglob('records.jsonl'):
 for line in journal.read_text().splitlines():
  row=json.loads(line)
  if row.get('group') in reserved:reserve_hits.append(str(journal.relative_to(ROOT))+'/'+row['key'])
check('unused_final_reserve_no_model_records',reserve_hits,not reserve_hits)
for policy in ['replay','continuation']:
 r=[x for x in mr if x.get('site')==selection[policy]['site'] and x.get('condition')=='activation'];rates[policy]=[mean(x['outcome']['counterfactual'] for x in r if x['group']==g) for g in groups]
 check('independent_count/'+policy,sum(x['outcome']['counterfactual'] for x in r),sum(x['outcome']['counterfactual'] for x in r)==result['policies'][policy]['success'])
diff=mean(c-b for c,b in zip(rates['continuation'],rates['replay']));check('independent_primary_mean',diff,abs(diff-result['primary_difference']['mean'])<1e-12)
control=[json.loads(l) for l in (ROOT/'04-Results/final/qwen06-controls-v1/records.jsonl').read_text().splitlines()];mainkeys={r['key']:r for r in mr if r.get('kind')=='intervention'}
for r in control:
 if r['key'] in mainkeys and r.get('available') and mainkeys[r['key']].get('available'):check('repeated_main_control/'+r['key'],r['key'],r['output_ids']==mainkeys[r['key']]['output_ids'])
for script,dest,original,name in [('final_analysis_v2.py','fresh-primary','04-Results/analysis/final-v2/results.json','results.json'),('secondary_analysis.py','fresh-secondary','04-Results/analysis/secondary-v1/secondary.json','secondary.json')]:
 cp=subprocess.run([sys.executable,str(ROOT/'02-Code'/script),'--output',str(out/dest)],cwd=ROOT,capture_output=True,text=True);check('fresh_reanalysis/'+script,cp.returncode,cp.returncode==0)
 check('fresh_results_equal/'+script,original,readj(out/dest/name)==readj(ROOT/original))
models=[(readj(ROOT/'06-Study-History/environment/model-files.json'),ROOT/'../study1-value-transfer-and-readout/models/Qwen3-0.6B'),(readj(ROOT/'06-Study-History/environment/deepseek-model-files.json'),ROOT/'03-Data/raw/models/DeepSeek-R1-Distill-Qwen-1.5B')]
for m,p in models:
 files=m['files'].items() if isinstance(m['files'],dict) else [(f['name'],f) for f in m['files']]
 for name,meta in files:check('model_hash/'+p.name+'/'+name,meta['sha256'],sha(p/name)==meta['sha256'])
summary=dict(passed=all(c['passed'] for c in checks),checks=checks,finished_utc=utc(),note='Internal saved-data reproduction and independent count arithmetic, not external scientific review.')
writej(out/'checks.json',summary);print(json.dumps(dict(passed=summary['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']]),indent=2))
if not summary['passed']:raise SystemExit(1)
