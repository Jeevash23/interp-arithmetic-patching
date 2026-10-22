from pathlib import Path
import argparse,json,numpy as np
from ai04.common import ROOT,readj,digest,writej
from ai04.tasks import parse,classify
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args();d=ROOT/'04-Results'/a.run
rows=[json.loads(l) for l in (d/'records.jsonl').read_text().splitlines()];groups=readj(ROOT/readj(d/'freeze.json')['config']['data']);gs={x['id']:x for x in groups};issues=[];checks=0;identities=[]
bykey={r['key']:r for r in rows};assert len(bykey)==len(rows)
for r in rows:
 if r.get('kind')=='baseline' or r.get('available'):
  checks+=1
  text=r['text'].replace('<｜end▁of▁sentence｜>','') if r.get('parser_normalization')=='remove native DeepSeek end-of-sentence marker only' else r['text']
  if parse(text)!=r['parsed']:issues.append(r['key']+'/parser')
  if r['output_ids']!=r['replayed_output_ids']+r['fresh_output_ids']:issues.append(r['key']+'/tokens')
  if len(r['prefix_ids'])-len(r['prompt_ids'])!=len(r['replayed_output_ids']):issues.append(r['key']+'/prefix')
 if r.get('kind')=='intervention' and r.get('available'):
  g=gs[r['group']]
  if classify(r,g)!=r['outcome']:issues.append(r['key']+'/classify')
  if r['condition']=='unchanged' and r['stream']==0:
   b=bykey[g['id']+'/baseline/recipient'];same=r['output_ids']==b['output_ids'];identities.append(dict(key=r['key'],exact=same,semantic=r['parsed']['final']==b['parsed']['final']))
   if not same:issues.append(r['key']+'/no_op_identity')
  if r['write'] and any(r['write'][k]!=0 for k in ['other_positions_max','write_error']):issues.append(r['key']+'/locality')
  with np.load(ROOT/r['activation_path']) as v:
   if digest(v['source'].tobytes())!=r['source_hash']:issues.append(r['key']+'/source_hash')
   if digest(v['written'].tobytes())!=r['written_hash']:issues.append(r['key']+'/written_hash')
   if r['condition']=='activation' and not np.array_equal(v['source'],v['written']):issues.append(r['key']+'/replacement_vector')
   if r['condition'].startswith('random'):
    if abs(np.linalg.norm(v['written'].astype(float)-v['recipient'].astype(float))-np.linalg.norm(v['source'].astype(float)-v['recipient'].astype(float)))>1e-4:issues.append(r['key']+'/norm')
  assay=readj(ROOT/r['assays_path'])
  for score in (assay['B'] or {}).values():
   if not np.all(np.isfinite(score['token_logprobs'])) or abs(sum(score['token_logprobs'])-score['sum_logprob'])>1e-8:issues.append(r['key']+'/score_sum')
summary=dict(run=a.run,rows=len(rows),checked_outputs=checks,issues=issues,passed=not issues,identity_checks=identities,journal_hash=digest((d/'records.jsonl').read_bytes()))
writej(ROOT/a.output,summary);print(json.dumps(summary,indent=2))
