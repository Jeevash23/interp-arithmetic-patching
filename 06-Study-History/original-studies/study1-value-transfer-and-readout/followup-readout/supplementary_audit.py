"""Post-analysis case audit; no new model inference or confirmatory endpoint."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=[json.loads(s) for f in (HERE/'results').glob('seed-*/interventions.jsonl') for s in f.read_text().splitlines()]
index={(r['seed'],r['id'],r['condition']):r for r in raw}
rows=[]
for r in raw:
    if r['condition']=='activation' and r['width']==2 and r['success']:
        short=index[r['seed'],r['id'].replace('-w2','-w1'),'activation'];text=index[r['seed'],r['id'],'text_first_digit']
        rows.append({'id':r['id'],'seed':r['seed'],'group':r['group'],'short_z':r['leading_digit'],'long_z':r['donor_z'],'target_short':short['predictions']['counterfactual'],'target_long':r['predictions']['counterfactual'],'actual_intermediate':r['parsed']['intermediate'],'actual_answer':r['parsed']['answer'],'short_source_exact':short['output_ids']==r['output_ids'],'digit_text_exact':text['output_ids']==r['output_ids']})
with (HERE/'results/analysis/two-digit-success-audit.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
assert len(rows)==8 and len({r['group'] for r in rows})==3 and all(r['short_source_exact'] and r['digit_text_exact'] for r in rows)
positions=[]
for c in ['donor_pre_final','donor_after_final']:
    rr=[r for r in raw if r['condition']==c and r['status']=='generated']
    positions.append({'condition':c,'n':len(rr),'source_next_token_match':sum(r['fresh_output_ids'][0]==r['source_next_token'] for r in rr),'immediate_eos':sum(r['fresh_output_ids']==[151645] for r in rr)})
(HERE/'results/analysis/source-position-audit.json').write_text(json.dumps(positions,indent=2)+'\n')
print({'two_digit_successes':len(rows),'groups':len({r['group'] for r in rows}),'positions':positions})
