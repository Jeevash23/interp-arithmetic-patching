"""Analyze all frozen replication endpoints, including missing and failed cases."""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from prepare import HERE,ROOT,digest
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import parse,classify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def table(p,rows):
    if rows:
        with p.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def ci(values,labels):
    if not values:return {'n':0,'mean':None,'ci':None,'clusters':0}
    v=np.asarray(values,float);groups=np.asarray(labels);unique=np.unique(groups)
    sums=np.array([v[groups==g].sum() for g in unique]);sizes=np.array([sum(groups==g) for g in unique])
    rng=np.random.default_rng(20265909);idx=rng.integers(0,len(unique),(10000,len(unique)))
    means=sums[idx].sum(1)/sizes[idx].sum(1)
    return {'n':len(values),'mean':float(v.mean()),'ci':np.quantile(means,[.025,.975]).tolist(),'clusters':len(unique)}

def main():
    out=HERE/'results/analysis';out.mkdir(exist_ok=True)
    cfg=json.loads((HERE/'config.json').read_text());data={r['id']:r for r in read(HERE/'data.jsonl')}
    raw=[];base=[];checks=[]
    def check(name,passed,**extra):checks.append({'check':name,'pass':bool(passed),**extra})
    freeze=json.loads((HERE/'freeze.json').read_text())
    check('frozen_sources_unchanged',all(digest(ROOT/p)==h for p,h in freeze['files'].items()))
    import torch
    for seed in cfg['sampling_seeds']:
        folder=HERE/f'results/seed-{seed}'
        b=read(folder/'baselines.jsonl');rr=read(folder/'interventions.jsonl')
        expected=1140 if seed==20262909 else 1080
        check(f'counts_{seed}',len(b)==360 and len(rr)==expected,baselines=len(b),interventions=len(rr))
        check(f'unique_keys_{seed}',len({r['key'] for r in rr})==len(rr) and len({r['key'] for r in b})==len(b))
        allrows=b+[r for r in rr if r['status']=='generated']
        check(f'parser_{seed}',all(parse(r['text'])==r['parsed'] for r in allrows))
        check(f'token_bookkeeping_{seed}',all(r['output_ids']==r['replayed_output_ids']+r['fresh_output_ids'] and len(r['output_ids'])==r['total_output_tokens']<=512 for r in allrows))
        check(f'scoring_{seed}',all(r['correct']==(r['parsed']['answer']==r['target']) for r in b) and all(all(classify(r['parsed'],data[r['id']]['predictions'])[k]==r[k] for k in ['category','success','prediction_matches']) for r in rr))
        vectors=torch.load(folder/'source-vectors.pt',weights_only=True) if (folder/'source-vectors.pt').exists() else {}
        vv=[r for r in rr if r.get('donor_source_hash')]
        import hashlib
        check(f'vectors_{seed}',all(r['key'] in vectors and hashlib.sha256(vectors[r['key']].float().numpy().tobytes()).hexdigest()==r['donor_source_hash'] for r in vv),checked=len(vv))
        for name in ['native-validation','operator-validation']:
            path=folder/f'{name}.json';check(f'{name}_{seed}',path.exists() and json.loads(path.read_text())['pass'])
        bidx={r['key']:r for r in b}
        identities=[r for r in rr if r['condition'] in ['unpatched','same_example','text_recipient_clamp'] and r['status']=='generated']
        check(f'identities_{seed}',all(r['output_ids']==bidx[f"{r['id']}/recipient_full"]['output_ids'] for r in identities),n=len(identities))
        noise=[r for r in rr if r['condition']=='isotropic_delta' and r['status']=='generated']
        check(f'norms_{seed}',all(abs(r['matched_delta_norm']-r['random_delta_norm'])<max(1e-5,1e-6*r['matched_delta_norm']) for r in noise),n=len(noise))
        raw += [{**r,'seed':seed} for r in rr];base += [{**r,'seed':seed} for r in b]
    audit={'pass':all(c['pass'] for c in checks),'checks':checks,'original_validation_evaluated':False}
    dump(out/'audit.json',audit)
    if not audit['pass']:raise RuntimeError('Artifact audit failed; inspect audit.json')
    idx={(r['seed'],r['id'],r['condition']):r for r in raw}
    summary=[]
    for condition in sorted({r['condition'] for r in raw}):
        for variable,levels in [('all',['all']),('width',[1,2]),('family',['box','register']),('seed',cfg['sampling_seeds'])]:
            for level in levels:
                ss=[r for r in raw if r['condition']==condition and (variable=='all' or r[variable]==level)]
                if not ss:continue
                gen=[r for r in ss if r['status']=='generated']
                summary.append({'condition':condition,'stratum':variable,'level':level,'n':len(ss),'available':len(gen),'counterfactual':sum(r['success'] for r in ss),'complete_donor_z':sum(r['parsed']['intermediate']==r['donor_z'] for r in ss),'unchanged_answer':sum(r['category']=='recipient_answer' for r in ss),'invalid':sum(r['category']=='invalid' for r in ss),'parser_failures_generated':sum(r['parsed']['parser_failure'] for r in gen),'truncations':sum(r['truncated'] for r in ss)})
    table(out/'conditions.csv',summary)
    baseline_summary=[]
    for seed in cfg['sampling_seeds']:
        for family in ['box','register']:
            for condition in ['donor_full','recipient_full','textual_counterfactual']:
                ss=[r for r in base if r['seed']==seed and data[r['id']]['family']==family and r['condition']==condition]
                baseline_summary.append({'seed':seed,'family':family,'condition':condition,'n':len(ss),'correct':sum(r['correct'] for r in ss),'parser_failures':sum(r['parsed']['parser_failure'] for r in ss),'truncations':sum(r['truncated'] for r in ss)})
    table(out/'baselines.csv',baseline_summary)
    paired=[];differences=[]
    for r in raw:
        if r['condition']!='activation':continue
        digit=idx[r['seed'],r['id'],'text_first_digit'];full=idx[r['seed'],r['id'],'text_full_value']
        both=r['status']==digit['status']=='generated'
        row={k:r[k] for k in ['seed','id','group','family','width','leading_digit']}
        row.update(available=r['status']=='generated',joint_digit_available=both,activation_cf=r['success'],digit_cf=digit['success'],full_cf=full['success'],cf_indicator_agree=r['success']==digit['success'],exact_digit_agree=bool(both and r['output_ids']==digit['output_ids']),same_final=bool(both and r['parsed']['answer']==digit['parsed']['answer']),source_next_copy=bool(r['status']=='generated' and r['fresh_output_ids'][0]==r['source_next_token']),oracle_first_digit_copy=bool(r['status']=='generated' and r['fresh_output_ids'][0]==r['oracle_first_digit_token']),competent=all(r[k] for k in ['baseline_donor_correct','baseline_recipient_correct','baseline_counterfactual_correct']))
        paired.append(row)
        if both and not row['exact_digit_agree']:differences.append({'comparison':'activation_vs_digit','row':row,'activation':r,'text':digit})
    table(out/'paired-endpoints.csv',paired)
    metrics={}
    for label,ss in [('all',paired),('competent',[r for r in paired if r['competent']])]:
        for field,selector in [('exact_digit_agree',lambda r:r['joint_digit_available']),('cf_indicator_agree',lambda r:True),('source_next_copy',lambda r:r['available']),('oracle_first_digit_copy',lambda r:r['available']),('two_digit_full_minus_activation',lambda r:r['width']==2)]:
            rows=[r for r in ss if selector(r)]
            vals=[int(r['full_cf'])-int(r['activation_cf']) if field=='two_digit_full_minus_activation' else int(r[field]) for r in rows]
            metrics[f'{label}/{field}']={'program_group':ci(vals,[r['group'] for r in rows]),'leading_digit_sensitivity':ci(vals,[r['leading_digit'] for r in rows])}
    invariance=[];clamps=[]
    for seed in cfg['sampling_seeds']:
        for g in range(30):
            for family in ['box','register']:
                a=idx[seed,f'readout-{g:02}-{family}-w1','activation'];b=idx[seed,f'readout-{g:02}-{family}-w2','activation']
                available=a['status']==b['status']=='generated'
                invariance.append({'seed':seed,'group':g,'family':family,'available':available,'identical_prefix':available and a['prefix_ids']==b['prefix_ids'],'exact_output_agree':available and a['output_ids']==b['output_ids'],'same_final':available and a['parsed']['answer']==b['parsed']['answer'],'distinct_counterfactual_targets':a['predictions']['counterfactual']!=b['predictions']['counterfactual']})
        for rid in data:
            for fixed,ac,tc in [('recipient','activation_recipient_clamp','text_recipient_clamp'),('donor','activation_donor_clamp','text_full_value')]:
                a=idx[seed,rid,ac];b=idx[seed,rid,tc];available=a['status']==b['status']=='generated'
                clamps.append({'seed':seed,'id':rid,'fixed':fixed,'available':available,'exact_output_agree':available and a['output_ids']==b['output_ids'],'same_final':available and a['parsed']['answer']==b['parsed']['answer'],'activation_cf':a['success'],'text_cf':b['success']})
                if available and a['output_ids']!=b['output_ids']:differences.append({'comparison':f'{fixed}_clamp','activation':a,'text':b})
    table(out/'width-source-invariance.csv',invariance);table(out/'clamp-comparisons.csv',clamps)
    dump(out/'nonidentical-continuations.json',differences)
    def rate(key):return metrics[key]['program_group']['mean']
    criteria={'exact_digit_at_least_90pct':rate('all/exact_digit_agree') is not None and rate('all/exact_digit_agree')>=.9,'source_copy_at_least_95pct':rate('all/source_next_copy') is not None and rate('all/source_next_copy')>=.95,'two_digit_full_text_advantage_at_least_10pp':rate('all/two_digit_full_minus_activation') is not None and rate('all/two_digit_full_minus_activation')>=.1}
    result={'metrics':metrics,'replication_criteria':criteria,'all_criteria_pass':all(criteria.values()),'planned_pair_seed_records':240,'program_groups':30,'audit_pass':audit['pass'],'conditions':summary,'baseline_summary':baseline_summary,'width_invariance':{'planned':len(invariance),'available':sum(r['available'] for r in invariance),'exact':sum(r['exact_output_agree'] for r in invariance)},'clamps':{fixed:{'planned':sum(r['fixed']==fixed for r in clamps),'available':sum(r['fixed']==fixed and r['available'] for r in clamps),'exact':sum(r['fixed']==fixed and r['exact_output_agree'] for r in clamps),'same_final':sum(r['fixed']==fixed and r['same_final'] for r in clamps)} for fixed in ['recipient','donor']}}
    dump(out/'summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,width in zip(axes,[1,2]):
        ss=[next(r for r in summary if r['condition']==c and r['stratum']=='width' and r['level']==width) for c in ['activation','text_first_digit','text_full_value']]
        ax.bar(['Activation','First-digit text','Complete-value text'],[100*r['counterfactual']/r['n'] for r in ss],color=['#2563eb','#d97706','#0f766e'])
        ax.set_ylim(0,105);ax.set_ylabel('Counterfactual success (%)');ax.set_title(f'{width}-digit donor values')
        for j,r in enumerate(ss):ax.text(j,100*r['counterfactual']/r['n']+2,f"{r['counterfactual']}/{r['n']}",ha='center')
    fig.suptitle('Fixed-site replication: 30 program groups, two wordings, two seeds')
    fig.savefig(out/'replication.png',dpi=180);fig.savefig(out/'replication.svg');plt.close(fig)
    print(json.dumps({'criteria':criteria,'width_invariance':result['width_invariance'],'clamps':result['clamps'],'metrics':metrics},indent=2))

if __name__=='__main__':main()
