"""Analyze saved generations; never runs a model or opens validation for selection."""
import csv
import json
import os
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import load_data,parse,classify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def clustered_interval(values,labels,seed=20260909,reps=5000):
    """One-axis cluster bootstrap of the mean, with all repeated pairs kept together."""
    values=np.asarray(values,dtype=float); labels=np.asarray([str(x) for x in labels])
    unique=np.unique(labels)
    sums=np.array([values[labels==u].sum() for u in unique])
    sizes=np.array([(labels==u).sum() for u in unique])
    if len(unique)<2:return {'mean':float(values.mean()),'ci':None,'clusters':len(unique)}
    rng=np.random.default_rng(seed); indices=rng.integers(0,len(unique),(reps,len(unique)))
    estimates=sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)
    return {'mean':float(values.mean()),'ci':np.quantile(estimates,[.025,.975]).tolist(),'clusters':len(unique)}

def summarize(branch='v3'):
    results=ROOT/'results'/branch;out=results/'analysis';out.mkdir(parents=True,exist_ok=True)
    data={r['id']:r for r in load_data('development')}
    base=read(results/'development/baselines.jsonl')
    baseline_table=[]
    for condition in sorted({r['condition'] for r in base}):
        rr=[r for r in base if r['condition']==condition]
        baseline_table.append({'condition':condition,'correct':sum(r['correct'] for r in rr),'n':len(rr),'percent':100*np.mean([r['correct'] for r in rr]),'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr),'mean_tokens':float(np.mean([r['total_output_tokens'] for r in rr]))})
    write_csv(out/'baselines.csv',baseline_table)
    for stage in ['screen','controls']:
        raw=read(results/f'development/{stage}.jsonl')
        if stage=='controls' and not raw:
            candidates=sorted((results/'development').glob('controls-n*/controls.jsonl'),key=lambda p:int(p.parent.name.split('n')[-1]))
            if candidates:raw=read(candidates[-1])
        groups=defaultdict(list)
        for r in raw:groups[(r['timing'],r['layer'],r['condition'])].append(r)
        table=[]
        for (timing,layer,condition),rr in sorted(groups.items()):
            cats=Counter(r['category'] for r in rr)
            matches=Counter(k for r in rr for k in r.get('prediction_matches',[]))
            actual_matches=Counter(k for r in rr for k in r.get('actual_source_matches',[]))
            generated=[r for r in rr if r.get('status')!='unavailable_timing']
            table.append({'timing':timing,'layer':layer,'condition':condition,'n':len(rr),'available':len(generated),'unavailable':len(rr)-len(generated),'counterfactual':sum(r['success'] for r in rr),'percent':100*np.mean([r['success'] for r in rr]),'recipient_answer':cats['recipient_answer'],'donor_answer':cats['donor_answer'],'donor_z':cats['donor_z'],'operand_transfer':cats['operand_transfer'],'invalid':cats['invalid'],'other_numeric':cats['other_numeric'],'other_alternative':sum(v for k,v in cats.items() if k not in ['counterfactual','recipient_answer','donor_answer','donor_z','operand_transfer','invalid','other_numeric']),'parser_failures':sum(r['parsed']['parser_failure'] for r in generated),'truncations':sum(r['truncated'] for r in rr),'actual_source_counterfactual':sum(r.get('actual_source_success',False) for r in rr),'donor_answer_as_z':sum(r.get('donor_answer_as_z_success',False) for r in rr),'distractor_copied':sum(r.get('distractor_copied',False) for r in rr),'distractor_as_z':sum(r.get('distractor_as_z',False) for r in rr),'all_prediction_matches':json.dumps(matches,sort_keys=True),'actual_source_prediction_matches':json.dumps(actual_matches,sort_keys=True)})
        write_csv(out/f'{stage}.csv',table)
        if stage=='screen' and table:
            config=json.loads((ROOT/'configs/pilot.json').read_text())
            vals=np.array([[next((x['percent'] for x in table if x['timing']==t and x['layer']==l),np.nan) for l in config['layers']] for t in config['timings']])
            invalids=np.array([[next((100*x['invalid']/x['n'] for x in table if x['timing']==t and x['layer']==l),np.nan) for l in config['layers']] for t in config['timings']])
            labels={'prompt_end':'Prompt end','pre_z':'Before labelled intermediate','post_z':'After labelled intermediate','pre_final':'Before labelled final answer','pre_equation_result':'Before first equation result','post_equation_result':'After first equation result'}
            fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True,layout='constrained')
            for ax,matrix,title,cmap in zip(axes,[vals,invalids],['Counterfactual','Invalid / unavailable'],['Blues','Oranges']):
                im=ax.imshow(matrix,vmin=0,vmax=100,cmap=cmap,aspect='auto')
                ax.set_xticks(range(len(config['layers'])),config['layers']);ax.set_yticks(range(len(config['timings'])),[labels[t] for t in config['timings']])
                ax.set_xlabel('Decoder block (zero-based)');ax.set_title(title)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):ax.text(j,i,f'{matrix[i,j]/5:.0f}/20',ha='center',va='center',color='white' if matrix[i,j]>60 else 'black')
                fig.colorbar(im,ax=ax,label='Percent',shrink=.8)
            fig.suptitle('Exploratory English screen: every candidate, complete denominators')
            fig.savefig(out/'screen.png',dpi=180);fig.savefig(out/'screen.svg');plt.close(fig)
        if stage=='controls' and table:
            paired=[];competent={}
            for row in data.values():
                rr=[r for r in base if r['id']==row['id'] and r['condition'] in ['donor_full','recipient_full','textual_counterfactual']]
                competent[row['id']]=len(rr)==3 and all(r['correct'] for r in rr)
            for timing,layer in sorted({(r['timing'],r['layer']) for r in raw}):
                primary={r['id']:r for r in groups[(timing,layer,'donor')]}
                for condition in sorted({r['condition'] for r in raw} - {'donor'}):
                    control={r['id']:r for r in groups[(timing,layer,condition)]}
                    ids=sorted(primary.keys()&control.keys())
                    for subset in ['all','baseline_competent']:
                        chosen=ids if subset=='all' else [i for i in ids if competent[i]]
                        if not chosen:continue
                        delta=np.array([int(primary[i]['success'])-int(control[i]['success']) for i in chosen])
                        axes={'donor_z':[data[i]['donor']['z'] for i in chosen],'recipient_z':[data[i]['recipient']['z'] for i in chosen],'structure':[data[i]['structure'] for i in chosen],'recipient_program':[json.dumps(data[i]['recipient'],sort_keys=True) for i in chosen]}
                        cis={name:clustered_interval(delta,labels) for name,labels in axes.items()}
                        valid=[v['ci'] for v in cis.values() if v['ci'] is not None]
                        paired.append({'timing':timing,'layer':layer,'control':condition,'subset':subset,'n':len(chosen),'coverage':len(chosen)/len(ids),'primary_success':sum(primary[i]['success'] for i in chosen),'control_success':sum(control[i]['success'] for i in chosen),'paired_difference':float(delta.mean()),'uncertainty_sensitivity':cis,'interval_envelope':[min(c[0] for c in valid),max(c[1] for c in valid)] if valid else None,'discordant_primary_only':int(sum(d==1 for d in delta)),'discordant_control_only':int(sum(d==-1 for d in delta))})
            (out/'paired-analysis.json').write_text(json.dumps(paired,indent=2))
            # No claim that the envelope is an exact simultaneous/multiway interval.
            strata=[]
            for (timing,layer,condition),rr in sorted(groups.items()):
                for variable in ['family','remaining']:
                    buckets=defaultdict(list)
                    for r in rr:buckets[data[r['id']]['family'] if variable=='family' else data[r['id']]['recipient']['remaining']].append(r)
                    for label,ss in buckets.items():strata.append({'timing':timing,'layer':layer,'condition':condition,'stratum_variable':variable,'stratum':label,'n':len(ss),'counterfactual':sum(r['success'] for r in ss),'percent':100*np.mean([r['success'] for r in ss])})
            write_csv(out/'strata.csv',strata)
            fig,axes=plt.subplots(1,len({(x['timing'],x['layer']) for x in table}),figsize=(12,6),squeeze=False,layout='constrained')
            for ax,site in zip(axes[0],sorted({(x['timing'],x['layer']) for x in table})):
                tt=[x for x in table if (x['timing'],x['layer'])==site]
                ax.barh([x['condition'].replace('_',' ') for x in tt],[x['percent'] for x in tt],color=['#2563eb' if x['condition']=='donor' else '#94a3b8' for x in tt]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Original-pair counterfactual (%)');ax.set_title(f'{site[0]}, block {site[1]} — exploratory development')
                for i,x in enumerate(tt):ax.text(x['percent']+1,i,f'{x["counterfactual"]}/{x["n"]}',va='center',fontsize=9)
            fig.savefig(out/'controls.png',dpi=180);fig.savefig(out/'controls.svg');plt.close(fig)
    print(f'Analysis saved to {out}')

if __name__=='__main__':summarize(sys.argv[1] if len(sys.argv)>1 else 'v3')
