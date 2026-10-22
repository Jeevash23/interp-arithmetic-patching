"""Describe all development outcomes and apply the declared progression heuristics."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')


def interval(values):
    """Paired differences averaged within each of 12 equally sized program groups."""
    keys=sorted({g for g,v in values});means=np.array([np.mean([v for g,v in values if g==key]) for key in keys])
    rng=np.random.default_rng(20261002)
    samples=means[rng.integers(0,len(means),(10000,len(means)))].mean(axis=1)
    return dict(mean=float(means.mean()),interval95=np.quantile(samples,[.025,.975]).tolist(),groups=len(keys),
        scope='exploratory program-group percentile bootstrap; not adjusted for selected site or crossed reuse')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='qwen4b');args=ap.parse_args()
    path=HERE/'results'/args.candidate;out=path/'analysis';out.mkdir(parents=True,exist_ok=True)
    data={r['id']:r for r in read(HERE/'development-with-source-controls.jsonl')}
    baseline={r['key']:r for r in read(path/'baselines.jsonl')}
    source={r['key']:r for r in read(path/'source-baselines.jsonl')}
    records=read(path/'controls.jsonl');screen=read(path/'screen.jsonl')
    competence=json.loads((path/'competence.json').read_text())
    selection=json.loads((path/'screen-summary.json').read_text()) if (path/'screen-summary.json').exists() else None
    result=dict(candidate=args.candidate,stage='development',competence=competence,
        n_control_records=len(records),n_screen_records=len(screen),reserve_evaluated=False,
        source_baselines={},sites=[],promising_sites=[])
    for cond in ['same_z_decomposition','same_z_different_answer','different_z_same_answer','same_leading_different_value']:
        result['source_baselines'][cond]={}
        for kind in ['full','first','target_remaining']:
            rr=[r for r in source.values() if r['source_condition']==cond and r['condition']==kind]
            result['source_baselines'][cond][kind]=dict(n=len(rr),correct=sum(r['correct'] for r in rr))
    table=[]
    if selection and records:
        for site in selection['selected']:
            rr=[r for r in records if r['layer']==site['layer'] and r['role']==site['role']]
            by={(r['id'],r['condition']):r for r in rr}
            assert len(by)==len(rr)
            summary={}
            for cond in ['donor','unpatched','same_example','isotropic_delta','random_source',
                'same_z_decomposition','same_z_different_answer','different_z_same_answer','same_leading_different_value','text_first_answer_digit']:
                items=[r for r in rr if r['condition']==cond]
                s=dict(n=len(items),counterfactual=sum(r['counterfactual_success'] for r in items),
                    actual_source_counterfactual=sum(r['actual_source_success'] for r in items),
                    unchanged=sum(r['answer']==r['recipient_task']['y'] for r in items),
                    actual_donor_answer=sum(r['answer']==r['actual_source_predictions']['donor_answer'] for r in items),
                    actual_donor_z=sum(r['answer']==r['actual_source_predictions']['donor_z'] for r in items),
                    operand_transfer=sum(r['answer']==r['actual_source_predictions']['operand_transfer'] for r in items),
                    intermediate_digit_splice=sum(r['answer']==r['digit_alternatives']['intermediate_leading_digit_splice'] for r in items),
                    answer_digit_splice=sum(r['answer']==r['digit_alternatives']['donor_answer_leading_digit_splice'] for r in items),
                    invalid=sum(r['answer'] is None for r in items),
                    cf_recipient_same_first_digit_success=sum(r['counterfactual_success'] and str(r['predictions']['counterfactual'])[0]==str(r['recipient_task']['y'])[0] for r in items))
                if cond in result['source_baselines']:
                    eligible=[r for r in items if source[r['id']+'/'+cond+'/full']['correct'] and source[r['id']+'/'+cond+'/target_remaining']['correct'] and baseline[r['id']+'/recipient_full']['correct']]
                    s.update(triple_competent=len(eligible),competent_actual_source_success=sum(r['actual_source_success'] for r in eligible))
                summary[cond]=s;table.append(dict(layer=site['layer'],role=site['role'],condition=cond,**s))
            differences={}
            for cond in ['isotropic_delta','text_first_answer_digit','same_z_decomposition','same_z_different_answer']:
                differences[cond]=interval([(row['group'],int(by[id,'donor']['counterfactual_success'])-int(by[id,cond]['counterfactual_success'])) for id,row in data.items()])
            n=len(data);d=summary['donor']['counterfactual']/n
            ops={op:dict(n=sum(r['condition']=='donor' and r['recipient_operation']==op for r in rr),
                correct=sum(r['condition']=='donor' and r['recipient_operation']==op and r['counterfactual_success'] for r in rr)) for op in ['add','sub','mul']}
            both=sum(by[id,'donor']['counterfactual_success'] and by[id,'same_leading_different_value']['actual_source_success'] for id in data)
            same_output=sum(by[id,'donor']['output_ids']==by[id,'same_leading_different_value']['output_ids'] for id in data)
            gates=dict(all_planned_controls_present=len(rr)==n*10,
                primary_at_least_20pct=d>=.2,
                noise_advantage_10pp=d-summary['isotropic_delta']['counterfactual']/n>=.1,
                digit_advantage_10pp=d-summary['text_first_answer_digit']['counterfactual']/n>=.1,
                all_recipient_operations=all(v['correct']>0 for v in ops.values()),
                same_z_robustness=all(summary[c]['counterfactual']/n>=max(.2,.8*d) for c in ['same_z_decomposition','same_z_different_answer']),
                different_z_same_answer=summary['different_z_same_answer']['actual_source_counterfactual']/n>=.2 and (summary['different_z_same_answer']['actual_source_counterfactual']-summary['different_z_same_answer']['actual_donor_answer'])/n>=.1 and summary['different_z_same_answer'].get('triple_competent',0)/n>=.8,
                matched_first_digit_both_succeed=both/n>=.2 and summary['same_leading_different_value'].get('triple_competent',0)/n>=.8)
            item=dict(site=site,conditions=summary,paired_differences=differences,operations=ops,
                same_leading_digit_pair=dict(n=n,both_correct=both,exact_output_agreement=same_output),progression=gates,all_progression_pass=all(gates.values()))
            result['sites'].append(item)
            if item['all_progression_pass']:result['promising_sites'].append(site)
    dump(out/'summary.json',result)
    if table:
        fields=list(dict.fromkeys(k for r in table for k in r))
        with (out/'conditions.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    if False and selection:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        roles=['a_end','b_end','first_stage_end','prompt_end'];layers=max(s['layer'] for s in selection['sites'])+1
        matrix=np.array([[next(s['mean_delta_log_ratio'] for s in selection['sites'] if s['layer']==l and s['role']==r) for r in roles] for l in range(layers)])
        fig,ax=plt.subplots(figsize=(8,8),layout='constrained');limit=max(abs(matrix.min()),abs(matrix.max()),.1)
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks(range(4),['First operand','Second operand','First-step end','Answer prefix']);ax.set_ylabel('Block output (zero based)')
        ax.set_title('Development change in complete-answer log probability ratio\n36 pairs in 12 donor groups')
        fig.colorbar(im,ax=ax,label='Counterfactual versus recipient answer, change in nats')
        fig.savefig(out/'layer-position-screen.png',dpi=180);fig.savefig(out/'layer-position-screen.svg');plt.close(fig)
    print(json.dumps({'candidate':args.candidate,'competence_pass':competence['pass_check'],
        'screen_records':len(screen),'controls':len(records),'promising_sites':result['promising_sites'],
        'selected_site_primary_counts':[(r['site']['layer'],r['site']['role'],r['conditions']['donor']['counterfactual']) for r in result['sites']]},indent=2))


if __name__=='__main__':main()
