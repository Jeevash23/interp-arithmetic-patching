"""Descriptive mechanism diagnostics on saved development results only."""
import json
from collections import defaultdict,Counter
from analyze import ROOT,read,write_csv,plt
from value_transfer.tasks import load_data,op

def main():
    data={r['id']:r for r in load_data('development')};out=ROOT/'results/v3/analysis';out.mkdir(exist_ok=True)
    screen=read(ROOT/'results/v3/development/screen.jsonl');controls=read(ROOT/'results/v3/development/controls-n100/controls.jsonl')
    collision_rows=[{'id':r['id'],'counterfactual':r['predictions']['counterfactual'],'also_matches':[k for k,v in r['predictions'].items() if k!='counterfactual' and v==r['predictions']['counterfactual']]} for r in data.values()]
    collision_counts=Counter(k for r in collision_rows for k in r['also_matches'])
    successful_overlaps=[{'stage':stage,'key':r['key'],'also_matches':[k for k in r['prediction_matches'] if k!='counterfactual']} for stage,rr in [('screen',screen),('controls',controls)] for r in rr if r['success']]
    (out/'prediction-collisions.json').write_text(json.dumps({'development_n':100,'counterfactual_collisions':dict(collision_counts),'per_pair':collision_rows,'successful_run_overlaps':successful_overlaps,'interpretation':'Full-expression transfer is identical to value transfer under this output assay. Other overlaps remain visible; no case is silently discarded.'},indent=2))
    table=[];examples=[]
    for stage,raw in [('screen',screen),('controls',controls)]:
        groups=defaultdict(list)
        for r in raw:groups[(r['timing'],r['layer'],r['condition'])].append(r)
        for (timing,layer,c),rr in sorted(groups.items()):
            buckets={'all':rr,'discovery_20':[r for r in rr if int(r['id'].split('-')[-1])<20],'additional_80':[r for r in rr if int(r['id'].split('-')[-1])>=20],'donor_z_one_digit':[r for r in rr if abs(data[r['id']]['donor']['z'])<10],'donor_z_two_digits':[r for r in rr if 10<=abs(data[r['id']]['donor']['z'])<100]}
            for subgroup,ss in buckets.items():
                if not ss:continue
                counts=Counter()
                for r in ss:
                    task=data[r['id']];z=r['parsed']['intermediate'];answer=r['parsed']['answer'];source=r.get('actual_source_task',task['donor'])
                    counts['counterfactual']+=r['success'];counts['intermediate_missing']+=z is None
                    counts['intermediate_original_donor_z']+=z==task['donor']['z'];counts['intermediate_actual_source_z']+=z==source['z'];counts['intermediate_recipient_z']+=z==task['recipient']['z']
                    counts['intermediate_donor_answer']+=z==task['donor']['y'];counts['counterfactual_and_printed_donor_z']+=r['success'] and z==task['donor']['z']
                    counts['final_follows_printed_intermediate']+=z is not None and answer is not None and answer==op(z,task['recipient']['k'],task['recipient']['remaining'])
                    counts['donor_answer_as_z']+=answer==op(task['donor']['y'],task['recipient']['k'],task['recipient']['remaining']) if answer is not None else False
                    counts['actual_source_counterfactual']+=answer==op(source['z'],task['recipient']['k'],task['recipient']['remaining']) if answer is not None else False
                table.append({'stage':stage,'timing':timing,'layer':layer,'condition':c,'subgroup':subgroup,'n':len(ss),**counts})
        if stage=='controls':
            for r in raw:
                if r['condition'] in ['donor','donor_pre_final_position']:
                    examples.append({'id':r['id'],'condition':r['condition'],'donor':data[r['id']]['donor'],'recipient':data[r['id']]['recipient'],'predictions':data[r['id']]['predictions'],'parsed':r['parsed'],'category':r['category'],'raw_key':r['key']})
    write_csv(out/'mechanism-diagnostics.csv',table);(out/'mechanism-examples.json').write_text(json.dumps(examples,indent=2))
    variants=read(ROOT/'results/v3/development/controls-n100/control-baselines.jsonl');variant_table=[]
    for c in sorted({r['condition'] for r in variants}):
        rr=[r for r in variants if r['condition']==c]
        variant_table.append({'condition':c,'correct':sum(r['correct'] for r in rr),'n':len(rr),'printed_z_correct':sum(r['parsed']['intermediate']==r['task']['z'] for r in rr) if c!='textual_donor_answer_as_z' else None,'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr)})
    write_csv(out/'control-baselines.csv',variant_table)
    baseline=read(ROOT/'results/v3/development/baselines.jsonl');printed=[]
    for c in ['donor_full','recipient_full']:
        rr=[r for r in baseline if r['condition']==c];side=c.split('_')[0]
        printed.append({'condition':c,'n':len(rr),'printed_z_correct':sum(r['parsed']['intermediate']==data[r['id']][side]['z'] for r in rr),'correct_final_wrong_or_missing_z':[r['id'] for r in rr if r['correct'] and r['parsed']['intermediate']!=data[r['id']][side]['z']]})
    (out/'baseline-printed-values.json').write_text(json.dumps(printed,indent=2))
    # Paired text/activation comparison is descriptive and retains exact output differences.
    text=read(ROOT/'results/v3/development/textual-scratchpad.jsonl');clamp=read(ROOT/'results/v3/development/clamp-mediation.jsonl')
    primary={r['id']:r for r in screen if r['timing']=='pre_z' and r['layer']==26};paired=[]
    for c in ['text_donor_z','text_donor_final']:
        match={r['id']:r for r in text if r['condition']==c and r['timing']=='post_z'}
        ids=sorted(primary.keys()&match.keys())
        if ids:paired.append({'comparison':f'activation vs open-prefix {c}','n':len(ids),'same_final':sum(primary[i]['parsed']['answer']==match[i]['parsed']['answer'] for i in ids),'same_intermediate':sum(primary[i]['parsed']['intermediate']==match[i]['parsed']['intermediate'] for i in ids),'same_cf_indicator':sum(primary[i]['success']==match[i]['success'] for i in ids)})
    for value in ['recipient','donor']:
        a={r['id']:r for r in clamp if r['condition']==f'activation_{value}_z_clamped'};b={r['id']:r for r in clamp if r['condition']==f'text_{value}_z_clamped'};ids=sorted(a.keys()&b.keys())
        if ids:paired.append({'comparison':f'activation vs text only, {value} z line clamped','n':len(ids),'same_final':sum(a[i]['parsed']['answer']==b[i]['parsed']['answer'] for i in ids),'same_output_ids':sum(a[i]['output_ids']==b[i]['output_ids'] for i in ids),'activation_cf':sum(a[i]['success'] for i in ids),'text_cf':sum(b[i]['success'] for i in ids)})
    digit=read(ROOT/'results/v3/development/digit-controls.jsonl');a={r['id']:r for r in controls if r['condition']=='donor'}
    invariance=[]
    for c in ['same_z_decomposition','same_z_different_answer','mismatched_z','different_z_same_answer','random_source']:
        b={r['id']:r for r in controls if r['condition']==c};shared=sorted(a.keys()&b.keys())
        for name,ids in [('all',shared),('different_z_same_first_digit',[i for i in shared if data[i]['donor']['z']!=b[i]['actual_source_task']['z'] and str(data[i]['donor']['z'])[0]==str(b[i]['actual_source_task']['z'])[0]]),('different_first_digit',[i for i in shared if str(data[i]['donor']['z'])[0]!=str(b[i]['actual_source_task']['z'])[0]])]:
            if not ids:continue
            generated=[i for i in ids if a[i]['status']=='generated' and b[i]['status']=='generated'];valid=[i for i in generated if a[i]['parsed']['answer'] is not None and b[i]['parsed']['answer'] is not None]
            invariance.append({'control':c,'subgroup':name,'n':len(ids),'jointly_generated':len(generated),'jointly_valid_final':len(valid),'same_final_among_valid':sum(a[i]['parsed']['answer']==b[i]['parsed']['answer'] for i in valid),'same_output_ids_among_generated':sum(a[i]['output_ids']==b[i]['output_ids'] for i in generated),'original_donor_cf':sum(a[i]['success'] for i in ids),'control_actual_source_cf':sum(b[i].get('actual_source_success',False) for i in ids)})
    write_csv(out/'source-invariance.csv',invariance)
    for c in ['text_first_digit_prefix','text_full_value_line']:
        b={r['id']:r for r in digit if r['condition']==c};common=sorted(a.keys()&b.keys())
        for name,ids in [('all',common),('both_available',[i for i in common if a[i]['status']=='generated' and b[i]['status']=='generated']),('one_digit',[i for i in common if data[i]['donor']['z']<10]),('two_digits',[i for i in common if data[i]['donor']['z']>=10])]:
            if not ids:continue
            generated=[i for i in ids if a[i]['status']=='generated' and b[i]['status']=='generated']
            paired.append({'comparison':f'activation vs {c}','subgroup':name,'n':len(ids),'jointly_generated':len(generated),'same_final_among_generated':sum(a[i]['parsed']['answer']==b[i]['parsed']['answer'] for i in generated),'same_output_ids_among_generated':sum(a[i]['output_ids']==b[i]['output_ids'] for i in generated),'same_cf_indicator':sum(a[i]['success']==b[i]['success'] for i in ids),'activation_cf':sum(a[i]['success'] for i in ids),'text_cf':sum(b[i]['success'] for i in ids)})
    token_audit=json.loads((out/'number-tokenization.json').read_text());digit_ids={str(r['integer']):r['ids'][0] for r in token_audit['all'] if r['integer']<10}
    transports=[]
    for c in ['donor','unpatched','same_example','isotropic_delta','same_z_decomposition','same_z_different_answer','different_z_same_answer','random_source','mismatched_z','donor_distractor','recipient_distractor','donor_pre_final_position']:
        rr=[r for r in controls if r['condition']==c and r['status']=='generated']
        expected=lambda r:data[r['id']]['donor']['y'] if c=='donor_pre_final_position' else r['actual_source_task']['z']
        transports.append({'condition':c,'planned_n':sum(r['condition']==c for r in controls),'available_n':len(rr),'first_digit_matches_source':sum(r['fresh_output_ids'][0]==digit_ids[str(expected(r))[0]] for r in rr) if c!='isotropic_delta' else None,'first_digit_matches_original_donor':sum(r['fresh_output_ids'][0]==digit_ids[str(data[r['id']]['donor']['z'])[0]] for r in rr),'source_numeric_role':'donor final answer' if c=='donor_pre_final_position' else 'no numeric source; norm-matched perturbation' if c=='isotropic_delta' else 'actual-source intermediate'})
    write_csv(out/'first-digit-transport.csv',transports)
    eos=json.loads((ROOT/'results/v3/model-runtime.json').read_text())['eos_ids']
    endings=[r for r in controls if r['condition']=='donor_final_position' and r['status']=='generated']
    (out/'next-token-controls.json').write_text(json.dumps({'after_donor_final_number':{'planned_n':sum(r['condition']=='donor_final_position' for r in controls),'available_n':len(endings),'immediate_eos':sum(r['fresh_output_ids'][0] in eos for r in endings),'single_fresh_token':sum(len(r['fresh_output_ids'])==1 for r in endings)},'interpretation':'The after-number source is a termination/readout control; it is not evidence that a matched before-number source lacks numeric content.'},indent=2))
    (out/'text-activation-comparison.json').write_text(json.dumps(paired,indent=2))
    if len(digit)==200 and len(a)==100:
        labels=['Activation write','First-digit text','Complete-value text'];colors=['#2563eb','#d97706','#059669']
        rr=[list(a.values()),[r for r in digit if r['condition']=='text_first_digit_prefix'],[r for r in digit if r['condition']=='text_full_value_line']]
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        counts=[sum(r['success'] for r in group) for group in rr]
        axes[0].bar(range(3),counts,color=colors);axes[0].set_xticks(range(3),['Activation\nwrite','First-digit\ntext','Complete-value\ntext']);axes[0].set_ylim(0,100);axes[0].set_ylabel('Counterfactual (%)');axes[0].set_title('Complete 100-pair development sample')
        for i,k in enumerate(counts):axes[0].text(i,k+2,f'{k}/100',ha='center')
        for j,(name,predicate) in enumerate([('One digit',lambda z:z<10),('Two digits',lambda z:z>=10)]):
            for i,group in enumerate(rr):
                chosen=[r for r in group if predicate(data[r['id']]['donor']['z'])];k=sum(r['success'] for r in chosen);x=j+(i-1)*.25
                axes[1].bar(x,100*k/len(chosen),width=.23,color=colors[i],label=labels[i] if j==0 else None);axes[1].text(x,100*k/len(chosen)+2,f'{k}/{len(chosen)}',ha='center',fontsize=8)
        axes[1].set_xticks([0,1],['One digit','Two digits']);axes[1].set_ylim(0,100);axes[1].set_title('Exploratory value-length strata');axes[1].legend(fontsize=8,loc='upper right')
        fig.suptitle('A first-digit text intervention closely reproduces the selected activation effect')
        fig.savefig(out/'value-vs-digit.png',dpi=180);fig.savefig(out/'value-vs-digit.svg');plt.close(fig)
    print(f'Development diagnostic tables saved to {out}')
if __name__=='__main__':main()
