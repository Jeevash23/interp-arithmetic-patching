"""Exploratory controlled scratchpad continuation, not a natural mediation estimator."""
import json
from run import Engine,RESULTS,Journal,load_data,dump,classify,vector_hash

CONDITIONS=['text_recipient_z_clamped','text_donor_z_clamped','activation_recipient_z_clamped','activation_donor_z_clamped']

def value_line_end(engine,record,post_z):
    # Include the original line ending so fresh generation cannot append another digit.
    ids=record['prompt_ids']+record['output_ids'];end=len(post_z)
    while end<len(ids):
        end+=1
        if '\n' in engine.tokenizer.decode(ids[len(post_z):end],skip_special_tokens=False):return ids[len(post_z):end]
    raise ValueError('Intermediate line has no newline')

def main():
    engine=Engine();base=Journal(RESULTS/'development/baselines.jsonl');journal=Journal(RESULTS/'development/clamp-mediation.jsonl')
    selected_checks=[];development=load_data('development')
    for index in [0,15,50]:
        row=development[index];d=base.rows[f'{row["id"]}/donor_full'];r=base.rows[f'{row["id"]}/recipient_full']
        dc,rc=engine.cuts(d)['pre_z'],engine.cuts(r)['pre_z'];source=engine.capture(dc,[26])[26]
        job={'prompt_ids':r['prompt_ids'],'prefix_ids':rc,'layer':26,'source':source}
        a,la=engine.generate(**job,trace=True);b,lb=engine.generate(**job,trace=True,use_cache=False)
        same=a['output_ids']==b['output_ids'];diff=max(float((x-y).abs().max()) for x,y in zip(la,lb)) if same else None
        selected_checks.append({'id':row['id'],'timing':'pre_z','layer':26,'same_ids':same,'max_logits_abs':diff,'pass':same and diff<=engine.config['cache_logits_atol'],'cached':a,'recompute':b})
    dump(RESULTS/'smoke/selected-site-cache-validation.json',{'checks':selected_checks,'pass':all(c['pass'] for c in selected_checks)})
    if not all(c['pass'] for c in selected_checks):raise RuntimeError('Selected-site cache validation failed')
    checks=[]
    for i,row in enumerate(load_data('development')[:20]):
        d=base.rows[f'{row["id"]}/donor_full'];r=base.rows[f'{row["id"]}/recipient_full'];dc,rc=engine.cuts(d),engine.cuts(r)
        source=engine.capture(dc['pre_z'],[26])[26];position=len(rc['pre_z'])-1
        delimiter=value_line_end(engine,r,rc['post_z'])
        for condition in CONDITIONS:
            key=f'{row["id"]}/{condition}'
            if key in journal.rows:continue
            recipient_value='recipient_z' in condition;activation=condition.startswith('activation_')
            prefix=(rc['post_z'] if recipient_value else rc['pre_z']+engine.tokenizer.encode(str(row['donor']['z']),add_special_tokens=False))+delimiter
            job={'prompt_ids':r['prompt_ids'],'prefix_ids':prefix,'layer':26 if activation else None,'source':source,'position':position}
            if i<2 and activation:
                out,logits=engine.generate(**job,trace=True);ref,ref_logits=engine.generate(**job,trace=True,use_cache=False)
                same=out['output_ids']==ref['output_ids'];diff=max(float((a-b).abs().max()) for a,b in zip(logits,ref_logits)) if same else None
                checks.append({'id':row['id'],'condition':condition,'same_ids':same,'max_logits_abs':diff,'pass':same and diff<=engine.config['cache_logits_atol']})
                dump(RESULTS/'smoke/clamp-cache-validation.json',{'checks':checks,'pass':all(c['pass'] for c in checks)})
                if not checks[-1]['pass']:raise RuntimeError('Fixed pre-z-node clamp semantics failed')
            else:out=engine.generate(**job)
            if condition=='text_recipient_z_clamped' and out['output_ids']!=r['output_ids']:raise RuntimeError('Clamped identity differs from original baseline')
            journal.put(key,{'id':row['id'],'condition':condition,'timing':'after_intermediate_line_with_pre_z_write','layer':26 if activation else None,'donor_position':len(dc['pre_z'])-1,'donor_prefix_ids':dc['pre_z'],'donor_source_hash':vector_hash(source) if activation else None,'clamped_value':row['recipient']['z'] if recipient_value else row['donor']['z'],'delimiter_ids':delimiter,'original_pair_predictions':row['predictions'],'status':'generated',**out,**classify(out['parsed'],row['predictions'])})
        print(f'Clamped continuations {i+1}/20',flush=True)
    table=[]
    for c in CONDITIONS:
        rr=[r for r in journal.rows.values() if r['condition']==c]
        table.append({'condition':c,'n':len(rr),'counterfactual':sum(r['success'] for r in rr),'recipient_answer':sum(r['category']=='recipient_answer' for r in rr),'invalid':sum(r['category']=='invalid' for r in rr),'truncations':sum(r['truncated'] for r in rr)})
    dump(RESULTS/'development/clamp-mediation-summary.json',table);print(json.dumps(table,indent=2))
if __name__=='__main__':main()
