"""Execute all declared controls at a chosen development site in validated batches."""
import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime,timezone
from run import ROOT,RESULTS,Engine,Journal,load_data,dump,prompt,op,predictions,classify,torch,vector_hash
from controls import CONDITIONS,first_numeral_position,final_numeral_prefix
from baselines_batched import baseline_batched,fill_jobs
from value_transfer.batch import generate_batch

# Amendment 005: a next-number readout control, distinct from an after-number source.
CONDITIONS=CONDITIONS+['donor_pre_final_position']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=100);parser.add_argument('--sites',type=int,default=1);args=parser.parse_args()
    assert json.loads((RESULTS/'smoke/extension-validation.json').read_text())['pass']
    assert json.loads((RESULTS/'smoke/control-operator-validation.json').read_text())['pass']
    selected=json.loads((RESULTS/'development/screen-summary.json').read_text())['selected_for_controls'][:args.sites]
    engine=Engine();rows=load_data('development')[:args.n];base=Journal(RESULTS/'development/baselines.jsonl')
    summary=baseline_batched(engine,rows,base)
    dump(RESULTS/f'development/baselines-summary-n{args.n}.json',summary)
    if any(summary[x]['rate']<engine.config['competence_gate'] for x in ['donor_full','recipient_full','textual_counterfactual']):
        dump(RESULTS/f'development/competence-gate-n{args.n}.json',{'pass':False,'summary':summary});return
    outdir=RESULTS/f'development/controls-n{args.n}';outdir.mkdir(parents=True,exist_ok=True)
    variants=Journal(outdir/'control-baselines.jsonl');journal=Journal(outdir/'controls.jsonl')
    rng=random.Random(engine.config['seed']+73);order=list(range(len(rows)))
    for _ in range(10000):
        rng.shuffle(order)
        if all(rows[i]['donor']['z']!=rows[j]['donor']['z'] for i,j in enumerate(order)):break
    else:raise RuntimeError('No random-source derangement')
    dump(outdir/'control-plan.json',{'utc':datetime.now(timezone.utc).isoformat(),'selected':selected,'n':len(rows),'conditions':CONDITIONS,'random_source_ids':{r['id']:rows[order[i]]['id'] for i,r in enumerate(rows)},'source':'development only','batch_size':4})
    tasks=[]
    for row in rows:
        specs=[(name,t,None,'full',None) for name,t in row['controls'].items()]
        specs += [('donor_distractor',row['donor'],row['distractor'],'full',None),('recipient_distractor',row['recipient'],row['distractor'],'full',None),('textual_donor_answer_as_z',row['recipient'],None,'remaining',row['donor']['y'])]
        for label,t,dist,kind,supplied in specs:
            key=f'{row["id"]}/{label}'
            if key in variants.rows:continue
            text=prompt(t,row['family'],kind,supplied,dist);target=t['y'] if supplied is None else op(supplied,t['k'],t['remaining'])
            tasks.append({'key':key,'id':row['id'],'condition':label,'prompt':text,'task':t,'target':target,'prompt_ids':engine.encode(text)})
    fill_jobs(engine,tasks,variants)
    vector_path=outdir/'control-vectors.pt'
    captured=torch.load(vector_path,weights_only=True) if vector_path.exists() else {}
    for i,row in enumerate(rows):
        d=base.rows[f'{row["id"]}/donor_full'];r=base.rows[f'{row["id"]}/recipient_full']
        random_row=rows[order[i]];random_example=base.rows[f'{random_row["id"]}/donor_full']
        jobs=[];metas=[]
        for site in selected:
            timing,layer=site['timing'],site['layer'];dc,rc=engine.cuts(d),engine.cuts(r)
            vec=engine.capture(dc[timing],[layer])[layer] if timing in dc else None
            rv=engine.capture(rc[timing],[layer])[layer] if timing in rc else None
            for condition in CONDITIONS:
                key=f'{row["id"]}/{timing}/{layer}/{condition}'
                if key in journal.rows:continue
                src=d;rec=r;src_task=row['donor'];source=vec;rule='replace';dp=None;src_prefix=None;actual_layer=layer
                if condition=='unpatched':actual_layer=None;source=None;src=r;src_task=row['recipient']
                elif condition=='same_example':src=r;src_task=row['recipient'];source=rv
                elif condition=='random_source':src=random_example;src_task=random_row['donor'];source=None
                elif condition in row['controls']:src=variants.rows[f'{row["id"]}/{condition}'];src_task=row['controls'][condition];source=None
                elif condition=='donor_distractor':src=variants.rows[f'{row["id"]}/{condition}'];source=None
                elif condition=='recipient_distractor':rec=variants.rows[f'{row["id"]}/{condition}']
                elif condition=='isotropic_delta' and vec is not None and rv is not None:
                    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16);noise=torch.randn(vec.shape,generator=torch.Generator().manual_seed(seed));source=noise/noise.norm()*(vec-rv).norm();rule='add'
                elif condition=='first_operand_position':
                    dp=first_numeral_position(engine,d['prompt_ids'],row['donor']['a']);src_prefix=d['prompt_ids'];source=engine.capture(src_prefix,[layer],dp)[layer]
                elif condition=='first_token_position':dp=0;src_prefix=d['prompt_ids'];source=engine.capture(src_prefix,[layer],0)[layer]
                elif condition=='donor_final_position':
                    src_prefix=final_numeral_prefix(engine,d)
                    if src_prefix is not None:dp=len(src_prefix)-1;source=engine.capture(src_prefix,[layer])[layer]
                elif condition=='donor_pre_final_position':
                    src_prefix=dc.get('pre_final')
                    if src_prefix is not None:dp=len(src_prefix)-1;source=engine.capture(src_prefix,[layer])[layer]
                rec_cuts=engine.cuts(rec);src_cuts=engine.cuts(src)
                available=timing in rec_cuts and (actual_layer is None or src_prefix is not None or timing in src_cuts) and (condition not in ['donor_final_position','donor_pre_final_position'] or src_prefix is not None)
                actual=predictions(src_task,row['recipient'])
                meta={'key':key,'id':row['id'],'condition':condition,'timing':timing,'layer':layer,'actual_source_task':src_task,'actual_source_predictions':actual,'distractor':row['distractor'],'reference_baseline_answer':rec['parsed']['answer'],'reference_baseline_id':rec.get('key'),'random_source_id':random_row['id'] if condition=='random_source' else None}
                meta['donor_answer_as_z_prediction']=op(row['donor']['y'],row['recipient']['k'],row['recipient']['remaining'])
                if not available:
                    journal.put(key,{**meta,'status':'unavailable_timing','success':False,'actual_source_success':False,'category':'invalid','prediction_matches':[],'actual_source_matches':[],'parsed':{'answer':None,'intermediate':None,'parser_failure':True},'truncated':False});continue
                if src_prefix is None:src_prefix=src_cuts[timing]
                if dp is None:dp=len(src_prefix)-1
                if actual_layer is not None and source is None:source=engine.capture(src_prefix,[layer],dp)[layer]
                if source is not None:captured[key]=source
                meta.update(donor_position=dp,donor_prefix_ids=src_prefix,donor_source_hash=vector_hash(source) if source is not None else None)
                if condition=='isotropic_delta' and vec is not None and rv is not None:meta.update(matched_delta_norm=float((vec-rv).norm()),random_delta_norm=float(source.norm()))
                jobs.append({'prompt_ids':rec['prompt_ids'],'prefix_ids':rec_cuts[timing],'layer':actual_layer,'source':source,'rule':rule});metas.append(meta)
        for offset in range(0,len(jobs),4):
            outs=generate_batch(engine,jobs[offset:offset+4])
            for meta,out in zip(metas[offset:offset+4],outs):
                actual=meta['actual_source_predictions'];answer=out['parsed']['answer']
                result={**meta,**out,**classify(out['parsed'],row['predictions']),'status':'generated','actual_source_success':answer==actual['counterfactual'] if answer is not None else False,'actual_source_matches':classify(out['parsed'],actual)['prediction_matches'],'distractor_copied':answer==row['distractor'],'distractor_as_z':answer==op(row['distractor'],row['recipient']['k'],row['recipient']['remaining'])}
                result['donor_answer_as_z_success']=answer==meta['donor_answer_as_z_prediction'] if answer is not None else False
                journal.put(meta['key'],result)
        torch.save(captured,outdir/'control-vectors.pt');print(f'Controls {i+1}/{len(rows)}',flush=True)
    table=[]
    for site in selected:
        for c in CONDITIONS:
            rr=[journal.rows[f'{r["id"]}/{site["timing"]}/{site["layer"]}/{c}'] for r in rows]
            table.append({'timing':site['timing'],'layer':site['layer'],'condition':c,'n':len(rr),'available':sum(r['status']=='generated' for r in rr),'counterfactual':sum(r['success'] for r in rr),'actual_source_counterfactual':sum(r['actual_source_success'] for r in rr),'categories':dict(Counter(r['category'] for r in rr)),'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr)})
    dump(outdir/'controls-summary.json',table);print(json.dumps(table,indent=2),flush=True)

if __name__=='__main__':main()
