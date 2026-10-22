"""Run prespecified crossed-donor, position and matched-noise development controls."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime,timezone

from run import ROOT,RESULTS,Engine,Journal,baseline,load_data,prompt,predictions,op,classify,dump,record_patch,torch

CONDITIONS=['donor','unpatched','same_example','random_source','mismatched_z','isotropic_delta','same_z_decomposition','same_z_different_answer','different_z_same_answer','donor_final_position','first_operand_position','first_token_position','donor_distractor','recipient_distractor']

def first_numeral_position(engine,ids,value):
    text=engine.tokenizer.decode(ids,skip_special_tokens=False)
    m=re.search(r'(?<!\d)'+str(value)+r'(?!\d)',text)
    if m is None: raise ValueError('Operand not found in donor prompt')
    boundaries=[len(engine.tokenizer.decode(ids[:i],skip_special_tokens=False)) for i in range(len(ids)+1)]
    return next(i for i,n in enumerate(boundaries) if n>=m.end())-1

def final_numeral_prefix(engine,record):
    ids=record['output_ids']; text=engine.tokenizer.decode(ids,skip_special_tokens=False)
    search=text
    if '</think>' in text:
        end=text.rfind('</think>')+len('</think>'); search=' '*end+text[end:]
    m=re.search(r'Final:[ \t]*(-?\d+)',search)
    if m is None: return None
    lengths=[len(engine.tokenizer.decode(ids[:i],skip_special_tokens=False)) for i in range(len(ids)+1)]
    end=next(i for i,n in enumerate(lengths) if n>=m.end(1))
    return record['prompt_ids']+ids[:end]

def ensure_variant(engine,journal,row,label,t,recipient=False,distractor=None,kind='full',supplied=None):
    key=f'{row["id"]}/{label}'
    if key not in journal.rows:
        text=prompt(t,row['family'],kind=kind,supplied=supplied,distractor=distractor)
        out=engine.generate(engine.encode(text))
        target=t['y'] if supplied is None else op(supplied,t['k'],t['remaining'])
        journal.put(key,{'id':row['id'],'condition':label,'prompt':text,'target':target,'task':t,'correct':out['parsed']['answer']==target,**out})
    return journal.rows[key]

def run_controls(engine,rows,base,selected,output_dir):
    journal=Journal(output_dir/'controls.jsonl'); variants=Journal(output_dir/'control-baselines.jsonl'); captured={}
    rng=__import__('random').Random(engine.config['seed']+73)
    order=list(range(len(rows)))
    for _ in range(10000):
        rng.shuffle(order)
        if all(i!=j and rows[i]['donor']['z']!=rows[j]['donor']['z'] for i,j in enumerate(order)): break
    else: raise RuntimeError('No random-source derangement found')
    dump(output_dir/'control-plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'conditions':CONDITIONS,'random_source_ids':{r['id']:rows[order[i]]['id'] for i,r in enumerate(rows)},'selected':selected})
    for i,row in enumerate(rows):
        d=base.rows[f'{row["id"]}/donor_full']; r=base.rows[f'{row["id"]}/recipient_full']
        source_examples={name:ensure_variant(engine,variants,row,name,t) for name,t in row['controls'].items()}
        source_examples['donor_distractor']=ensure_variant(engine,variants,row,'donor_distractor',row['donor'],distractor=row['distractor'])
        rec_dist=ensure_variant(engine,variants,row,'recipient_distractor',row['recipient'],distractor=row['distractor'])
        ensure_variant(engine,variants,row,'textual_donor_answer_as_z',row['recipient'],kind='remaining',supplied=row['donor']['y'])
        random_row=rows[order[i]]; random_example=base.rows[f'{random_row["id"]}/donor_full']
        for site in selected:
            timing,layer=site['timing'],site['layer']; dc,rc=engine.cuts(d),engine.cuts(r)
            donor_vec=engine.capture(dc[timing],[layer])[layer] if timing in dc else None
            recipient_vec=engine.capture(rc[timing],[layer])[layer] if timing in rc else None
            for condition in CONDITIONS:
                key=f'{row["id"]}/{timing}/{layer}/{condition}'
                if key in journal.rows: continue
                src=d; rec=r; src_task=row['donor']; vector=donor_vec; rule='replace'; dp=None; prefix_override=None
                if condition=='unpatched':
                    if timing in rc:
                        out=engine.generate(r['prompt_ids'],rc[timing])
                        out={'id':row['id'],'condition':condition,'timing':timing,'layer':layer,'status':'generated',**out,**classify(out['parsed'],row['predictions'])}
                    else: out=record_patch(engine,row,r,d,timing,layer,condition)
                else:
                    if condition=='same_example':
                        src=r;src_task=row['recipient'];vector=recipient_vec
                    elif condition=='random_source':
                        src=random_example;src_task=random_row['donor'];vector=None
                    elif condition in row['controls']:
                        src=source_examples[condition];src_task=row['controls'][condition];vector=None
                    elif condition=='isotropic_delta' and donor_vec is not None and recipient_vec is not None:
                        seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
                        noise=torch.randn(donor_vec.shape,generator=torch.Generator().manual_seed(seed))
                        vector=noise/noise.norm()*(donor_vec-recipient_vec).norm();rule='add'
                    elif condition=='donor_final_position':
                        prefix_override=final_numeral_prefix(engine,d)
                        if prefix_override is not None:
                            vector=engine.capture(prefix_override,[layer])[layer];dp=len(prefix_override)-1
                    elif condition=='first_operand_position':
                        dp=first_numeral_position(engine,d['prompt_ids'],row['donor']['a'])
                        vector=engine.capture(d['prompt_ids'],[layer],dp)[layer]
                    elif condition=='first_token_position':
                        dp=0;vector=engine.capture(d['prompt_ids'],[layer],dp)[layer]
                    elif condition=='donor_distractor':
                        src=source_examples[condition];vector=None
                    elif condition=='recipient_distractor':
                        rec=rec_dist
                    out=record_patch(engine,row,rec,src,timing,layer,condition,source=vector,rule=rule,source_position=dp)
                    if prefix_override is not None:out['donor_prefix_ids']=prefix_override
                actual=predictions(src_task,row['recipient'])
                out.update(actual_source_task=src_task,actual_source_predictions=actual,actual_source_success=out['parsed']['answer']==actual['counterfactual'] if out['parsed']['answer'] is not None else False,
                           actual_source_matches=classify(out['parsed'],actual)['prediction_matches'],distractor=row['distractor'],distractor_copied=out['parsed']['answer']==row['distractor'],distractor_as_z=out['parsed']['answer']==op(row['distractor'],row['recipient']['k'],row['recipient']['remaining']),
                           reference_baseline_answer=rec['parsed']['answer'],reference_baseline_id=rec.get('key'),random_source_id=random_row['id'] if condition=='random_source' else None)
                if condition=='isotropic_delta' and vector is not None and recipient_vec is not None:
                    out['matched_delta_norm']=float((donor_vec-recipient_vec).norm());out['random_delta_norm']=float(vector.norm())
                if vector is not None:captured[key]=vector
                journal.put(key,out)
        if i%5==4:
            torch.save(captured,output_dir/'control-vectors.pt')
            print(f'Controls {i+1}/{len(rows)}',flush=True)
    table=[]
    for site in selected:
        for condition in CONDITIONS:
            rr=[journal.rows[f'{r["id"]}/{site["timing"]}/{site["layer"]}/{condition}'] for r in rows]
            table.append({'timing':site['timing'],'layer':site['layer'],'condition':condition,'n':len(rr),'counterfactual':sum(r['success'] for r in rr),'actual_source_counterfactual':sum(r['actual_source_success'] for r in rr),'categories':dict(Counter(r['category'] for r in rr)),'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr)})
    dump(output_dir/'controls-summary.json',table)
    print(json.dumps(table,indent=2),flush=True)

def main():
    selected=json.loads((RESULTS/'development/screen-summary.json').read_text())['selected_for_controls']
    engine=Engine();rows=load_data('development');base=Journal(RESULTS/'development/baselines.jsonl')
    summary=baseline(engine,rows,base);dump(RESULTS/'development/baselines-summary.json',summary)
    if any(summary[x]['rate']<engine.config['competence_gate'] for x in ['donor_full','recipient_full','textual_counterfactual']):
        dump(RESULTS/'development/competence-gate.json',{'pass':False,'reason':'100-pair baseline competence below threshold','summary':summary})
        print('100-pair competence gate failed; do not run controls',flush=True);return
    run_controls(engine,rows,base,selected,RESULTS/'development')

if __name__=='__main__':main()
