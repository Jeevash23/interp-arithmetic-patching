"""Separate first-digit substitution from completed-value replacement on all development pairs."""
import json
from collections import Counter
from run import Engine,RESULTS,Journal,load_data,dump,classify
from value_transfer.batch import generate_batch
from clamp_mediation import value_line_end

def main():
    engine=Engine();rows=load_data('development');base=Journal(RESULTS/'development/baselines.jsonl');journal=Journal(RESULTS/'development/digit-controls.jsonl')
    jobs=[];metas=[]
    for row in rows:
        r=base.rows[f'{row["id"]}/recipient_full'];rc=engine.cuts(r)
        for c in ['text_first_digit_prefix','text_full_value_line']:
            key=f'{row["id"]}/{c}'
            if key in journal.rows:continue
            meta={'id':row['id'],'condition':c,'key':key,'original_pair_predictions':row['predictions'],'donor_z':row['donor']['z'],'donor_z_digits':len(str(row['donor']['z']))}
            if 'pre_z' not in rc or 'post_z' not in rc:
                journal.put(key,{**meta,'status':'unavailable_timing','success':False,'category':'invalid','prediction_matches':[],'parsed':{'answer':None,'intermediate':None,'parser_failure':True},'truncated':False});continue
            number=str(row['donor']['z'])[:1] if c=='text_first_digit_prefix' else str(row['donor']['z'])
            inserted=engine.tokenizer.encode(number,add_special_tokens=False)
            delimiter=[] if c=='text_first_digit_prefix' else value_line_end(engine,r,rc['post_z'])
            meta.update(inserted_ids=inserted,delimiter_ids=delimiter)
            jobs.append({'prompt_ids':r['prompt_ids'],'prefix_ids':rc['pre_z']+inserted+delimiter});metas.append(meta)
    for offset in range(0,len(jobs),4):
        outs=generate_batch(engine,jobs[offset:offset+4])
        for meta,out in zip(metas[offset:offset+4],outs):journal.put(meta['key'],{**meta,**out,**classify(out['parsed'],meta['original_pair_predictions']),'status':'generated'})
        print(f'Digit controls batch {offset//4+1}/{(len(jobs)+3)//4}',flush=True)
    table=[]
    for c in ['text_first_digit_prefix','text_full_value_line']:
        rr=[r for r in journal.rows.values() if r['condition']==c]
        for name,ss in [('all',rr),('one_digit',[r for r in rr if r['donor_z_digits']==1]),('two_digits',[r for r in rr if r['donor_z_digits']==2])]:
            table.append({'condition':c,'subgroup':name,'n':len(ss),'available':sum(r['status']=='generated' for r in ss),'counterfactual':sum(r['success'] for r in ss),'categories':dict(Counter(r['category'] for r in ss)),'truncations':sum(r['truncated'] for r in ss)})
    dump(RESULTS/'development/digit-controls-summary.json',table);print(json.dumps(table,indent=2))
if __name__=='__main__':main()
