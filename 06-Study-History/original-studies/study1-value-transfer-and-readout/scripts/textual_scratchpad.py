"""Development companion: textual replacement within the original full-problem trace."""
from run import ROOT,RESULTS,Engine,Journal,load_data,classify,dump
from value_transfer.batch import generate_batch
import json
from collections import Counter

def main():
    assert json.loads((RESULTS/'smoke/extension-validation.json').read_text())['pass']
    engine=Engine();rows=load_data('development')[:20];base=Journal(RESULTS/'development/baselines.jsonl');j=Journal(RESULTS/'development/textual-scratchpad.jsonl')
    for before_name,after_name in [('pre_z','post_z'),('pre_equation_result','post_equation_result')]:
        jobs=[];metas=[]
        for row in rows:
            r=base.rows[f'{row["id"]}/recipient_full'];cuts=engine.cuts(r)
            for condition in ['text_identity','text_donor_z','text_donor_final']:
                key=f'{row["id"]}/{after_name}/{condition}'
                if key in j.rows:continue
                meta={'id':row['id'],'condition':condition,'timing':after_name,'key':key}
                if before_name not in cuts or after_name not in cuts:
                    j.put(key,{**meta,'status':'unavailable_timing','success':False,'category':'invalid','parsed':{'answer':None,'intermediate':None,'parser_failure':True},'truncated':False});continue
                before,after=cuts[before_name],cuts[after_name]
                original=after[len(before):]
                replacement=original if condition=='text_identity' else engine.tokenizer.encode(str(row['donor']['z' if condition=='text_donor_z' else 'y']),add_special_tokens=False)
                modified=before+replacement
                meta.update(original_value_tokens=original,replacement_value_tokens=replacement,changed_token_count=len(replacement)-len(original),baseline_after_prefix_ids=after,original_pair_predictions=row['predictions'],baseline_output_ids=r['output_ids'])
                jobs.append({'prompt_ids':r['prompt_ids'],'prefix_ids':modified});metas.append(meta)
        for offset in range(0,len(jobs),4):
            outs=generate_batch(engine,jobs[offset:offset+4])
            for meta,out in zip(metas[offset:offset+4],outs):
                scored=classify(out['parsed'],meta['original_pair_predictions'])
                j.put(meta['key'],{**meta,**out,**scored,'status':'generated','identity_equal_to_baseline':out['output_ids']==meta['baseline_output_ids'] if meta['condition']=='text_identity' else None})
            print(f'Textual {after_name}: batch {offset//4+1}/{(len(jobs)+3)//4}',flush=True)
    table=[]
    for timing in ['post_z','post_equation_result']:
        for c in ['text_identity','text_donor_z','text_donor_final']:
            rr=[j.rows[f'{r["id"]}/{timing}/{c}'] for r in rows]
            table.append({'timing':timing,'condition':c,'n':20,'available':sum(r['status']=='generated' for r in rr),'counterfactual':sum(r['success'] for r in rr),'categories':dict(Counter(r['category'] for r in rr)),'truncations':sum(r['truncated'] for r in rr),'identity_passes':sum(r.get('identity_equal_to_baseline') is True for r in rr)})
    dump(RESULTS/'development/textual-scratchpad-summary.json',table);print(json.dumps(table,indent=2),flush=True)

if __name__=='__main__':main()
