"""Complete the declared 24-site screen after validated batch equivalence."""
import json
from collections import Counter
from datetime import datetime,timezone
from run import ROOT,RESULTS,Engine,Journal,load_data,dump,classify,torch,vector_hash
from value_transfer.batch import generate_batch

TIMINGS=['prompt_end','pre_z','post_z','pre_final','pre_equation_result','post_equation_result']

def main():
    validation=json.loads((RESULTS/'smoke/extension-validation.json').read_text());assert validation['pass']
    engine=Engine();rows=load_data('development')[:20];base=Journal(RESULTS/'development/baselines.jsonl');j=Journal(RESULTS/'development/screen.jsonl')
    vector_path=RESULTS/'development/screen-batched-donor-vectors.pt'
    captures=torch.load(vector_path,weights_only=True) if vector_path.exists() else {}
    dump(RESULTS/'development/expanded-screen-plan.json',{'utc':datetime.now(timezone.utc).isoformat(),'timings':TIMINGS,'layers':engine.config['layers'],'n':20,'batch_size':4,'total_candidates':24,'selection_rule':'counterfactual count descending, invalid count ascending, timing order, layer order','amendment':'003'})
    for i,row in enumerate(rows):
        r=base.rows[f'{row["id"]}/recipient_full'];d=base.rows[f'{row["id"]}/donor_full'];rc,dc=engine.cuts(r),engine.cuts(d)
        for timing in TIMINGS:
            missing=[l for l in engine.config['layers'] if f'{row["id"]}/{timing}/{l}' not in j.rows]
            if not missing:continue
            if timing not in rc or timing not in dc:
                for layer in missing:
                    j.put(f'{row["id"]}/{timing}/{layer}',{'id':row['id'],'condition':'donor','timing':timing,'layer':layer,'status':'unavailable_timing','success':False,'category':'invalid','prediction_matches':[],'parsed':{'answer':None,'intermediate':None,'parser_failure':True},'truncated':False})
                continue
            vectors=engine.capture(dc[timing],missing)
            jobs=[{'prompt_ids':r['prompt_ids'],'prefix_ids':rc[timing],'layer':l,'source':vectors[l]} for l in missing]
            for layer,out in zip(missing,generate_batch(engine,jobs)):
                key=f'{row["id"]}/{timing}/{layer}';captures[key]=vectors[layer]
                j.put(key,{'id':row['id'],'condition':'donor','timing':timing,'layer':layer,'status':'generated','donor_position':len(dc[timing])-1,'donor_prefix_ids':dc[timing],'donor_source_hash':vector_hash(vectors[layer]),**out,**classify(out['parsed'],row['predictions'])})
        torch.save(captures,vector_path)
        print(f'Expanded screen {i+1}/20',flush=True)
    table=[]
    for ti,timing in enumerate(TIMINGS):
        for layer in engine.config['layers']:
            rr=[j.rows[f'{r["id"]}/{timing}/{layer}'] for r in rows]
            table.append({'timing':timing,'layer':layer,'n':len(rr),'available':sum(r['status']=='generated' for r in rr),'counterfactual':sum(r['success'] for r in rr),'categories':dict(Counter(r['category'] for r in rr)),'invalid':sum(r['category']=='invalid' for r in rr),'truncations':sum(r['truncated'] for r in rr),'timing_order':ti})
    ranked=sorted(table,key=lambda r:(-r['counterfactual'],r['invalid'],r['timing_order'],r['layer']))
    dump(RESULTS/'development/screen-summary.json',{'table':table,'selected_for_controls':ranked[:2]})
    print(json.dumps(ranked,indent=2),flush=True)

if __name__=='__main__':main()
