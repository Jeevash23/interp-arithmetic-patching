"""Staged local experiments; all scientific continuations are freshly generated."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import load_data,prompt,classify,ROOT,op,predictions
from value_transfer.engine import Engine,torch,vector_hash
RESULTS=ROOT/'results/v2'

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)

class Journal:
    def __init__(self,path):
        self.path=path; path.parent.mkdir(parents=True,exist_ok=True)
        self.rows={}
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line); self.rows[r['key']]=r
    def put(self,key,row):
        row={'key':key,**row}
        with self.path.open('a') as f: f.write(json.dumps(row)+'\n'); f.flush()
        self.rows[key]=row
        return row

def baseline(engine,rows,journal):
    specs=[('donor_full','donor','full'),('recipient_full','recipient','full'),('donor_first','donor','first'),('recipient_first','recipient','first'),('donor_remaining','donor','remaining'),('recipient_remaining','recipient','remaining'),('textual_counterfactual','recipient','remaining')]
    for i,row in enumerate(rows):
        for label,side,kind in specs:
            key=f'{row["id"]}/{label}'
            if key in journal.rows: continue
            supplied=row['donor']['z'] if label=='textual_counterfactual' else None
            target=row['predictions']['counterfactual'] if supplied is not None else row[side]['z' if kind=='first' else 'y']
            text=prompt(row[side],row['family'],kind,supplied)
            result=engine.generate(engine.encode(text))
            journal.put(key,{'id':row['id'],'condition':label,'prompt':text,'target':target,'correct':result['parsed']['answer']==target,**result})
        if i%5==4: print(f'Baselines {i+1}/{len(rows)}',flush=True)
    summary={}
    for label,_,_ in specs:
        rr=[journal.rows[f'{r["id"]}/{label}'] for r in rows]
        summary[label]={'correct':sum(x['correct'] for x in rr),'n':len(rr),'rate':sum(x['correct'] for x in rr)/len(rr),'parser_failures':sum(x['parsed']['parser_failure'] for x in rr),'truncations':sum(x['truncated'] for x in rr)}
    return summary

def validate(engine,rows,journal):
    tests=[]; positive=[]; cache_tests=[]
    atol=engine.config['native_logits_atol']
    for i,row in enumerate(rows):
        r=journal.rows[f'{row["id"]}/recipient_full']; d=journal.rows[f'{row["id"]}/donor_full']
        ids=r['prompt_ids']; pos=len(ids)-1; layer=13
        ref=engine.forward(ids,use_cache=False).logits.detach()
        diff={}
        for rule in ['noop','identity','read']:
            with engine.intervention(layer,pos,rule=rule): out=engine.forward(ids,use_cache=False).logits.detach()
            diff[rule]=float((ref-out).abs().max())
        native=engine.native_generate(ids)
        identity=engine.generate(ids,layer=layer,rule='identity')
        vector=engine.capture(ids,[layer])[layer]
        same=engine.generate(ids,layer=layer,source=vector)
        ok=max(diff.values())<=atol and native==r['output_ids']==identity['output_ids']==same['output_ids']
        tests.append({'id':row['id'],'logits_max_abs':diff,'native_ids':native,'custom_ids':r['output_ids'],'identity_ids':identity['output_ids'],'same_example_ids':same['output_ids'],'pass':ok})
        if i<5:
            stats={}
            base=engine.forward(ids,use_cache=True)
            with engine.intervention(layer,pos,rule='zero',stats=stats): changed=engine.forward(ids,use_cache=True)
            layer_deltas=[]
            for l,(a,b) in enumerate(zip(base.past_key_values.layers,changed.past_key_values.layers)):
                layer_deltas.append({'layer':l,'k_max_abs':float((a.keys-b.keys).abs().max()),'v_max_abs':float((a.values-b.values).abs().max())})
            logit_change=float((base.logits-changed.logits).abs().max())
            positive.append({'id':row['id'],'write':stats,'logit_max_abs_change':logit_change,'cache_deltas':layer_deltas,'pass':stats['delta_norm']>0 and stats['write_error']==0 and stats['other_positions_max_change']==0 and logit_change>0.001 and all(x['k_max_abs']==0 and x['v_max_abs']==0 for x in layer_deltas[:layer+1]) and any(x['k_max_abs']>0 or x['v_max_abs']>0 for x in layer_deltas[layer+1:])})
            del base,changed
            dc,rc=engine.cuts(d),engine.cuts(r)
            for timing in engine.config['timings']:
                if timing not in dc or timing not in rc:
                    cache_tests.append({'id':row['id'],'timing':timing,'pass':False,'error':'missing timing prefix'})
                    continue
                vec=engine.capture(dc[timing],[layer])[layer]
                cached,lc=engine.generate(ids,rc[timing],layer,vec,trace=True)
                uncached,ln=engine.generate(ids,rc[timing],layer,vec,use_cache=False,trace=True)
                same_ids=cached['output_ids']==uncached['output_ids']
                maxdiff=max(float((a-b).abs().max()) for a,b in zip(lc,ln)) if same_ids else None
                cache_tests.append({'id':row['id'],'timing':timing,'donor_position':len(dc[timing])-1,'source_hash':vector_hash(vec),'cached':cached,'recompute_fixed_node':uncached,'same_generated_ids':same_ids,'max_logit_abs':maxdiff,'pass':same_ids and maxdiff<=engine.config['cache_logits_atol']})
        print(f'Instrumentation {i+1}/{len(rows)}: {ok}',flush=True)
    result={'native_equivalence':tests,'positive_control':positive,'cache_semantics':cache_tests,'pass':all(x['pass'] for group in [tests,positive,cache_tests] for x in group),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    dump(RESULTS/'smoke/instrumentation.json',result)
    return result

def record_patch(engine,row,r,d,timing,layer,condition,source=None,rule='replace',source_position=None):
    rc,dc=engine.cuts(r),engine.cuts(d)
    if timing not in rc or timing not in dc:
        return {'id':row['id'],'condition':condition,'timing':timing,'layer':layer,'status':'unavailable_timing','success':False,'category':'invalid','prediction_matches':[],'parsed':{'answer':None,'intermediate':None,'parser_failure':True},'truncated':False}
    dp=len(dc[timing])-1 if source_position is None else source_position
    if source is None and rule in ['replace','add']:
        source=engine.capture(dc[timing],[layer],position=dp)[layer]
    result=engine.generate(r['prompt_ids'],rc[timing],layer,source,rule)
    return {'id':row['id'],'condition':condition,'timing':timing,'layer':layer,'status':'generated','donor_position':dp,'donor_prefix_ids':dc[timing],'donor_source_hash':vector_hash(source) if source is not None else None,**result,**classify(result['parsed'],row['predictions'])}

def screen(engine,rows,base):
    j=Journal(RESULTS/'development/screen.jsonl')
    captures={}
    for i,row in enumerate(rows):
        r=base.rows[f'{row["id"]}/recipient_full']; d=base.rows[f'{row["id"]}/donor_full']
        dc=engine.cuts(d)
        for timing in engine.config['timings']:
            vectors=engine.capture(dc[timing],engine.config['layers']) if timing in dc else {}
            captures[(row['id'],timing)]=vectors
            for layer in engine.config['layers']:
                key=f'{row["id"]}/{timing}/{layer}'
                if key in j.rows: continue
                out=record_patch(engine,row,r,d,timing,layer,'donor',source=vectors.get(layer))
                j.put(key,out)
        print(f'Screen {i+1}/{len(rows)}',flush=True)
    torch.save(captures,RESULTS/'development/screen-donor-vectors.pt')
    table=[]
    for ti,timing in enumerate(engine.config['timings']):
        for layer in engine.config['layers']:
            rr=[j.rows[f'{r["id"]}/{timing}/{layer}'] for r in rows]
            table.append({'timing':timing,'layer':layer,'n':len(rr),'counterfactual':sum(x['success'] for x in rr),'categories':dict(Counter(x['category'] for x in rr)),'invalid':sum(x['category']=='invalid' for x in rr),'truncations':sum(x['truncated'] for x in rr),'timing_order':ti})
    ranked=sorted(table,key=lambda x:(-x['counterfactual'],x['invalid'],x['timing_order'],x['layer']))
    dump(RESULTS/'development/screen-summary.json',{'table':table,'selected_for_controls':ranked[:engine.config['max_candidates_to_controls']]})
    print(json.dumps(ranked,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['smoke','screen','baselines']); parser.add_argument('--n',type=int,default=100); args=parser.parse_args()
    invocation={'utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'stage':args.stage,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for pat in ['src/**/*.py','scripts/*.py','configs/*.json','data/*manifest.json'] for p in ROOT.glob(pat)}}
    dump(ROOT/f'logs/invocation-{args.stage}-{datetime.now().strftime("%Y%m%dT%H%M%S")}.json',invocation)
    engine=Engine(); rows=load_data('development'); base=Journal(RESULTS/'development/baselines.jsonl')
    if args.stage=='smoke':
        rows=rows[:20]
        summary=baseline(engine,rows,base); dump(RESULTS/'smoke/baselines-summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        validation=validate(engine,rows,base)
        competence=all(summary[x]['rate']>=engine.config['competence_gate'] for x in ['donor_full','recipient_full','textual_counterfactual'])
        gate={'instrumentation_pass':validation['pass'],'competence_pass':competence,'proceed_to_screen':validation['pass'] and competence}
        dump(RESULTS/'smoke/gate.json',gate); print(json.dumps(gate),flush=True)
    elif args.stage=='screen':
        gate=json.loads((RESULTS/'smoke/gate.json').read_text()); assert gate['proceed_to_screen']
        screen(engine,rows[:20],base)
    else:
        summary=baseline(engine,rows[:args.n],base); dump(RESULTS/'development/baselines-summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
