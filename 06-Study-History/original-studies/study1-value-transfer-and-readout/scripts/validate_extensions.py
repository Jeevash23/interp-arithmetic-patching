"""Validate added equation timings and batching before accelerated scientific runs."""
import json
from run import ROOT,RESULTS,Engine,Journal,load_data,dump,torch
from value_transfer.batch import generate_batch

def main():
    assert json.loads((RESULTS/'smoke/gate.json').read_text())['proceed_to_screen']
    engine=Engine();data=load_data('development')[:20];base=Journal(RESULTS/'development/baselines.jsonl')
    native=[];cache_checks=[];patches=[];serial=[]
    for start in range(0,20,4):
        refs=[base.rows[f'{r["id"]}/recipient_full'] for r in data[start:start+4]]
        outs=generate_batch(engine,[{'prompt_ids':r['prompt_ids']} for r in refs])
        for ref,out in zip(refs,outs):
            native.append({'id':ref['id'],'native_equivalent_serial_ids':ref['output_ids'],'batch':out,'pass':out['output_ids']==ref['output_ids']})
        print(f'Batch baseline equivalence {start+len(refs)}/20: {all(r["pass"] for r in native)}',flush=True)
    for row in data[:3]:
        d=base.rows[f'{row["id"]}/donor_full'];r=base.rows[f'{row["id"]}/recipient_full'];dc,rc=engine.cuts(d),engine.cuts(r)
        for timing in ['pre_equation_result','post_equation_result']:
            source=engine.capture(dc[timing],[13])[13]
            a,la=engine.generate(r['prompt_ids'],rc[timing],13,source,trace=True)
            b,lb=engine.generate(r['prompt_ids'],rc[timing],13,source,use_cache=False,trace=True)
            identity=engine.generate(r['prompt_ids'],rc[timing])
            same=a['output_ids']==b['output_ids'];diff=max(float((x-y).abs().max()) for x,y in zip(la,lb)) if same else None
            cache_checks.append({'id':row['id'],'timing':timing,'cached':a,'recompute':b,'identity_ids':identity['output_ids'],'same_ids':same,'max_logits_abs':diff,'pass':same and diff<=engine.config['cache_logits_atol'] and identity['output_ids']==r['output_ids']})
            del la,lb
        print(f'Equation-cache check {row["id"]}: {all(r["pass"] for r in cache_checks)}',flush=True)
    # Mixed-layer writes and mixed prefix lengths within the same batch.
    for i,row in enumerate(data[:4]):
        timing=['pre_equation_result','post_equation_result','pre_z','pre_final'][i];layer=[6,13,20,26][i]
        d=base.rows[f'{row["id"]}/donor_full'];r=base.rows[f'{row["id"]}/recipient_full'];dc,rc=engine.cuts(d),engine.cuts(r)
        source=engine.capture(dc[timing],[layer])[layer]
        job={'prompt_ids':r['prompt_ids'],'prefix_ids':rc[timing],'layer':layer,'source':source}
        patches.append(job);serial.append(engine.generate(**job,trace=True))
    batched=generate_batch(engine,patches,trace=True);paired=[]
    for i,((a,la),(b,lb)) in enumerate(zip(serial,batched)):
        same=a['output_ids']==b['output_ids'];diff=max(float((x-y).abs().max()) for x,y in zip(la,lb)) if same else None
        paired.append({'id':data[i]['id'],'serial':a,'batch':b,'same_ids':same,'max_logits_abs':diff,'pass':same and diff<=engine.config['native_logits_atol'] and b['write']['write_error']==0})
    result={'batch_native_equivalence':native,'equation_cache_checks':cache_checks,'batch_patch_checks':paired,'pass':all(r['pass'] for group in [native,cache_checks,paired] for r in group),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    dump(RESULTS/'smoke/extension-validation.json',result)
    print(json.dumps({'pass':result['pass'],'peak_bytes':result['gpu_peak_allocated_bytes']}),flush=True)

if __name__=='__main__':main()
