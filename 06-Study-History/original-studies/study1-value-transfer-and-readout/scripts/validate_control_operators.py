"""Validate additive matched-noise and control replay at the selected late site."""
from run import Engine,RESULTS,Journal,load_data,dump,torch
from value_transfer.batch import generate_batch

def main():
    engine=Engine();base=Journal(RESULTS/'development/baselines.jsonl');jobs=[];serial=[]
    rules=['add','identity','replace','none']
    for row,rule in zip(load_data('development')[:4],rules):
        d=base.rows[f'{row["id"]}/donor_full'];r=base.rows[f'{row["id"]}/recipient_full']
        dc,rc=engine.cuts(d)['pre_z'],engine.cuts(r)['pre_z']
        dv=engine.capture(dc,[26])[26];rv=engine.capture(rc,[26])[26]
        noise=torch.randn(rv.shape,generator=torch.Generator().manual_seed(20260909));noise=noise/noise.norm()*(dv-rv).norm()
        source=noise if rule=='add' else rv
        job={'prompt_ids':r['prompt_ids'],'prefix_ids':rc,'layer':None if rule=='none' else 26,'source':source,'rule':'replace' if rule=='none' else rule}
        jobs.append(job);serial.append(engine.generate(**job,trace=True))
    batch=generate_batch(engine,jobs,trace=True);checks=[]
    for rule,(a,la),(b,lb) in zip(rules,serial,batch):
        same=a['output_ids']==b['output_ids'];diff=max(float((x-y).abs().max()) for x,y in zip(la,lb)) if same else None
        checks.append({'rule':rule,'serial':a,'batch':b,'same_output_ids':same,'max_logits_abs':diff,'pass':same and diff<=engine.config['native_logits_atol']})
    out={'checks':checks,'pass':all(x['pass'] for x in checks)}
    dump(RESULTS/'smoke/control-operator-validation.json',out);print({'pass':out['pass']},flush=True)
if __name__=='__main__':main()
