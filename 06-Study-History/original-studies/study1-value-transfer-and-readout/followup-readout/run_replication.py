"""Frozen fixed-site replication; append-only outputs and explicit reuse."""
import hashlib
import json
import sys
from pathlib import Path
from prepare import HERE,ROOT,prompt,digest
sys.path.insert(0,str(ROOT/'scripts'))
from run import Engine,Journal,dump,classify,torch,vector_hash
from baselines_batched import fill_jobs
from value_transfer.batch import generate_batch
from clamp_mediation import value_line_end
from controls import final_numeral_prefix

CONDITIONS=['activation','unpatched','same_example','isotropic_delta','text_first_digit','text_full_value','text_recipient_clamp','activation_recipient_clamp','activation_donor_clamp']

def unavailable(meta,reason):
    return {**meta,'status':'unavailable_timing','reason':reason,'success':False,'category':'invalid','prediction_matches':[],'parsed':{'answer':None,'intermediate':None,'parser_failure':True},'truncated':False}

def baseline_tasks(engine,rows,journal):
    tasks=[]
    for row in rows:
        for condition,side in [('donor_full','donor'),('recipient_full','recipient'),('textual_counterfactual','recipient')]:
            key=f"{row['id']}/{condition}"
            if key in journal.rows:continue
            supplied=row['donor']['z'] if condition=='textual_counterfactual' else None
            text=prompt(row[side],row['family'],'remaining' if supplied is not None else 'full',supplied)
            tasks.append({'key':key,'id':row['id'],'condition':condition,'prompt':text,'task':row[side],'target':row['predictions']['counterfactual'] if supplied is not None else row[side]['y'],'prompt_ids':engine.encode(text)})
    fill_jobs(engine,tasks,journal)

def verify_native(engine,row,base):
    checks=[]
    for c in ['donor_full','recipient_full']:
        r=base.rows[f"{row['id']}/{c}"];ids=engine.native_generate(r['prompt_ids'])
        checks.append({'condition':c,'id':row['id'],'pass':ids==r['output_ids'],'native_ids':ids,'batch_ids':r['output_ids']})
    return checks

def run_controls(engine,rows,base,outdir,seed):
    journal=Journal(outdir/'interventions.jsonl');vp=outdir/'source-vectors.pt'
    vectors=torch.load(vp,weights_only=True) if vp.exists() else {}
    checks=[]
    for ix,row in enumerate(rows):
        d=base.rows[f"{row['id']}/donor_full"];r=base.rows[f"{row['id']}/recipient_full"]
        dc,rc=engine.cuts(d),engine.cuts(r)
        src=engine.capture(dc['pre_z'],[26])[26] if 'pre_z' in dc else None
        rec=engine.capture(rc['pre_z'],[26])[26] if 'pre_z' in rc else None
        conditions=list(CONDITIONS)
        if seed==20262909 and row['width']==2 and row['family']=='box':conditions+=['donor_pre_final','donor_after_final']
        jobs=[];metas=[];serial=[]
        for condition in conditions:
            key=f"{row['id']}/{condition}"
            if key in journal.rows:continue
            meta={**{k:row[k] for k in ['id','group','family','width','leading_digit']},'seed':seed,'condition':condition,'key':key,'predictions':row['predictions'],'donor_z':row['donor']['z'],'recipient_z':row['recipient']['z'],'baseline_donor_correct':d['correct'],'baseline_recipient_correct':r['correct'],'baseline_counterfactual_correct':base.rows[f"{row['id']}/textual_counterfactual"]['correct']}
            if 'pre_z' not in rc:
                journal.put(key,unavailable(meta,'recipient_pre_z_missing'));continue
            prefix=rc['pre_z'];layer=None;source=None;rule='replace';position=len(prefix)-1;sp=None
            if condition in ['activation','isotropic_delta','activation_recipient_clamp','activation_donor_clamp']:
                layer=26;source=src;sp=dc.get('pre_z')
            elif condition=='same_example':layer=26;source=rec;sp=rc['pre_z']
            elif condition in ['donor_pre_final','donor_after_final']:
                sp=dc.get('pre_final') if condition=='donor_pre_final' else final_numeral_prefix(engine,d)
                layer=26;source=engine.capture(sp,[26])[26] if sp is not None else None
            if layer is not None and source is None:
                journal.put(key,unavailable(meta,'source_position_missing'));continue
            if condition=='isotropic_delta':
                noise_seed=int(hashlib.sha256(f'{seed}/{key}'.encode()).hexdigest()[:8],16)
                noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(noise_seed))
                norm=(src-rec).norm();source=noise/noise.norm()*norm;rule='add'
                meta.update(matched_delta_norm=float(norm),random_delta_norm=float(source.norm()),noise_seed=noise_seed)
            if condition=='text_first_digit':prefix=prefix+engine.tokenizer.encode(str(row['leading_digit']),add_special_tokens=False)
            if condition in ['text_full_value','text_recipient_clamp','activation_recipient_clamp','activation_donor_clamp']:
                if 'post_z' not in rc:
                    journal.put(key,unavailable(meta,'recipient_post_z_missing'));continue
                try:delimiter=value_line_end(engine,r,rc['post_z'])
                except ValueError:
                    journal.put(key,unavailable(meta,'recipient_line_boundary_missing'));continue
                recipient_fixed=condition in ['text_recipient_clamp','activation_recipient_clamp']
                prefix=(rc['post_z'] if recipient_fixed else rc['pre_z']+engine.tokenizer.encode(str(row['donor']['z']),add_special_tokens=False))+delimiter
                meta.update(delimiter_ids=delimiter,clamped_value=r['parsed']['intermediate'] if recipient_fixed else row['donor']['z'])
            if len(prefix)-len(r['prompt_ids'])>=engine.config['max_new_tokens']:
                journal.put(key,unavailable(meta,'no_fresh_generation_budget'));continue
            if source is not None:
                vectors[key]=source;meta.update(donor_prefix_ids=sp,donor_source_hash=vector_hash(source))
                if sp is not None:
                    offset=len(sp)-len(d['prompt_ids'] if condition!='same_example' else r['prompt_ids'])
                    sequence=(d if condition!='same_example' else r)['output_ids']
                    meta['source_next_token']=sequence[offset] if offset<len(sequence) else None
            meta['oracle_first_digit_token']=engine.tokenizer.encode(str(row['leading_digit']),add_special_tokens=False)[0]
            job={'prompt_ids':r['prompt_ids'],'prefix_ids':prefix,'layer':layer,'source':source,'rule':rule}
            if condition in ['activation_recipient_clamp','activation_donor_clamp']:
                serial.append((meta,{**job,'position':position}))
            else:jobs.append(job);metas.append(meta)
        torch.save(vectors,vp)
        def store(meta,out):
            if meta['condition'] in ['unpatched','same_example','text_recipient_clamp'] and out['output_ids']!=r['output_ids']:
                dump(outdir/'identity-failure.json',{'meta':meta,'out':out,'baseline':r});raise RuntimeError('Identity mismatch')
            journal.put(meta['key'],{**meta,**out,**classify(out['parsed'],row['predictions']),'status':'generated'})
        for offset in range(0,len(jobs),4):
            chunk=jobs[offset:offset+4];mm=metas[offset:offset+4];outs=generate_batch(engine,chunk)
            for meta,job,out in zip(mm,chunk,outs):
                if ix==0 and meta['condition'] in ['activation','same_example','isotropic_delta','text_first_digit']:
                    ref=engine.generate(**job);ok=out['output_ids']==ref['output_ids']
                    checks.append({'condition':meta['condition'],'kind':'batch_serial','pass':ok})
                    if not ok:dump(outdir/'operator-failure.json',{'out':out,'reference':ref});raise RuntimeError('Batch serial mismatch')
                store(meta,out)
        for meta,job in serial:
            if ix==0:
                out,la=engine.generate(**job,trace=True);ref,lb=engine.generate(**job,trace=True,use_cache=False)
                same=out['output_ids']==ref['output_ids'];diff=max(float((a-b).abs().max()) for a,b in zip(la,lb)) if same else None
                checks.append({'condition':meta['condition'],'kind':'fixed_historical_clamp_cache','same_ids':same,'max_logit_abs':diff,'pass':same and diff<=.0005})
                if not checks[-1]['pass']:dump(outdir/'cache-failure.json',checks);raise RuntimeError('Clamp cache mismatch')
            else:out=engine.generate(**job)
            store(meta,out)
        if ix==0 and src is not None and 'pre_z' in rc:
            job={'prompt_ids':r['prompt_ids'],'prefix_ids':rc['pre_z'],'layer':26,'source':src}
            a,la=engine.generate(**job,trace=True);b,lb=engine.generate(**job,use_cache=False,trace=True)
            same=a['output_ids']==b['output_ids'];diff=max(float((x-y).abs().max()) for x,y in zip(la,lb)) if same else None
            checks.append({'kind':'fixed_node_cache','same_ids':same,'max_logit_abs':diff,'pass':same and diff<=.0005})
            if not checks[-1]['pass']:dump(outdir/'cache-failure.json',checks);raise RuntimeError('Cache mismatch')
        if checks:dump(outdir/'operator-validation.json',{'checks':checks,'pass':all(c['pass'] for c in checks)})
        print(f'Seed {seed}: interventions {ix+1}/{len(rows)}',flush=True)

def main():
    freeze=json.loads((HERE/'freeze.json').read_text())
    for path,sha in freeze['files'].items():assert digest(ROOT/path)==sha,f'Frozen file changed: {path}'
    cfg=json.loads((HERE/'config.json').read_text());rows=[json.loads(x) for x in (HERE/'data.jsonl').read_text().splitlines()]
    engine=Engine(cfg['engine'])
    dump(HERE/'results/model-runtime.json',json.loads((ROOT/'results/readout-replication/model-runtime.json').read_text()))
    baselines={};summaries=[]
    for seed in cfg['sampling_seeds']:
        engine.config['seed']=seed;outdir=HERE/f'results/seed-{seed}';base=Journal(outdir/'baselines.jsonl');baselines[seed]=base
        baseline_tasks(engine,rows[:2],base)
        native_path=outdir/'native-validation.json'
        if not native_path.exists():
            checks=verify_native(engine,rows[0],base);dump(native_path,{'checks':checks,'pass':all(c['pass'] for c in checks)})
        assert json.loads(native_path.read_text())['pass']
        baseline_tasks(engine,rows,base)
        for c in ['donor_full','recipient_full','textual_counterfactual']:
            rr=[r for r in base.rows.values() if r['condition']==c]
            summaries.append({'seed':seed,'condition':c,'n':len(rr),'correct':sum(r['correct'] for r in rr),'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr)})
        dump(HERE/'results/baseline-summary.json',summaries)
    gate=all(sum(r['correct'] for r in summaries if r['condition']==c)/sum(r['n'] for r in summaries if r['condition']==c)>=.8 for c in ['donor_full','recipient_full','textual_counterfactual'])
    dump(HERE/'results/competence-gate.json',{'pass':gate,'summary':summaries})
    if not gate:print('Competence gate failed; intervention branch stopped.',flush=True);return
    for seed in cfg['sampling_seeds']:
        engine.config['seed']=seed
        run_controls(engine,rows,baselines[seed],HERE/f'results/seed-{seed}',seed)
    dump(HERE/'results/execution-complete.json',{'complete':True,'pair_seed_records':240,'baseline_records':720,'planned_intervention_records':2220,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    print('Replication execution complete.',flush=True)

if __name__=='__main__':main()
