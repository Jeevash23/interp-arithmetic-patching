"""Serial second-checkpoint feasibility and bounded replication with strict gates."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent;ROOT=PARENT.parent
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(PARENT))
from prepare import prompt
from run import Journal,dump,classify,torch,vector_hash
from value_transfer.engine import Engine
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,LogitsProcessorList,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
from clamp_mediation import value_line_end

class LocalEngine(Engine):
    def __init__(self):
        self.config=json.loads((PARENT/'config.json').read_text())['engine']
        self.config.update(checkpoint='Qwen/Qwen3-1.7B',dtype='bfloat16',seed=20262909,run_label='second-checkpoint')
        torch.set_num_threads(4);torch.manual_seed(self.config['seed']);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.device='cuda';self.dtype=torch.bfloat16
        self.tokenizer=AutoTokenizer.from_pretrained(HERE/'model',local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(HERE/'model',local_files_only=True,dtype=self.dtype,attn_implementation='eager').to(self.device).eval()
        self.model.requires_grad_(False)
        self.eos=self.model.generation_config.eos_token_id
        if isinstance(self.eos,int):self.eos=[self.eos]
        self.generation_config=GenerationConfig(do_sample=True,num_beams=1,max_new_tokens=512,use_cache=True,temperature=.6,top_p=.95,top_k=20,repetition_penalty=1.,eos_token_id=self.eos,pad_token_id=self.tokenizer.pad_token_id,bos_token_id=self.tokenizer.bos_token_id)
        self.processors=LogitsProcessorList([TemperatureLogitsWarper(.6),TopKLogitsWarper(20),TopPLogitsWarper(.95)])
        dump(HERE/'results/runtime.json',{'config':self.config,'model_config':self.model.config.to_dict(),'native_generation_config':self.model.generation_config.to_dict(),'chat_template':self.tokenizer.chat_template,'parameters':sum(p.numel() for p in self.model.parameters()),'cuda_free_total':torch.cuda.mem_get_info(),'precision_scope':'bfloat16, not a controlled comparison against float32'})

def baselines(engine,rows,j):
    saved={tuple(r['prompt_ids']):r for r in j.rows.values()}
    for row in rows:
        for c,side in [('donor_full','donor'),('recipient_full','recipient'),('textual_counterfactual','recipient')]:
            key=f"{row['id']}/{c}"
            if key in j.rows:continue
            z=row['donor']['z'] if c=='textual_counterfactual' else None
            text=prompt(row[side],row['family'],'remaining' if z is not None else 'full',z);ids=engine.encode(text)
            target=row['predictions']['counterfactual'] if z is not None else row[side]['y']
            reused=saved.get(tuple(ids));out=engine.generate(ids) if reused is None else {k:v for k,v in reused.items() if k not in ['key','id','condition','task','prompt','target','correct']}
            r=j.put(key,{**out,'id':row['id'],'condition':c,'task':row[side],'prompt':text,'target':target,'correct':out['parsed']['answer']==target,'reused_exact_prompt_from':None if reused is None else reused['key']})
            saved[tuple(ids)]=r
        print(f"Checkpoint baselines: {len(j.rows)}/180",flush=True)

def instrument(engine,rows,base):
    checks=[]
    for row in rows:
        d=base.rows[f"{row['id']}/donor_full"];r=base.rows[f"{row['id']}/recipient_full"]
        dc,rc=engine.cuts(d),engine.cuts(r)
        native_d=engine.native_generate(d['prompt_ids']);native_r=engine.native_generate(r['prompt_ids'])
        test={'id':row['id'],'native_donor':native_d==d['output_ids'],'native_recipient':native_r==r['output_ids'],'native_donor_ids':native_d,'native_recipient_ids':native_r,'timings_available':'pre_z' in dc and 'pre_z' in rc}
        if test['timings_available']:
            src=engine.capture(dc['pre_z'],[26])[26];selfsrc=engine.capture(rc['pre_z'],[26])[26]
            a,la=engine.generate(r['prompt_ids'],rc['pre_z'],26,src,trace=True)
            b,lb=engine.generate(r['prompt_ids'],rc['pre_z'],26,src,use_cache=False,trace=True)
            identical=a['output_ids']==b['output_ids'];delta=max(float((x.float()-y.float()).abs().max()) for x,y in zip(la,lb))
            test.update(replay_identity=engine.generate(r['prompt_ids'],rc['pre_z'])['output_ids']==r['output_ids'],same_example_identity=engine.generate(r['prompt_ids'],rc['pre_z'],26,selfsrc)['output_ids']==r['output_ids'],cached_recomputed_exact=identical,max_cache_logit_abs=delta,cache_logit_tolerance=.125,cache_logits_pass=delta<=.125,write_pass=a['write']['delta_norm']>0 and a['write']['write_error']==0 and a['write']['other_positions_max_change']==0,cached=a,recomputed=b)
        test['pass']=all(test.get(k,False) for k in ['native_donor','native_recipient','timings_available','replay_identity','same_example_identity','cached_recomputed_exact','cache_logits_pass','write_pass'])
        checks.append(test);dump(HERE/'results/instrumentation.json',{'pass':all(t['pass'] for t in checks),'planned_cases':4,'completed_cases':len(checks),'checks':checks})
        print(f"Checkpoint instrumentation {row['id']}: {test['pass']}",flush=True)
    return all(t['pass'] for t in checks)

def controls(engine,rows,base):
    j=Journal(HERE/'results/interventions.jsonl');vp=HERE/'results/vectors.pt';vectors=torch.load(vp,weights_only=True) if vp.exists() else {}
    names=['activation','unpatched','same_example','isotropic_delta','text_first_digit','text_full_value','text_recipient_clamp','activation_recipient_clamp']
    for row in rows:
        d=base.rows[f"{row['id']}/donor_full"];r=base.rows[f"{row['id']}/recipient_full"];dc,rc=engine.cuts(d),engine.cuts(r)
        src=engine.capture(dc['pre_z'],[26])[26] if 'pre_z' in dc else None
        rv=engine.capture(rc['pre_z'],[26])[26] if 'pre_z' in rc else None
        for c in names:
            key=f"{row['id']}/{c}"
            if key in j.rows:continue
            meta={k:row[k] for k in ['id','group','family','width','leading_digit','predictions']};meta.update(condition=c,donor_z=row['donor']['z'])
            active=c in ['activation','isotropic_delta','activation_recipient_clamp']
            available='pre_z' in rc and (not active or src is not None)
            if not available:
                j.put(key,{**meta,'status':'unavailable','parsed':{'answer':None,'intermediate':None,'parser_failure':True},'success':False,'category':'invalid','prediction_matches':[],'truncated':False});continue
            prefix=rc['pre_z'];position=len(prefix)-1;layer=26 if active or c=='same_example' else None;source=rv if c=='same_example' else src;rule='replace'
            if c=='isotropic_delta':
                noise=torch.randn(src.shape,generator=torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8],16)));source=noise/noise.norm()*(src-rv).float().norm();rule='add'
            if c=='text_first_digit':prefix=prefix+engine.tokenizer.encode(str(row['leading_digit']),add_special_tokens=False)
            if c in ['text_full_value','text_recipient_clamp','activation_recipient_clamp']:
                try:delimiter=value_line_end(engine,r,rc['post_z'])
                except (KeyError,ValueError):
                    j.put(key,{**meta,'status':'unavailable','parsed':{'answer':None,'intermediate':None,'parser_failure':True},'success':False,'category':'invalid','prediction_matches':[],'truncated':False});continue
                prefix=(rc['pre_z']+engine.tokenizer.encode(str(row['donor']['z']),add_special_tokens=False) if c=='text_full_value' else rc['post_z'])+delimiter
            if len(prefix)-len(r['prompt_ids'])>=512:
                j.put(key,{**meta,'status':'unavailable','reason':'no_fresh_generation_budget','parsed':{'answer':None,'intermediate':None,'parser_failure':True},'success':False,'category':'invalid','prediction_matches':[],'truncated':False});continue
            if layer is not None:vectors[key]=source;meta.update(source_hash=vector_hash(source),donor_prefix_ids=rc['pre_z'] if c=='same_example' else dc['pre_z'])
            torch.save(vectors,vp)
            out=engine.generate(r['prompt_ids'],prefix,layer,source,rule,position=position)
            if c in ['unpatched','same_example','text_recipient_clamp'] and out['output_ids']!=r['output_ids']:
                dump(HERE/'results/identity-failure.json',{'id':row['id'],'condition':c,'out':out,'baseline':r});raise RuntimeError('Checkpoint replay identity failed; no comparable mechanism claim')
            j.put(key,{**meta,**out,**classify(out['parsed'],row['predictions']),'status':'generated'})
        print(f"Checkpoint interventions: {len(j.rows)}/480",flush=True)

def main():
    if (HERE/'results/complete.json').exists():print('Checkpoint branch already complete.');return
    frozen=json.loads((HERE/'freeze.json').read_text())
    for rel,sha in frozen['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    engine=LocalEngine();rows=[json.loads(x) for x in (PARENT/'data.jsonl').read_text().splitlines() if json.loads(x)['group']<15]
    base=Journal(HERE/'results/baselines.jsonl');ids=['readout-00-box-w1','readout-00-register-w1','readout-01-box-w2','readout-01-register-w2'];smoke=[next(r for r in rows if r['id']==i) for i in ids]
    baselines(engine,smoke,base)
    if not instrument(engine,smoke,base):
        dump(HERE/'results/complete.json',{'branch_complete':True,'mechanism_evaluated':False,'reason':'prespecified instrumentation gate failed'});return
    baselines(engine,rows,base)
    summary={c:{'n':sum(r['condition']==c for r in base.rows.values()),'correct':sum(r['condition']==c and r['correct'] for r in base.rows.values())} for c in ['donor_full','recipient_full','textual_counterfactual']}
    gate=all(v['correct']/v['n']>=.8 for v in summary.values());dump(HERE/'results/competence.json',{'pass':gate,'conditions':summary})
    if not gate:dump(HERE/'results/complete.json',{'branch_complete':True,'mechanism_evaluated':False,'reason':'competence gate failed'});return
    controls(engine,rows,base)
    dump(HERE/'results/complete.json',{'branch_complete':True,'mechanism_evaluated':True,'pairs':60,'groups':15,'gpu_peak_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__':main()
