"""Continue competence, then fixed-node localization and source controls."""
import argparse,contextlib,hashlib,importlib.metadata,json,os,platform,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(HERE/'.packages'))
sys.path.insert(0,str(PARENT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_CACHE']=str(HERE/'.cache/huggingface')
os.environ['TRITON_CACHE_DIR']=str(HERE/'.cache/triton')
os.environ['TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR']='1'
import run as previous
from tasks import ROLES,SEED,prompt,predictions,parse,dump,sha
from prompt_formats import FormatEngine,SYSTEM
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

KERNEL_REV='8c178950e8710e26a2210e4e909cd02dc16c8715'


def frozen(candidate):
    paths=[HERE/'PROTOCOL.md',HERE/'study.py',HERE/'analyze.py',HERE/'prepare_sources.py',HERE/'development-with-source-controls.jsonl',HERE/'dependency-manifest.json',PARENT/'run.py',PARENT/'tasks.py',
        PARENT/'prompt_formats.py',PARENT/'data/development.jsonl',PARENT/'data/reserve.jsonl',
        HERE/f'model-manifest-{candidate}.json']
    if candidate=='qwen4b':paths += [HERE/'kernel-inspection.json',HERE/'AMENDMENT-DENSE-LOADER.md']
    record=dict(files={str(p.relative_to(PARENT)):sha(p) for p in paths},candidate=candidate,
                client_date='2026-10-01',stage='development')
    target=HERE/f'freeze-{candidate}.json'
    if target.exists():assert json.loads(target.read_text())==record,'Frozen source changed'
    else:dump(target,record)


class Engine(FormatEngine):
    def __init__(self,candidate,phase):
        self.format_name='system'
        torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        before=torch.cuda.mem_get_info()
        if candidate=='qwen4b':
            import kernels
            from transformers.integrations import finegrained_fp8
            # Dense models have no experts implementation. The installed loader
            # dereferences this missing metadata entry even without tensor parallelism.
            finegrained_fp8.FP8Experts._impl_tp_layer_overrides.setdefault(None,{})
            kernel=kernels.get_kernel('kernels-community/finegrained-fp8',revision=KERNEL_REV)
            # Bind the same documented entry points as HF's lazy loader, at an immutable revision.
            finegrained_fp8._FINEGRAINED_FP8=finegrained_fp8.FineGrainedFP8(
                matmul=kernel.matmul_2d,batched_matmul=kernel.matmul_batched,grouped_matmul=kernel.matmul_grouped)
        model=HERE/'models'/candidate
        self.tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        self.model.requires_grad_(False)
        eos=self.model.generation_config.eos_token_id
        self.eos=eos if isinstance(eos,list) else [eos]
        self.genconfig=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=16,use_cache=False,
            eos_token_id=self.eos,pad_token_id=self.tok.pad_token_id,bos_token_id=self.tok.bos_token_id,
            repetition_penalty=1.)
        manifest=json.loads((HERE/f'model-manifest-{candidate}.json').read_text())
        runtime=dict(candidate=candidate,phase=phase,repository=manifest['repository'],revision=manifest['revision'],
            packages={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','numpy']},
            python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),cuda_free_total_before=before,
            cuda_free_total_loaded=torch.cuda.mem_get_info(),generation=self.genconfig.to_dict(),
            model_config=self.model.config.to_dict(),chat_template=self.tok.chat_template,system_prompt=SYSTEM,
            parameters=sum(p.numel() for p in self.model.parameters()),
            dtype_counts={str(d):sum(p.numel() for p in self.model.parameters() if p.dtype==d) for d in {p.dtype for p in self.model.parameters()}},
            kernel_revision=KERNEL_REV if candidate=='qwen4b' else None,score_method='sequential conditional log probabilities',
            dense_loader_compatibility='empty expert parallel-plan override for None' if candidate=='qwen4b' else None)
        dump(previous.RESULTS/f'runtime-{phase}.json',runtime)
        print('Loaded',manifest['repository'],'free GPU',runtime['cuda_free_total_loaded'],flush=True)

    def score_candidates(self,encoded,answers,spec=None):
        cache={};result={}
        for answer in answers:
            target=self.tok.encode(str(answer),add_special_tokens=False)+[self.eos[0]]
            prefix=[];values=[]
            for token in target:
                key=tuple(prefix)
                if key not in cache:
                    cache[key]=self.forward(encoded['prompt_ids']+prefix,spec=spec)[-1].log_softmax(-1)
                values.append(float(cache[key][token]));prefix.append(token)
            result[answer]=dict(target=answer,target_ids=target,token_log_probabilities=values,
                log_probability=sum(values),digits_log_probability=sum(values[:-1]))
        return result


def instrumentation(engine,rows):
    cases=[];n=engine.model.config.num_hidden_layers
    for row in rows[:4]:
        encoded=engine.encode(*prompt(row['recipient'],row['family']));ids=encoded['prompt_ids']
        baseline=engine.forward(ids);vectors=engine.capture(encoded);actual=engine.generate(encoded)
        with torch.inference_mode():
            native=engine.model.generate(input_ids=engine.tensor(ids),generation_config=engine.genconfig)[0,len(ids):].tolist()
        checks=[]
        for layer in [0,n//2,n-1]:
            for ri,role in enumerate(ROLES):
                for rule in ['read','identity','replace']:
                    spec=(layer,encoded['positions'][role],vectors[layer,ri],rule)
                    delta=float((engine.forward(ids,spec=spec)-baseline).abs().max())
                    checks.append(dict(layer=layer,role=role,rule=rule,max_logit_abs=delta,pass_check=delta<=1e-5))
        spec=(n//2,encoded['positions']['prompt_end'],vectors[n//2,3],'replace')
        own=engine.generate(encoded,spec)
        answers=[row['predictions']['counterfactual'],row['recipient']['y']]
        sequential=engine.score_candidates(encoded,answers)
        identity=engine.score_candidates(encoded,answers,(n//2,encoded['positions']['prompt_end'],None,'identity'))
        score_null=max(abs(sequential[a]['log_probability']-identity[a]['log_probability']) for a in answers)
        cross_shape={str(a):dict(sequential=sequential[a],single_pass=engine.score(encoded,a)) for a in answers}
        case=dict(id=row['id'],native_equal=native==actual['output_ids'],native_ids=native,
            custom_ids=actual['output_ids'],captured_self_generation_equal=own['output_ids']==actual['output_ids'],
            null_checks=checks,score_null_max=score_null,cross_shape_score_diagnostic=cross_shape)
        case['pass_check']=case['native_equal'] and case['captured_self_generation_equal'] and all(c['pass_check'] for c in checks) and score_null<=1e-5
        cases.append(case);print('Instrumentation',row['id'],case['pass_check'],flush=True)
    row=rows[0];recipient=engine.encode(*prompt(row['recipient']));donor=engine.encode(*prompt(row['donor']))
    source=engine.capture(donor);stats={}
    normal=engine.forward(recipient['prompt_ids'])
    effect=engine.forward(recipient['prompt_ids'],spec=(n//2,recipient['positions']['prompt_end'],source[n//2,3],'replace'),stats=stats)
    stats['max_logit_change']=float((effect-normal).abs().max())
    passed=all(c['pass_check'] for c in cases) and stats['delta_norm']>0 and stats['max_logit_change']>0 and stats['write_error']==0 and stats['other_positions_max_change']==0
    dump(previous.RESULTS/'instrumentation.json',dict(pass_check=passed,cases=cases,positive_write=stats))
    return passed


def screen(engine,rows):
    journal=previous.Journal('screen');layers=engine.model.config.num_hidden_layers
    start=time.monotonic()
    for index,row in enumerate(rows):
        r=engine.encode(*prompt(row['recipient'],row['family']));d=engine.encode(*prompt(row['donor'],row['family']))
        source=engine.capture(d);cf=row['predictions']['counterfactual'];normal=row['recipient']['y']
        baseline=engine.score_candidates(r,[cf,normal]);before=baseline[cf]['log_probability']-baseline[normal]['log_probability']
        for layer in range(layers):
            for ri,role in enumerate(ROLES):
                key=f'{row["id"]}/{layer}/{role}'
                if key in journal.rows:continue
                spec=(layer,r['positions'][role],source[layer,ri],'replace')
                scores=engine.score_candidates(r,[cf,normal],spec)
                journal.put(key,dict(id=row['id'],group=row['group'],layer=layer,role=role,
                    recipient_operation=row['recipient']['remaining'],donor_position=d['positions'][role],
                    recipient_position=r['positions'][role],counterfactual=scores[cf],recipient=scores[normal],
                    unpatched_counterfactual=baseline[cf],unpatched_recipient=baseline[normal],
                    delta_log_ratio=scores[cf]['log_probability']-scores[normal]['log_probability']-before))
        print('Screen pairs',index+1,'/',len(rows),'records',len(journal.rows),'seconds',round(time.monotonic()-start),flush=True)
    sites=[]
    for layer in range(layers):
        for role in ROLES:
            records=[r for r in journal.rows.values() if r['layer']==layer and r['role']==role]
            groups={g:[r['delta_log_ratio'] for r in records if r['group']==g] for g in sorted({r['group'] for r in records})}
            mean=sum(sum(v)/len(v) for v in groups.values())/len(groups)
            sites.append(dict(layer=layer,role=role,mean_delta_log_ratio=mean,n=len(records)))
    selected=sorted(sites,key=lambda s:(-s['mean_delta_log_ratio'],s['layer'],ROLES.index(s['role'])))[:3]
    dump(previous.RESULTS/'screen-summary.json',dict(sites=sites,selected=selected,selection='exploratory development only'))
    print('Selected sites',selected,flush=True)


def source_baselines(engine,rows):
    journal=previous.Journal('source-baselines')
    reuse={tuple(r['prompt_ids']):r for r in journal.rows.values()}
    for row in rows:
        for name,source in row['controls'].items():
            for condition in ['full','first','target_remaining']:
                key=f'{row["id"]}/{name}/{condition}'
                if key in journal.rows:continue
                if condition=='target_remaining':
                    text,spans=prompt(row['recipient'],row['family'],'counterfactual',source['z'])
                    target=predictions(source,row['recipient'])['counterfactual']
                else:
                    text,spans=prompt(source,row['family'],condition)
                    target=source['y' if condition=='full' else 'z']
                encoded=engine.encode(text,spans);old=reuse.get(tuple(encoded['prompt_ids']))
                out=engine.generate(encoded) if old is None else {k:old[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}
                saved=journal.put(key,dict(id=row['id'],group=row['group'],source_condition=name,
                    condition=condition,target=target,correct=out['answer']==target,
                    reused_from=old['key'] if old else None,**encoded,**out))
                reuse[tuple(encoded['prompt_ids'])]=saved
        print('Source baseline pairs',len(journal.rows)//12,'/',len(rows),flush=True)


def controls(engine,rows):
    sites=json.loads((previous.RESULTS/'screen-summary.json').read_text())['selected']
    journal=previous.Journal('controls');base=previous.Journal('baselines')
    vp=previous.RESULTS/'control-vectors.pt';saved=torch.load(vp,weights_only=True) if vp.exists() else {}
    group_donor={r['group']:r['donor'] for r in rows}
    for i,row in enumerate(rows):
        encoded=engine.encode(*prompt(row['recipient'],row['family']));rv=engine.capture(encoded)
        tasks=dict(donor=row['donor'],random_source=group_donor[(row['group']+1)%len(group_donor)],**row['controls'])
        captures={c:engine.capture(engine.encode(*prompt(t,row['family']))) for c,t in tasks.items()}
        names=['donor','unpatched','same_example','isotropic_delta','random_source',*row['controls'],'text_first_answer_digit']
        for site in sites:
            layer,role=site['layer'],site['role'];ri=ROLES.index(role)
            for condition in names:
                key=f'{row["id"]}/{layer}/{role}/{condition}'
                if key in journal.rows:continue
                vector=spec=forced=None;source_task=tasks.get(condition,row['donor'])
                if condition in captures:
                    vector=captures[condition][layer,ri];spec=(layer,encoded['positions'][role],vector,'replace')
                elif condition=='same_example':
                    vector=rv[layer,ri];spec=(layer,encoded['positions'][role],vector,'replace')
                elif condition=='isotropic_delta':
                    generator=torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
                    noise=torch.randn(rv[layer,ri].shape,generator=generator)
                    vector=noise/noise.norm()*(captures['donor'][layer,ri]-rv[layer,ri]).float().norm()
                    spec=(layer,encoded['positions'][role],vector,'add')
                elif condition=='text_first_answer_digit':forced=engine.tok.encode(str(row['donor']['y'])[0],add_special_tokens=False)
                output=engine.generate(encoded,spec,forced)
                if condition in ['unpatched','same_example']:
                    assert output['output_ids']==base.rows[row['id']+'/recipient_full']['output_ids'],'Identity generation failed'
                actual=predictions(source_task,row['recipient']);digest=None
                if vector is not None:
                    saved[key]=vector;digest=hashlib.sha256(vector.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                journal.put(key,dict(id=row['id'],group=row['group'],layer=layer,role=role,condition=condition,
                    recipient_operation=row['recipient']['remaining'],source_task=source_task,recipient_task=row['recipient'],
                    predictions=row['predictions'],actual_source_predictions=actual,digit_alternatives=row['digit_alternatives'],
                    source_hash=digest,counterfactual_success=output['answer']==row['predictions']['counterfactual'],
                    actual_source_success=output['answer']==actual['counterfactual'],
                    prediction_matches=[k for k,v in row['predictions'].items() if output['answer']==v],**encoded,**output))
        torch.save(saved,vp);print('Controlled pairs',i+1,'/',len(rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['qwen4b','qwen25'],default='qwen4b');ap.add_argument('--phase',choices=['baseline','screen','controls'],required=True);args=ap.parse_args()
    frozen(args.candidate)
    previous.RESULTS=HERE/'results'/args.candidate
    rows=[json.loads(s) for s in (HERE/'development-with-source-controls.jsonl').read_text().splitlines()]
    if args.phase!='baseline':
        assert json.loads((previous.RESULTS/'instrumentation.json').read_text())['pass_check']
        assert json.loads((previous.RESULTS/'competence.json').read_text())['pass_check']
    engine=Engine(args.candidate,args.phase)
    if args.phase=='baseline':
        assert instrumentation(engine,rows),'Instrumentation failed'
        previous.baselines(engine,rows)
    elif args.phase=='screen':screen(engine,rows)
    else:
        source_baselines(engine,rows)
        controls(engine,rows)
    dump(previous.RESULTS/f'completed-{args.phase}.json',dict(completed=True,phase=args.phase,
        candidate=args.candidate,reserve_evaluated=False,max_cuda_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
