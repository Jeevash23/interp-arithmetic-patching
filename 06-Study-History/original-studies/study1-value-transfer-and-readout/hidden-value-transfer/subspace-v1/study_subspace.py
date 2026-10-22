"""One fixed rank-eight value-conditioned projection; no gradient or search."""
import argparse,hashlib,json,os,sys
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;V3=PARENT/'competence-v3'
sys.path.insert(0,str(PARENT/'competence-v2'));sys.path.insert(0,str(PARENT))
import study as core
import numpy as np
import torch
from tasks import task,calc,prompt,predictions,sha,dump
from prompt_formats import EXAMPLES
RESULTS=HERE/'results';LAYER=23;ROLE='prompt_end';RANK=8


def read(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def digest(v):return hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare():
    dev=read(V3/'development-with-source-controls.jsonl')
    seen={tuple(prompt(t)[0:1]) for r in dev for t in [r['donor'],r['recipient'],*r['controls'].values()]}
    data=[]
    for z in range(15,141):
        for y in [47,83,137]:
            start=int(hashlib.sha256(f'20261002/{z}/{y}'.encode()).hexdigest()[:8],16)%8
            for offset in range(8):
                b=2+(start+offset)%8
                pair=[task(z,f,b,'add' if y>=z else 'sub',abs(y-z)) for f in ['add','sub']]
                if all((prompt(t)[0],) not in seen for t in pair):break
            else:raise AssertionError('No disjoint operand choice')
            for t in pair:
                assert t['y']==y and calc(t['first'],t['a'],t['b'])==z and t['a']>0
                data.append(dict(id=f'fit-z{z:03d}-y{y:03d}-{t["first"]}',task=t,label=z,family='box'))
    assert len(data)==756 and len({r['id'] for r in data})==756
    text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in data);p=HERE/'training.jsonl'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    return data


@torch.inference_mode()
def capture(engine,encoded):
    result={}
    def hook(module,args,output):result['vector']=output[0,encoded['positions'][ROLE]].detach().cpu().clone()
    handle=engine.model.model.layers[LAYER].register_forward_hook(hook)
    try:engine.forward(encoded['prompt_ids'])
    finally:handle.remove()
    return result['vector']


def freeze():
    paths=[HERE/'PROTOCOL.md',HERE/'study_subspace.py',HERE/'analyze_subspace.py',HERE/'training.jsonl',
        V3/'freeze-qwen4b-fewshot.json',V3/'development-with-source-controls.jsonl',
        PARENT/'competence-v2/study.py',PARENT/'run.py',PARENT/'tasks.py',PARENT/'prompt_formats.py']
    record=dict(stage='exploratory fitted development method',layer=LAYER,role=ROLE,rank=RANK,strength=1,
        files={str(p.relative_to(PARENT)):sha(p) for p in paths},reserve_evaluated=False)
    target=HERE/'source-freeze.json'
    if target.exists():assert json.loads(target.read_text())==record,'Frozen source changed'
    else:dump(target,record)


def fitting_records(engine,data):
    journal=core.previous.Journal('training-baselines');vp=RESULTS/'training-vectors.pt'
    vectors=torch.load(vp,weights_only=True,map_location='cpu') if vp.exists() else {}
    reuse={tuple(r['prompt_ids']):r for r in journal.rows.values()};dirty=False
    for index,row in enumerate(data):
        encoded=engine.encode(*prompt(row['task']))
        if row['id'] not in vectors:vectors[row['id']]=capture(engine,encoded);dirty=True
        for kind in ['full','first']:
            key=row['id']+'/'+kind
            if key in journal.rows:continue
            enc=encoded if kind=='full' else engine.encode(*prompt(row['task'],kind='first'))
            old=reuse.get(tuple(enc['prompt_ids']))
            output=engine.generate(enc) if old is None else {k:old[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}
            target=row['task']['y' if kind=='full' else 'z']
            saved=journal.put(key,dict(id=row['id'],task=row['task'],kind=kind,target=target,correct=output['answer']==target,
                reused_from=old['key'] if old else None,vector_sha256=digest(vectors[row['id']]) if kind=='full' else None,**enc,**output))
            reuse[tuple(enc['prompt_ids'])]=saved
        if (index+1)%42==0 or index+1==len(data):
            if dirty:torch.save(vectors,vp);dirty=False
            print('Fitting stimuli',index+1,'/',len(data),flush=True)
    assert len(journal.rows)==1512 and len(vectors)==756
    summary={kind:dict(n=len(data),correct=sum(r['correct'] for r in journal.rows.values() if r['kind']==kind),
        invalid=sum(r['answer'] is None for r in journal.rows.values() if r['kind']==kind),
        truncated=sum(r['truncated'] for r in journal.rows.values() if r['kind']==kind),
        unique_prompts=len({tuple(r['prompt_ids']) for r in journal.rows.values() if r['kind']==kind})) for kind in ['full','first']}
    dump(RESULTS/'training-competence.json',summary)
    return vectors


def mean_basis(matrix,labels):
    means=np.stack([matrix[labels==label].mean(0) for label in sorted(set(labels))])
    centered=means-means.mean(0)
    _,s,vh=np.linalg.svd(centered,full_matrices=False)
    basis=vh[:RANK].T.copy()
    # Canonical signs do not affect the projector, but simplify artifact comparisons.
    for col in range(RANK):
        if basis[np.abs(basis[:,col]).argmax(),col]<0:basis[:,col]*=-1
    return basis,s


def fit(data,vectors):
    artifact=RESULTS/'learned-bases.pt'
    if artifact.exists():
        record=json.loads((RESULTS/'learned-freeze.json').read_text())
        assert record['artifact_sha256']==sha(artifact) and record['training_vectors_sha256']==sha(RESULTS/'training-vectors.pt')
        assert record['source_freeze_sha256']==sha(HERE/'source-freeze.json')
        return torch.load(artifact,weights_only=True,map_location='cpu')
    matrix=torch.stack([vectors[r['id']] for r in data]).double().numpy();labels=np.array([r['label'] for r in data])
    u,s=mean_basis(matrix,labels)
    shuffled=np.random.default_rng(20261003).permutation(labels)
    sham,ss=mean_basis(matrix,shuffled)
    random,_=np.linalg.qr(np.random.default_rng(20261004).normal(size=(matrix.shape[1],RANK)))
    bases={k:torch.from_numpy(v.copy()).float() for k,v in [('learned',u),('shuffled',sham),('random',random)]}
    checks={}
    rng=np.random.default_rng(20261005)
    for name,basis in bases.items():
        b=basis.double().numpy();a=rng.normal(size=(16,b.shape[0]));c=rng.normal(size=a.shape)
        pa=(a@b)@b.T;pc=(c@b)@b.T
        checks[name]=dict(orthogonality_max=float(np.max(np.abs(b.T@b-np.eye(RANK)))),
            idempotence_max=float(np.max(np.abs((pa@b)@b.T-pa))),
            symmetry_inner_product_error=float(abs(np.sum(pa*c)-np.sum(a*pc))))
        assert checks[name]['orthogonality_max']<1e-6 and checks[name]['idempotence_max']<1e-6
        assert checks[name]['symmetry_inner_product_error']<1e-6
    q,_=np.linalg.qr(rng.normal(size=(16,16)));rd=rng.normal(size=16);dd=rng.normal(size=16)
    assert np.max(np.abs(rd+(dd-rd)@q@q.T-dd))<1e-12
    torch.save(bases,artifact)
    dump(RESULTS/'learned-freeze.json',dict(artifact_sha256=sha(artifact),training_vectors_sha256=sha(RESULTS/'training-vectors.pt'),
        training_data_sha256=sha(HERE/'training.jsonl'),source_freeze_sha256=sha(HERE/'source-freeze.json'),
        objective='top eight right singular vectors of centered value-class means',rank=RANK,layer=LAYER,role=ROLE,strength=1,
        singular_values=s.tolist(),explained_between_value_variance=float((s[:RANK]**2).sum()/(s**2).sum()),
        shuffled_singular_values=ss.tolist(),shuffled_labels=shuffled.tolist(),shuffled_label_seed=20261003,random_basis_seed=20261004,
        algebra_checks=checks,full_rank_synthetic_replacement=True,model_weights_trained=False,validation_used=False))
    print('Fitted and froze rank-eight bases.',flush=True)
    return bases


def displacement(source,recipient,basis):
    difference=source.float()-recipient.float()
    return basis@(basis.T@difference)


def control_records(engine,rows,bases):
    original=V3/'results/qwen4b-fewshot';eligible=V3/'results/eligible-sites'
    baseline={r['key']:r for r in read(original/'baselines.jsonl')}
    digit={r['id']:r for r in read(original/'actual-donor-digit-control.jsonl')}
    whole={r['id']:r for r in read(eligible/'controls.jsonl') if r['layer']==LAYER and r['role']==ROLE and r['condition']=='donor'}
    oracle={r['id']:r for r in read(original/'controls.jsonl') if r['condition']=='text_first_answer_digit'}
    journal=core.previous.Journal('controls');vp=RESULTS/'control-vectors.pt'
    saved=torch.load(vp,weights_only=True,map_location='cpu') if vp.exists() else {}
    cache={};groups={r['group']:r['donor'] for r in rows}
    def encoded_vector(t):
        enc=engine.encode(*prompt(t));key=tuple(enc['prompt_ids'])
        if key not in cache:cache[key]=capture(engine,enc)
        return enc,cache[key]
    first=rows[0];renc,rv=encoded_vector(first['recipient']);_,dv=encoded_vector(first['donor'])
    zero=displacement(rv,rv,bases['learned']);assert torch.equal(zero,torch.zeros_like(zero))
    selfgen=engine.generate(renc,(LAYER,renc['positions'][ROLE],(rv.float()+zero).to(rv),'replace'))
    assert selfgen['output_ids']==baseline[first['id']+'/recipient_full']['output_ids']
    target=(rv.float()+displacement(dv,rv,bases['learned'])).to(rv);stats={}
    normal=engine.forward(renc['prompt_ids']);patched=engine.forward(renc['prompt_ids'],spec=(LAYER,renc['positions'][ROLE],target,'replace'),stats=stats)
    stats['max_logit_change']=float((normal-patched).abs().max())
    assert stats['delta_norm']>0 and stats['max_logit_change']>0 and stats['write_error']==0 and stats['other_positions_max_change']==0
    dump(RESULTS/'instrumentation.json',dict(pass_check=True,self_generation_equal=True,zero_projection=True,positive_write=stats))
    names=['donor','unpatched','same_example','isotropic_delta','random_subspace','label_shuffled','random_source',
        'same_z_decomposition','same_z_different_answer','different_z_same_answer','same_leading_different_value',
        'text_first_answer_digit','text_actual_first_token','whole_vector']
    for index,row in enumerate(rows):
        encoded,rv=encoded_vector(row['recipient'])
        sources=dict(donor=row['donor'],random_source=groups[(row['group']+1)%len(groups)],**row['controls'])
        captures={name:encoded_vector(t)[1] for name,t in sources.items()}
        learned_delta=displacement(captures['donor'],rv,bases['learned']);norm=learned_delta.norm()
        for condition in names:
            key=row['id']+'/'+condition
            old_record=journal.rows.get(key)
            if old_record is not None and (old_record['source_hash'] is None or key in saved):continue
            target=None;reused=None;actual_source=None;delta=None
            if condition in sources:
                delta=displacement(captures[condition],rv,bases['learned']);actual_source=sources[condition]
            elif condition=='same_example':delta=displacement(rv,rv,bases['learned']);actual_source=row['recipient']
            elif condition=='isotropic_delta':
                generator=torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
                noise=torch.randn(rv.shape,generator=generator);delta=noise/noise.norm()*norm
            elif condition=='random_subspace':
                random_delta=displacement(captures['donor'],rv,bases['random'])
                assert random_delta.norm()>0
                delta=random_delta/random_delta.norm()*norm;actual_source=row['donor']
            elif condition=='label_shuffled':delta=displacement(captures['donor'],rv,bases['shuffled']);actual_source=row['donor']
            elif condition=='unpatched':reused=baseline[row['id']+'/recipient_full']
            elif condition=='text_first_answer_digit':reused=oracle[row['id']]
            elif condition=='text_actual_first_token':reused=digit[row['id']]
            elif condition=='whole_vector':reused=whole[row['id']];actual_source=row['donor']
            if delta is not None:
                target=(rv.float()+delta).to(rv);saved[key]=target
                if old_record is not None:
                    assert digest(target)==old_record['source_hash'];continue
                output=engine.generate(encoded,(LAYER,encoded['positions'][ROLE],target,'replace'))
                assert output['write']['write_error']==0 and output['write']['other_positions_max_change']==0
            else:
                assert reused is not None
                output={k:reused[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}
            if condition in ['unpatched','same_example']:assert output['output_ids']==baseline[row['id']+'/recipient_full']['output_ids']
            comparison=actual_source or row['donor'];pred=predictions(comparison,row['recipient']);r=row['recipient'];g=lambda z:calc(r['remaining'],z,r['k'])
            alternatives=dict(pred,first_operand_only=g(calc(r['first'],comparison['a'],r['b'])),
                second_operand_only=g(calc(r['first'],r['a'],comparison['b'])),
                first_operator_only=g(calc(comparison['first'],r['a'],r['b'])),
                downstream_operator_only=calc(comparison['remaining'],r['z'],r['k']),
                downstream_constant_only=calc(r['remaining'],r['z'],comparison['k']),
                downstream_program=calc(comparison['remaining'],r['z'],comparison['k']))
            journal.put(key,dict(id=row['id'],group=row['group'],condition=condition,layer=LAYER,role=ROLE,rank=RANK,
                recipient_operation=r['remaining'],recipient_task=r,original_donor=row['donor'],actual_activation_source=actual_source,
                comparison_reference=pred,alternative_predictions=alternatives,alternative_matches=[k for k,v in alternatives.items() if output['answer']==v],
                counterfactual_success=output['answer']==row['predictions']['counterfactual'],
                actual_source_success=output['answer']==pred['counterfactual'] if actual_source is not None else None,
                source_hash=digest(target) if target is not None else None,reused_from_key=reused['key'] if reused else None,
                reused_from_study=('competence-v3/results/eligible-sites' if condition=='whole_vector' else 'competence-v3/results/qwen4b-fewshot') if reused else None,
                intended_float32_displacement_norm=float(delta.norm()) if delta is not None else None,
                original_learned_displacement_norm=float(norm),**encoded,**output))
        torch.save(saved,vp);print('Subspace controlled pairs',index+1,'/',len(rows),flush=True)
    assert len(journal.rows)==504 and len(saved)==360
    for key,v in saved.items():assert digest(v)==journal.rows[key]['source_hash']
    dump(RESULTS/'completed.json',dict(completed=True,records=504,fresh_generations=360,reused_records=144,
        vector_hashes=len(saved),self_generation_equal=36,source_freeze_sha256=sha(HERE/'source-freeze.json'),
        learned_artifact_sha256=sha(RESULTS/'learned-bases.pt'),reserve_evaluated=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    data=prepare()
    if a.prepare_only:print('Prepared',len(data),'development fitting tasks.');return
    eligible=V3/'results/eligible-sites';assert (eligible/'completed.json').exists()
    assert not json.loads((eligible/'analysis/summary.json').read_text())['promising_sites'],'Prefer passing simpler intervention'
    assert json.loads((V3/'results/qwen4b-fewshot/competence.json').read_text())['pass_check']
    freeze();core.previous.RESULTS=RESULTS
    engine=core.Engine('qwen4b','subspace');engine.format_name='fewshot'
    rp=RESULTS/'runtime-subspace.json';runtime=json.loads(rp.read_text())
    runtime.update(calibration_format='fewshot',examples=EXAMPLES,source_freeze_sha256=sha(HERE/'source-freeze.json'))
    dump(rp,runtime)
    vectors=fitting_records(engine,data);bases=fit(data,vectors)
    control_records(engine,read(V3/'development-with-source-controls.jsonl'),bases)


if __name__=='__main__':main()
