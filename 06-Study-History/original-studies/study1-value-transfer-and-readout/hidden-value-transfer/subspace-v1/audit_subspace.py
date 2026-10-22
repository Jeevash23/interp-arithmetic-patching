"""Reconstruct intervention targets from independently recaptured source vectors."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import study_subspace as s
import torch


def main():
    root=s.RESULTS;raw=s.read(root/'controls.jsonl');data=s.read(s.V3/'development-with-source-controls.jsonl')
    assert len(raw)==504 and len({r['key'] for r in raw})==504
    frozen=json.loads((HERE/'source-freeze.json').read_text())
    for name,digest in frozen['files'].items():assert s.sha(s.PARENT/name)==digest
    learned=json.loads((root/'learned-freeze.json').read_text())
    assert learned['artifact_sha256']==s.sha(root/'learned-bases.pt')
    bases=torch.load(root/'learned-bases.pt',weights_only=True,map_location='cpu')
    vectors=torch.load(root/'control-vectors.pt',weights_only=True,map_location='cpu')
    fit_vectors=torch.load(root/'training-vectors.pt',weights_only=True,map_location='cpu')
    training=s.read(HERE/'training.jsonl');trainbase=s.read(root/'training-baselines.jsonl')
    full={r['id']:r for r in trainbase if r['kind']=='full'}
    for r in training:
        assert s.digest(fit_vectors[r['id']])==full[r['id']]['vector_sha256']
        t=r['task'];assert s.calc(t['first'],t['a'],t['b'])==r['label'] and s.calc(t['remaining'],r['label'],t['k'])==t['y']
    original=s.V3/'results/qwen4b-fewshot'
    base={r['key']:r for r in s.read(original/'baselines.jsonl')}
    whole={r['key']:r for r in s.read(s.V3/'results/eligible-sites/controls.jsonl')}
    digits={r['key']:r for r in s.read(original/'actual-donor-digit-control.jsonl')}
    oracle={r['key']:r for r in s.read(original/'controls.jsonl')}
    s.core.previous.RESULTS=root
    engine=s.core.Engine('qwen4b','source-recapture-audit');engine.format_name='fewshot'
    captures={};ledger={}
    def get(t):
        encoded=engine.encode(*s.prompt(t));key=hashlib.sha256(json.dumps(encoded['prompt_ids']).encode()).hexdigest()
        if key not in captures:
            captures[key]=s.capture(engine,encoded)
            ledger[key]=dict(task=t,**encoded,vector_sha256=s.digest(captures[key]))
        return captures[key]
    checked=reused=self_count=0;rounding=[];norm_checks=[]
    for r in raw:
        if r['source_hash'] is None:
            lookup=whole if r['condition']=='whole_vector' else digits if r['condition']=='text_actual_first_token' else oracle if r['condition']=='text_first_answer_digit' else base
            src=lookup[r['reused_from_key']]
            for field in ['output_ids','answer','text','truncated','forced_output_ids']:assert r[field]==src[field]
            reused+=1;continue
        rv=get(r['recipient_task']);dv=get(r['original_donor']);normal=s.displacement(dv,rv,bases['learned']);cond=r['condition']
        if cond=='isotropic_delta':
            generator=torch.Generator().manual_seed(int(hashlib.sha256(r['key'].encode()).hexdigest()[:8],16))
            noise=torch.randn(rv.shape,generator=generator);delta=noise/noise.norm()*normal.norm()
        elif cond=='random_subspace':
            d=s.displacement(dv,rv,bases['random']);delta=d/d.norm()*normal.norm()
        elif cond=='label_shuffled':delta=s.displacement(dv,rv,bases['shuffled'])
        else:delta=s.displacement(get(r['actual_activation_source']),rv,bases['learned'])
        target=(rv.float()+delta).to(rv)
        assert torch.equal(target,vectors[r['key']]) and s.digest(target)==r['source_hash']
        assert r['write']['write_error']==0 and r['write']['other_positions_max_change']==0
        assert abs(float(delta.norm())-r['intended_float32_displacement_norm'])<1e-5
        cpu_norm=float((target-rv).float().norm())
        gpu_norm=float((target.to('cuda')-rv.to('cuda')).float().norm())
        norm_checks.append(dict(key=r['key'],cpu_norm=cpu_norm,gpu_norm=gpu_norm,recorded_gpu_norm=r['write']['delta_norm']))
        assert abs(gpu_norm-r['write']['delta_norm'])<1e-5,(r['key'],cpu_norm,gpu_norm,r['write']['delta_norm'])
        if cond=='same_example':
            assert torch.equal(target,rv) and r['output_ids']==base[r['id']+'/recipient_full']['output_ids'];self_count+=1
        if cond in ['isotropic_delta','random_subspace']:
            assert abs(float(delta.norm()-normal.norm()))<1e-4
            rounding.append(dict(key=r['key'],float32_intended_norm=float(delta.norm()),bfloat16_actual_norm=r['write']['delta_norm']))
        checked+=1
    assert checked==360 and reused==144 and self_count==36
    torch.save(captures,root/'evaluation-source-vectors.pt')
    s.dump(root/'evaluation-source-vector-ledger.json',ledger)
    s.dump(root/'completed-audit.json',dict(pass_check=True,raw_records=len(raw),reconstructed_vectors=checked,
        reused_records_checked=reused,self_generations_equal=self_count,training_vector_hashes_checked=len(fit_vectors),
        distinct_evaluation_sources=len(captures),source_hashes_valid=True,learned_hash_valid=True,
        matched_norm_rounding=rounding,norm_device_checks=norm_checks,
        max_cpu_gpu_norm_difference=max(abs(r['cpu_norm']-r['gpu_norm']) for r in norm_checks),
        max_recaptured_gpu_recorded_norm_difference=max(abs(r['gpu_norm']-r['recorded_gpu_norm']) for r in norm_checks),
        source_capture_artifact_sha256=s.sha(root/'evaluation-source-vectors.pt'),
        audit_source_sha256=s.sha(HERE/'audit_subspace.py'),reserve_evaluated=False))
    print('Audited 360 independently reconstructed targets, 144 reused outputs and 756 fitting vectors.',flush=True)


if __name__=='__main__':main()
