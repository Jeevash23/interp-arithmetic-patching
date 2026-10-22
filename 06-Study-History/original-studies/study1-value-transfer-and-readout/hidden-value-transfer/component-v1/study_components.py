"""Two fixed native contribution writes, with the original controls unchanged."""
import contextlib,hashlib,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;V3=PARENT/'competence-v3'
sys.path.insert(0,str(PARENT/'competence-v2'));sys.path.insert(0,str(PARENT))
import study as core
import torch
from tasks import ROLES,dump,sha
from prompt_formats import EXAMPLES
import analyze_components


def read(p):return [json.loads(s) for s in p.read_text().splitlines()]


class ComponentEngine(core.Engine):
    component='attention'

    def module(self,layer):
        block=self.model.model.layers[layer]
        return block.self_attn.o_proj if self.component=='attention' else block.mlp.down_proj

    @contextlib.contextmanager
    def patch(self,spec=None,stats=None):
        if spec is None:yield;return
        layer,pos,source,rule=spec
        def hook(module,args,output):
            assert isinstance(output,torch.Tensor) and output.ndim==3 and output.shape[-1]==2560
            assert 0<=pos<output.shape[1]
            if rule=='read':return None
            after=output.clone();old=output[0,pos]
            target=old.clone() if rule=='identity' else source.to(old)
            if rule=='add':target=old+target
            after[0,pos]=target
            if stats is not None and not stats:
                others=after-output;others[0,pos]=0
                stats.update(component=self.component,layer=layer,position=pos,delta_norm=float((target-old).float().norm()),
                    write_error=float((after[0,pos]-target).abs().max()),other_positions_max_change=float(others.abs().max()))
            return after
        handle=self.module(layer).register_forward_hook(hook)
        try:yield
        finally:handle.remove()

    @torch.inference_mode()
    def capture(self,encoded):
        result={};handles=[];positions=[encoded['positions'][role] for role in ROLES]
        for layer in range(self.model.config.num_hidden_layers):
            def hook(module,args,output,layer=layer):
                assert isinstance(output,torch.Tensor) and output.ndim==3 and output.shape[-1]==2560
                result[layer]=output[0,positions].detach().cpu().clone()
            handles.append(self.module(layer).register_forward_hook(hook))
        try:self.forward(encoded['prompt_ids'])
        finally:
            for handle in handles:handle.remove()
        return torch.stack([result[layer] for layer in range(len(handles))])


def audit(component,rows):
    root=HERE/'results'/component;raw=read(root/'controls.jsonl')
    assert len(raw)==360 and len({r['key'] for r in raw})==360
    vectors=torch.load(root/'control-vectors.pt',weights_only=True,map_location='cpu')
    baseline={r['key']:r for r in read(root/'baselines.jsonl')}
    actual={r['id']:r for r in read(V3/'results/qwen4b-fewshot/actual-donor-digit-control.jsonl')}
    by={(r['id'],r['condition']):r for r in raw}
    hashes=identities=0;ledger=[]
    for r in raw:
        if r['source_hash']:
            v=vectors[r['key']]
            assert hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==r['source_hash']
            assert r['write']['component']==component and r['write']['write_error']==0 and r['write']['other_positions_max_change']==0
            hashes+=1
        if r['condition'] in ['same_example','unpatched']:
            assert r['output_ids']==baseline[r['id']+'/recipient_full']['output_ids'];identities+=1
        origin=r['recipient_task'] if r['condition']=='same_example' else None if r['condition'] in ['isotropic_delta','unpatched','text_first_answer_digit'] else r['source_task']
        ledger.append(dict(key=r['key'],component=component,actual_activation_origin=origin,
            reference_donor=r['source_task'],condition=r['condition']))
    assert hashes==288 and identities==72
    matched=[r['id'] for r in rows if str(r['donor']['y'])[0]==str(r['controls']['same_leading_different_value']['y'])[0]]
    good=set(json.loads((root/'competence.json').read_text())['triple_competent'])
    summary=dict(component=component,records=360,vector_hashes=hashes,identity_generations=identities,
        original_triple_competent=dict(n=len(good),counterfactual=sum(by[id,'donor']['counterfactual_success'] for id in good)),
        joint_original_and_both_same_z=sum(all(by[r['id'],c]['counterfactual_success'] for c in ['donor','same_z_decomposition','same_z_different_answer']) for r in rows),
        matched_leading_z_and_y=dict(n=len(matched),both_correct=sum(by[id,'donor']['counterfactual_success'] and by[id,'same_leading_different_value']['actual_source_success'] for id in matched),
            exact_output_agreement=sum(by[id,'donor']['output_ids']==by[id,'same_leading_different_value']['output_ids'] for id in matched)),
        actual_first_donor_token=dict(n=36,counterfactual=sum(actual[r['id']]['answer']==r['predictions']['counterfactual'] for r in rows),
            exact_output_agreement=sum(actual[r['id']]['output_ids']==by[r['id'],'donor']['output_ids'] for r in rows),
            reused_from='competence-v3/results/qwen4b-fewshot/actual-donor-digit-control.jsonl'),reserve_evaluated=False)
    dump(root/'completed-audit.json',summary)
    (root/'actual-source-ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))


def main():
    paths=[HERE/'PROTOCOL.md',HERE/'study_components.py',HERE/'analyze_components.py',HERE/'development-with-source-controls.jsonl',
        HERE/'model-forward-inspection.json',PARENT/'competence-v2/study.py',PARENT/'run.py',PARENT/'tasks.py',PARENT/'prompt_formats.py',V3/'freeze-qwen4b-fewshot.json']
    freeze=dict(stage='adaptive development; two fixed native components',files={str(p.relative_to(PARENT)):sha(p) for p in paths})
    fp=HERE/'freeze.json'
    if fp.exists():assert json.loads(fp.read_text())==freeze
    else:dump(fp,freeze)
    original=V3/'results/qwen4b-fewshot';rows=read(HERE/'development-with-source-controls.jsonl')
    core.previous.RESULTS=HERE/'results/runtime';engine=ComponentEngine('qwen4b','native-components');engine.format_name='fewshot'
    runtime=json.loads((core.previous.RESULTS/'runtime-native-components.json').read_text());runtime.update(calibration_format='fewshot',examples=EXAMPLES)
    dump(core.previous.RESULTS/'runtime-native-components.json',runtime)
    for component in ['attention','mlp']:
        engine.component=component;root=HERE/'results'/component;root.mkdir(parents=True,exist_ok=True)
        inherited={}
        for name in ['baselines.jsonl','source-baselines.jsonl','competence.json']:
            target=root/name
            if target.exists():assert sha(target)==sha(original/name)
            else:shutil.copyfile(original/name,target)
            inherited[name]=dict(source=str((original/name).relative_to(PARENT)),sha256=sha(original/name))
        dump(root/'inherited-data.json',inherited)
        dump(root/'screen-summary.json',dict(selected=[dict(component=component,layer=23,role='prompt_end')],
            selection='fixed native component at earlier whole-vector development site; no component screen'))
        core.previous.RESULTS=root
        if not (root/'instrumentation.json').exists():
            passed=core.instrumentation(engine,rows)
        else:passed=json.loads((root/'instrumentation.json').read_text())['pass_check']
        if not passed:
            dump(root/'stopped.json',dict(reason='Native component instrumentation failed; no causal controls'));continue
        core.controls(engine,rows);audit(component,rows)
        analyze_components.HERE=HERE;sys.argv=['analyze_components','--candidate',component];analyze_components.main()
        dump(root/'completed.json',dict(completed=True,component=component,source_freeze_sha256=sha(fp),reserve_evaluated=False))
    dump(HERE/'completed.json',dict(finished=True,components={c:dict(completed=(HERE/'results'/c/'completed.json').exists(),
        stopped=(HERE/'results'/c/'stopped.json').exists()) for c in ['attention','mlp']},reserve_evaluated=False))


if __name__=='__main__':main()
