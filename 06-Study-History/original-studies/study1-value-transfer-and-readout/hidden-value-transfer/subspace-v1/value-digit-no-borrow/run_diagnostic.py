"""One no-borrow competence calibration of the fixed rank-eight diagnostic."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=HERE/'results'
sys.path.insert(0,str(PARENT))
import study_subspace as s
import torch


def data():
    rows=[]
    for group,z in enumerate([44,46,48,54,56,58,64,66,68]):
        donor=s.task(z,'sub',1,'add',4)
        sources=dict(same_z_decomposition=s.task(z,'add',1,'add',4),same_z_different_answer=s.task(z,'sub',1,'add',11))
        for rz in [72,76,79]:
            for op,k in [('add',5),('sub',5),('mul',2)]:
                r=s.task(rz,'add',8,op,k);pred=s.predictions(donor,r)
                coarse=s.calc(op,int(str(z)[0]+str(rz)[1:]),k)
                rows.append(dict(id=f'digit-z{z}-r{rz}-{op}',group=group,donor=donor,recipient=r,controls=sources,
                    family='box',predictions=pred,coarse_intermediate_prediction=coarse,discriminative=coarse!=pred['counterfactual']))
    assert len(rows)==81 and sum(r['discriminative'] for r in rows)==72
    text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows);p=HERE/'data.jsonl'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    return rows


def copy_output(r):return {k:r[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}


def extra_baselines(engine,rows):
    main=s.core.previous.Journal('baselines');journal=s.core.previous.Journal('source-baselines')
    reuse={tuple(r['prompt_ids']):r for r in [*main.rows.values(),*journal.rows.values()]}
    for row in rows:
        for name,source in row['controls'].items():
            for kind in ['full','first','target_remaining']:
                key=row['id']+'/'+name+'/'+kind
                if key in journal.rows:continue
                if kind=='target_remaining':text,spans=s.prompt(row['recipient'],kind='counterfactual',z=source['z']);target=row['predictions']['counterfactual']
                else:text,spans=s.prompt(source,kind=kind);target=source['y' if kind=='full' else 'z']
                encoded=engine.encode(text,spans);old=reuse.get(tuple(encoded['prompt_ids']))
                output=copy_output(old) if old else engine.generate(encoded)
                r=journal.put(key,dict(id=row['id'],group=row['group'],condition=name,kind=kind,source_task=source,
                    target=target,correct=output['answer']==target,reused_from=old['key'] if old else None,**encoded,**output))
                reuse[tuple(encoded['prompt_ids'])]=r
    assert len(journal.rows)==486
    print('Auxiliary baseline records',len(journal.rows),flush=True)


def controls(engine,rows,basis):
    baseline=s.core.previous.Journal('baselines').rows;journal=s.core.previous.Journal('controls')
    vp=ROOT/'vectors.pt';state=torch.load(vp,weights_only=True,map_location='cpu') if vp.exists() else dict(sources={},targets={})
    ledger=json.loads((ROOT/'source-ledger.json').read_text()) if (ROOT/'source-ledger.json').exists() else {}
    def encoded_vector(t):
        encoded=engine.encode(*s.prompt(t));key=hashlib.sha256(json.dumps(encoded['prompt_ids']).encode()).hexdigest()
        if key not in state['sources']:
            state['sources'][key]=s.capture(engine,encoded);ledger[key]=dict(task=t,**encoded,vector_sha256=s.digest(state['sources'][key]))
        return encoded,state['sources'][key],key
    text_reuse={(tuple(r['prompt_ids']),tuple(r['forced_output_ids'])):r for r in journal.rows.values() if r['condition']=='text_actual_first_token'}
    names=['donor','same_z_decomposition','same_z_different_answer','isotropic_delta','same_example','unpatched','text_actual_first_token']
    for index,row in enumerate(rows):
        enc,rv,rkey=encoded_vector(row['recipient']);sources=dict(donor=row['donor'],**row['controls'])
        captures={c:encoded_vector(t) for c,t in sources.items()}
        normal=s.displacement(captures['donor'][1],rv,basis)
        for condition in names:
            key=row['id']+'/'+condition;old_record=journal.rows.get(key)
            if old_record and (old_record['target_hash'] is None or key in state['targets']):continue
            target=None;actual_source=None;source_key=None;reused=None;delta=None
            if condition in sources:
                actual_source=sources[condition];source_key=captures[condition][2]
                delta=s.displacement(captures[condition][1],rv,basis)
            elif condition=='isotropic_delta':
                generator=torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
                noise=torch.randn(rv.shape,generator=generator);delta=noise/noise.norm()*normal.norm()
            elif condition=='same_example':actual_source=row['recipient'];source_key=rkey;delta=torch.zeros_like(normal)
            if delta is not None:
                target=(rv.float()+delta).to(rv);state['targets'][key]=target
                if old_record:assert s.digest(target)==old_record['target_hash'];continue
                output=engine.generate(enc,(23,enc['positions']['prompt_end'],target,'replace'))
                assert output['write']['write_error']==0 and output['write']['other_positions_max_change']==0
            elif condition=='unpatched':reused=baseline[row['id']+'/recipient_full'];output=copy_output(reused)
            else:
                assert condition=='text_actual_first_token'
                forced=[baseline[row['id']+'/donor_full']['output_ids'][0]]
                cache_key=(tuple(enc['prompt_ids']),tuple(forced));reused=text_reuse.get(cache_key)
                output=copy_output(reused) if reused else engine.generate(enc,forced=forced)
            if condition in ['unpatched','same_example']:assert output['output_ids']==baseline[row['id']+'/recipient_full']['output_ids']
            r=row['recipient'];d=actual_source if condition in sources else row['donor'];g=lambda z:s.calc(r['remaining'],z,r['k'])
            pred=s.predictions(d,r)
            alternatives=dict(pred,coarse_intermediate=g(int(str(d['z'])[0]+str(r['z'])[1:])),
                first_operand_only=g(s.calc(r['first'],d['a'],r['b'])),second_operand_only=g(s.calc(r['first'],r['a'],d['b'])),
                first_operand_leading_digit=g(s.calc(r['first'],int(str(d['a'])[0]+str(r['a'])[1:]),r['b'])),
                answer_leading_digit=int(str(d['y'])[0]+str(r['y'])[1:]))
            saved=journal.put(key,dict(id=row['id'],group=row['group'],condition=condition,donor=row['donor'],recipient=r,
                actual_source=actual_source,source_key=source_key,recipient_key=rkey,alternatives=alternatives,
                matches=[k for k,v in alternatives.items() if output['answer']==v],full_success=output['answer']==row['predictions']['counterfactual'],
                coarse_success=output['answer']==row['coarse_intermediate_prediction'],discriminative=row['discriminative'],
                target_hash=s.digest(target) if target is not None else None,displacement_norm=float(delta.norm()) if delta is not None else None,
                reused_from=reused['key'] if reused else None,**enc,**output))
            if condition=='text_actual_first_token':text_reuse[cache_key]=saved
        torch.save(state,vp);s.dump(ROOT/'source-ledger.json',ledger)
        if (index+1)%9==0:print('Diagnostic controlled pairs',index+1,'/',len(rows),flush=True)
    assert len(journal.rows)==567 and len(state['targets'])==405
    reconstructed=0
    for r in journal.rows.values():
        if r['target_hash'] is None:continue
        rv=state['sources'][r['recipient_key']]
        if r['condition']=='isotropic_delta':
            row=next(x for x in rows if x['id']==r['id']);_,dv,_=encoded_vector(row['donor'])
            norm=s.displacement(dv,rv,basis).norm();gen=torch.Generator().manual_seed(int(hashlib.sha256(r['key'].encode()).hexdigest()[:8],16))
            noise=torch.randn(rv.shape,generator=gen);delta=noise/noise.norm()*norm
        else:delta=s.displacement(state['sources'][r['source_key']],rv,basis)
        target=(rv.float()+delta).to(rv)
        assert torch.equal(target,state['targets'][r['key']]) and s.digest(target)==r['target_hash'];reconstructed+=1
    for key,v in state['sources'].items():assert s.digest(v)==ledger[key]['vector_sha256']
    anchor={r['condition']:r for r in s.read(PARENT/'results/controls.jsonl') if r['id']=='development-01-mul'}
    for condition in ['same_example','unpatched','text_actual_first_token']:
        assert journal.rows['digit-z56-r76-mul/'+condition]['output_ids']==anchor[condition]['output_ids']
    s.dump(ROOT/'completed.json',dict(completed=True,records=567,reconstructed_targets=reconstructed,self_outputs_equal=81,
        source_hashes=len(state['sources']),anchor_outputs_equal=3,learned_basis_sha256=s.sha(PARENT/'results/learned-bases.pt'),reserve_evaluated=False))


def analyze(rows):
    raw=s.read(ROOT/'controls.jsonl');base=s.read(ROOT/'baselines.jsonl');source=s.read(ROOT/'source-baselines.jsonl')
    competence=json.loads((ROOT/'competence.json').read_text());good=set(competence['triple_competent'])
    by={(r['id'],r['condition']):r for r in raw};out={}
    for cond in dict.fromkeys(r['condition'] for r in raw):
        rr=[r for r in raw if r['condition']==cond];dd=[r for r in rr if r['discriminative']]
        out[cond]=dict(n=len(rr),full=sum(r['full_success'] for r in rr),coarse=sum(r['coarse_success'] for r in rr),
            unchanged=sum(r['answer']==r['recipient']['y'] for r in rr),invalid=sum(r['answer'] is None for r in rr),
            truncated=sum(r['truncated'] for r in rr),reused=sum(r['reused_from'] is not None for r in rr),
            discriminative=dict(n=len(dd),full=sum(r['full_success'] for r in dd),coarse=sum(r['coarse_success'] for r in dd)),
            operations={op:dict(n=27,full=sum(r['full_success'] for r in rr if r['recipient']['remaining']==op),coarse=sum(r['coarse_success'] for r in rr if r['recipient']['remaining']==op)) for op in ['add','sub','mul']},
            triple_competent=dict(n=len(good),full=sum(r['full_success'] for r in rr if r['id'] in good)),
            alternatives={k:sum(k in r['matches'] for r in rr) for k in rr[0]['alternatives']})
    joint={name:dict(n=len(selected),full=sum(all(by[r['id'],c]['full_success'] for c in ['donor','same_z_decomposition','same_z_different_answer']) for r in selected),
        coarse=sum(all(by[r['id'],c]['coarse_success'] for c in ['donor','same_z_decomposition','same_z_different_answer']) for r in selected))
        for name,selected in [('all',rows),('discriminative',[r for r in rows if r['discriminative']])]}
    triplets=[]
    for ten in [4,5,6]:
        for rz in [72,76,79]:
            for op in ['add','sub','mul']:
                rr=[by[f'digit-z{10*ten+u}-r{rz}-{op}','donor'] for u in [4,6,8]]
                triplets.append(dict(donor_tens=ten,recipient_z=rz,operation=op,
                    outputs=[r['answer'] for r in rr],full_targets=[r['alternatives']['counterfactual'] for r in rr],
                    coarse_target=rr[0]['alternatives']['coarse_intermediate'],all_full=all(r['full_success'] for r in rr),
                    all_coarse=all(r['coarse_success'] for r in rr),identical_outputs=len({tuple(r['output_ids']) for r in rr})==1))
    summary=dict(stage='adaptive fixed-intervention development diagnostic',pairs=81,donors=9,recipients=9,discriminative_pairs=72,
        competence=competence,source_competence={c:{k:dict(n=81,correct=sum(r['correct'] for r in source if r['condition']==c and r['kind']==k)) for k in ['full','first','target_remaining']} for c in rows[0]['controls']},
        conditions=out,joint_sources=joint,unit_triplets=dict(n=27,all_full=sum(r['all_full'] for r in triplets),all_coarse=sum(r['all_coarse'] for r in triplets),identical_outputs=sum(r['identical_outputs'] for r in triplets),records=triplets),reserve_evaluated=False)
    s.dump(ROOT/'summary.json',summary)
    print(json.dumps({'conditions':{c:{k:v[k] for k in ['full','coarse','unchanged','discriminative']} for c,v in out.items()},'joint_sources':joint},indent=2))


def main():
    rows=data();paths=[HERE/'PROTOCOL.md',HERE/'run_diagnostic.py',HERE/'data.jsonl',PARENT/'study_subspace.py',PARENT/'source-freeze.json',PARENT/'results/learned-freeze.json',PARENT/'results/learned-bases.pt']
    freeze=dict(stage='adaptive development',files={str(p.relative_to(s.PARENT)):s.sha(p) for p in paths},reserve_evaluated=False)
    fp=HERE/'freeze.json'
    if fp.exists():assert json.loads(fp.read_text())==freeze
    else:s.dump(fp,freeze)
    assert json.loads((PARENT/'results/completed-audit.json').read_text())['pass_check']
    basis=torch.load(PARENT/'results/learned-bases.pt',weights_only=True,map_location='cpu')['learned']
    s.core.previous.RESULTS=ROOT;engine=s.core.Engine('qwen4b','digit-diagnostic');engine.format_name='fewshot'
    s.core.previous.baselines(engine,rows);extra_baselines(engine,rows)
    gate=json.loads((ROOT/'competence.json').read_text())
    if not gate['pass_check']:
        s.dump(ROOT/'stopped-at-competence.json',dict(reason='Unchanged competence gate failed; no interventions',competence=gate));print('Stopped at competence gate.',flush=True);return
    controls(engine,rows,basis);analyze(rows)


if __name__=='__main__':main()
