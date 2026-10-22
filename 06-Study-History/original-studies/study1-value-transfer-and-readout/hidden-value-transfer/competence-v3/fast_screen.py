"""Verify exact upstream memoization before resuming the original screen journal."""
import argparse,json,sys,time
from pathlib import Path
import study_formats as formats
from fast_engine import FastEngine,BaseEngine
from tasks import dump,sha,prompt,ROLES
HERE=Path(__file__).resolve().parent;core=formats.core


def freeze():
    files=[HERE/'SCREEN-EXECUTION-AMENDMENT.md',HERE/'fast_engine.py',HERE/'fast_screen.py',
        HERE/'freeze-qwen4b-fewshot.json',HERE/'selected-configuration.json']
    data=dict(client_date='2026-10-02',purpose='execution equivalence only',files={p.name:sha(p) for p in files})
    p=HERE/'fast-execution-freeze.json'
    if p.exists():assert json.loads(p.read_text())==data
    else:dump(p,data)


def verify():
    core.previous.RESULTS=HERE/'results/qwen4b-fewshot'
    engine=FastEngine('qwen4b','screen-fast-verification');engine.format_name='fewshot'
    rows=[json.loads(s) for s in (HERE/'development-with-source-controls.jsonl').read_text().splitlines()]
    checks=[];generation=[];start=time.monotonic()
    for row in rows[:4]:
        enc=engine.encode(*prompt(row['recipient'],row['family']))
        donor=engine.capture(engine.encode(*prompt(row['donor'],row['family'])))
        engine._upstream.clear()
        for layer in [0,18,35]:
            for ri,role in enumerate(ROLES):
                spec=(layer,enc['positions'][role],donor[layer,ri],'replace')
                ref=BaseEngine.forward(engine,enc['prompt_ids'],spec=spec)
                fast=engine.cached_forward(enc['prompt_ids'],spec=spec)
                delta=float((ref-fast).abs().max())
                checks.append(dict(id=row['id'],layer=layer,role=role,max_logit_abs=delta,pass_check=delta<=1e-5))
        spec=(18,enc['positions']['prompt_end'],donor[18,3],'replace')
        ref=engine.generate(enc,spec);emitted=[]
        for _ in range(16):
            scores=engine.cached_forward(enc['prompt_ids']+emitted,spec=spec)[-1]
            token=int(scores.argmax());emitted.append(token)
            if token in engine.eos:break
        generation.append(dict(id=row['id'],reference_ids=ref['output_ids'],optimized_ids=emitted,pass_check=emitted==ref['output_ids']))
        print('Numerical equivalence',row['id'],all(x['pass_check'] for x in checks),flush=True)
    row=rows[0];enc=engine.encode(*prompt(row['recipient'],row['family']));donor=engine.capture(engine.encode(*prompt(row['donor'],row['family'])))
    old=[json.loads(s) for s in (core.previous.RESULTS/'screen.jsonl').read_text().splitlines() if json.loads(s)['id']==row['id']]
    assert len(old)==144,'Reference first pair incomplete'
    cf=row['predictions']['counterfactual'];normal=row['recipient']['y'];comparisons=[]
    for record in old:
        layer=record['layer'];role=record['role'];ri=ROLES.index(role)
        scores=engine.score_candidates(enc,[cf,normal],(layer,enc['positions'][role],donor[layer,ri],'replace'))
        delta=max(abs(a-b) for answer,name in [(cf,'counterfactual'),(normal,'recipient')] for a,b in zip(scores[answer]['token_log_probabilities'],record[name]['token_log_probabilities']))
        comparisons.append(dict(layer=layer,role=role,max_token_log_probability_difference=delta,pass_check=delta<=1e-5))
    result=dict(pass_check=all(x['pass_check'] for x in checks+generation+comparisons),logit_checks=checks,generation_checks=generation,reference_first_pair= comparisons,seconds=time.monotonic()-start)
    dump(HERE/'fast-equivalence.json',result)
    print('All equivalence checks',result['pass_check'],'seconds',result['seconds'],flush=True)
    assert result['pass_check']


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    assert a.verify != a.run
    freeze()
    if a.verify:verify()
    else:
        assert json.loads((HERE/'fast-equivalence.json').read_text())['pass_check']
        core.Engine=FastEngine
        sys.argv=['study_formats.py','--candidate','qwen4b','--format','fewshot','--phase','screen']
        formats.main()
