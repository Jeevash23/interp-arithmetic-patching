"""Bounded adaptive site revision; retain the original raw screen and results."""
import hashlib,json,math,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'competence-v2'))
sys.path.insert(0,str(HERE.parent))
import study as core
import analyze
from tasks import dump,sha


def read(p):return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    original=HERE/'results/qwen4b-fewshot';dest=HERE/'results/eligible-sites'
    assert (original/'analysis/actual-donor-digit-summary.json').exists()
    assert (original/'completed-result-audit.json').exists()
    dest.mkdir(parents=True,exist_ok=True)
    paths=[HERE/'ELIGIBLE-SITE-FOLLOWUP.md',HERE/'eligible_sites.py',HERE/'freeze-qwen4b-fewshot.json',
        HERE/'development-with-source-controls.jsonl',HERE.parent/'competence-v2/study.py',HERE.parent/'competence-v2/analyze.py',
        HERE.parent/'run.py',HERE.parent/'tasks.py',HERE.parent/'prompt_formats.py']
    freeze=dict(stage='adaptive exploratory development',files={str(p.relative_to(HERE.parent)):sha(p) for p in paths})
    fp=HERE/'eligible-sites-freeze.json'
    if fp.exists():assert json.loads(fp.read_text())==freeze
    else:dump(fp,freeze)
    inherited={}
    for name in ['baselines.jsonl','source-baselines.jsonl','competence.json','screen.jsonl']:
        source=original/name;target=dest/name
        if target.exists():assert sha(target)==sha(source)
        else:shutil.copyfile(source,target)
        inherited[name]=dict(source=str(source.relative_to(HERE)),sha256=sha(source),reused=True)
    dump(dest/'inherited-data.json',inherited)
    screen=read(original/'screen.jsonl');summary=json.loads((original/'screen-summary.json').read_text())
    for site in summary['sites']:
        rr=[r for r in screen if r['layer']==site['layer'] and r['role']==site['role']]
        site['mean_complete_cf_probability']=sum(math.exp(r['counterfactual']['log_probability']) for r in rr)/len(rr)
    summary['selected']=[sorted([s for s in summary['sites'] if s['role']==role],key=lambda s:(-s['mean_complete_cf_probability'],s['layer']))[0]
                         for role in ['b_end','first_stage_end','prompt_end']]
    assert [(s['layer'],s['role']) for s in summary['selected']]==[(14,'b_end'),(20,'first_stage_end'),(23,'prompt_end')]
    summary['selection']='adaptive development: largest mean complete-CF probability, one site per causally eligible role'
    dump(dest/'screen-summary.json',summary)
    rows=read(HERE/'development-with-source-controls.jsonl')
    core.previous.RESULTS=dest
    engine=core.Engine('qwen4b','eligible-controls');engine.format_name='fewshot'
    core.controls(engine,rows)
    controls=read(dest/'controls.jsonl');assert len(controls)==1080
    baseline={r['key']:r for r in read(dest/'baselines.jsonl')}
    vectors=core.torch.load(dest/'control-vectors.pt',weights_only=True,map_location='cpu')
    lookup={(r['id'],r['layer'],r['role']):r for r in screen}
    identity=hashed=high=0
    for r in controls:
        if r['source_hash']:
            v=vectors[r['key']]
            assert hashlib.sha256(v.contiguous().view(core.torch.uint8).numpy().tobytes()).hexdigest()==r['source_hash']
            assert r['write']['write_error']==0 and r['write']['other_positions_max_change']==0
            hashed+=1
        if r['condition'] in ['unpatched','same_example']:
            assert r['output_ids']==baseline[r['id']+'/recipient_full']['output_ids'];identity+=1
        if r['condition']=='donor' and lookup[r['id'],r['layer'],r['role']]['counterfactual']['log_probability']>math.log(.5):
            assert r['counterfactual_success'];high+=1
    assert identity==216 and hashed==864
    actual={r['id']:r for r in read(original/'actual-donor-digit-control.jsonl')}
    supplemental=[]
    for site in summary['selected']:
        rr=[r for r in controls if r['layer']==site['layer'] and r['role']==site['role']]
        by={(r['id'],r['condition']):r for r in rr}
        matched=[r['id'] for r in rows if str(r['donor']['y'])[0]==str(r['controls']['same_leading_different_value']['y'])[0]]
        supplemental.append(dict(site=site,
            joint_original_and_both_same_z=sum(all(by[r['id'],c]['counterfactual_success'] for c in ['donor','same_z_decomposition','same_z_different_answer']) for r in rows),
            same_leading_z_and_y=dict(n=len(matched),both_correct=sum(by[id,'donor']['counterfactual_success'] and by[id,'same_leading_different_value']['actual_source_success'] for id in matched),
                exact_output_agreement=sum(by[id,'donor']['output_ids']==by[id,'same_leading_different_value']['output_ids'] for id in matched)),
            actual_first_digit=dict(n=len(rows),counterfactual=sum(actual[r['id']]['answer']==r['predictions']['counterfactual'] for r in rows),
                donor_exact_output_agreement=sum(actual[r['id']]['output_ids']==by[r['id'],'donor']['output_ids'] for r in rows),
                source='qwen4b-fewshot/actual-donor-digit-control.jsonl; identical text-only prompts reused')))
    dump(dest/'supplemental-interpretation.json',supplemental)
    analyze.HERE=HERE;sys.argv=['analyze','--candidate','eligible-sites'];analyze.main()
    dump(dest/'completed.json',dict(completed=True,controls=1080,identity=identity,vector_hashes=hashed,
        high_probability_implies_greedy=high,source_freeze_sha256=sha(fp),reserve_evaluated=False))


if __name__=='__main__':main()
