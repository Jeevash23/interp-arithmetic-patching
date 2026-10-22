"""Read-only audit of raw results, saved vectors, and immutable data/model hashes."""
import hashlib
import json
from collections import Counter
from analyze import ROOT,read
from value_transfer.tasks import load_data,parse,classify
from value_transfer.engine import torch,vector_hash

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    data={r['id']:r for r in load_data('development')};checks=[]
    def record(name,ok,**details):checks.append({'check':name,'pass':bool(ok),**details})
    manifest=json.loads((ROOT/'data/model-manifest.json').read_text())
    for name,meta in manifest['files'].items():record('checkpoint_file_hash',sha(ROOT/'models/Qwen3-0.6B'/name)==meta['sha256'],file=name)
    split_manifest=json.loads((ROOT/'data/split-manifest.json').read_text())
    # Dataset hashes are read without evaluating any validation prompts.
    for split,expected in [('development','b7f7d559169fd3ab85cbc9b26b8319dffe3908b5285aa102ed863a97cb7a6452'),('validation','630dd6e32e8b7a48f62743f0791e6a29123cb27c81cb68ad32809ec47c0bb697')]:
        path=ROOT/f'data/{split}.jsonl';record('split_immutable',sha(path)==expected,split=split,sha256=sha(path))
    journals={'baselines':('development/baselines.jsonl',700),'screen':('development/screen.jsonl',480),'control_baselines':('development/controls-n100/control-baselines.jsonl',700),'controls':('development/controls-n100/controls.jsonl',1500),'textual':('development/textual-scratchpad.jsonl',120),'clamp':('development/clamp-mediation.jsonl',80),'language_baselines':('multilingual-behavioral/baselines.jsonl',420)}
    journals['digit_controls']=('development/digit-controls.jsonl',200)
    journals['budget_extension']=('multilingual-budget1024/baselines.jsonl',420)
    raw={}
    for name,(rel,n) in journals.items():
        rr=read(ROOT/'results/v3'/rel);raw[name]=rr
        record('journal_count',len(rr)==n,journal=name,actual=len(rr),expected=n)
        record('journal_unique_keys',len({r['key'] for r in rr})==len(rr),journal=name)
        available=[r for r in rr if r.get('status')!='unavailable_timing']
        parser_mismatches=[r['key'] for r in available if parse(r['text'])!=r['parsed']]
        record('parser_reproducibility',not parser_mismatches,journal=name,mismatches=parser_mismatches)
        ceiling=1024 if name=='budget_extension' else 512
        token_mismatches=[r['key'] for r in available if r['replayed_output_ids']+r['fresh_output_ids']!=r['output_ids'] or r['total_output_tokens']!=len(r['output_ids']) or len(r['output_ids'])>ceiling]
        record('token_bookkeeping',not token_mismatches,journal=name,mismatches=token_mismatches)
        if name in ['baselines','control_baselines','language_baselines','budget_extension']:
            wrong=[r['key'] for r in available if r['correct']!=(r['parsed']['answer']==r['target'])]
            record('baseline_scoring_reproducibility',not wrong,journal=name,mismatches=wrong)
        if name in ['screen','controls','textual','clamp','digit_controls']:
            wrong=[r['key'] for r in available if any(classify(r['parsed'],data[r['id']]['predictions'])[k]!=r[k] for k in ['success','category','prediction_matches'])]
            record('causal_scoring_reproducibility',not wrong,journal=name,mismatches=wrong)
    base={r['key']:r for r in raw['baselines']}
    original_language={r['key']:r for r in raw['language_baselines']}
    wrong=[r['key'] for r in raw['budget_extension'] if r['output_ids'][:len(original_language[r['key']]['output_ids'])]!=original_language[r['key']]['output_ids'] or (not original_language[r['key']]['truncated'] and r['output_ids']!=original_language[r['key']]['output_ids'])]
    record('budget_extension_preserves_original_tokens',not wrong,n=len(raw['budget_extension']),mismatches=wrong)
    discovery={r['id']:r for r in raw['screen'] if r['timing']=='pre_z' and r['layer']==26}
    expanded={r['id']:r for r in raw['controls'] if r['condition']=='donor'}
    same=[i for i in discovery if i in expanded and discovery[i].get('output_ids')==expanded[i].get('output_ids')]
    record('discovery_reproduced_in_controls',len(same)==20,n=20,exact_output_matches=len(same))
    active_clamps=[r for r in raw['clamp'] if r.get('donor_source_hash')]
    wrong=[r['key'] for r in active_clamps if r['donor_source_hash']!=discovery[r['id']]['donor_source_hash'] or r['donor_prefix_ids']!=discovery[r['id']]['donor_prefix_ids']]
    record('clamp_sources_match_screen',not wrong,n=len(active_clamps),mismatches=wrong)
    for condition in ['isotropic_delta']:
        rr=[r for r in raw['controls'] if r['condition']==condition and r['status']=='generated']
        wrong=[r['key'] for r in rr if abs(r['matched_delta_norm']-r['random_delta_norm'])>max(1e-5,1e-6*r['matched_delta_norm']) or r['write']['write_error']!=0]
        record('matched_delta_norm_and_write',not wrong,n=len(rr),mismatches=wrong)
    for condition in ['unpatched','same_example']:
        rr=[r for r in raw['controls'] if r['condition']==condition and r['status']=='generated']
        wrong=[r['key'] for r in rr if r['output_ids']!=base[f'{r["id"]}/recipient_full']['output_ids']]
        record('full_development_identity',not wrong,condition=condition,n=len(rr),mismatches=wrong)
    for label,path in [('screen','development/screen-batched-donor-vectors.pt'),('controls','development/controls-n100/control-vectors.pt')]:
        p=ROOT/'results/v3'/path;vectors=torch.load(p,weights_only=True) if p.exists() else {}
        rr=[r for r in raw[label] if r.get('donor_source_hash')]
        missing=[r['key'] for r in rr if r['key'] not in vectors]
        wrong=[r['key'] for r in rr if r['key'] in vectors and vector_hash(vectors[r['key']])!=r['donor_source_hash']]
        record('stored_source_vector_hashes',not wrong and not missing,stage=label,checked=len(rr)-len(missing),missing_vectors=missing,mismatches=wrong)
    out={'pass':all(r['pass'] for r in checks),'checks':checks,'validation_model_runs':len(list((ROOT/'results').glob('**/validation/*.jsonl'))),'notes':['The 80 source vectors initially absent from the interrupted serial screen were reconstructed and matched their original hashes; see source-reconstruction.json. Missing stored vectors fail the audit.','No validation generation is performed by this audit.']}
    (ROOT/'results/v3/analysis/artifact-audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'pass':out['pass'],'failed_checks':[r for r in checks if not r['pass']]},indent=2))
    if not out['pass']:raise SystemExit(1)
if __name__=='__main__':main()
