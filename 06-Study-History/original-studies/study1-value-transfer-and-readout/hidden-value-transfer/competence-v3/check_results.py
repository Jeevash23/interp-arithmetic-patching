"""Audit mathematical, intervention, numerical and provenance invariants."""
import argparse,hashlib,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE/'results/qwen4b-fewshot'
sys.path.insert(0,str(HERE.parent))
from tasks import ROLES,dump


def read(path):
    text=path.read_text();lines=text.splitlines(keepends=True)
    return [json.loads(s) for s in lines if s.endswith('\n')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    data={r['id']:r for r in read(HERE/'development-with-source-controls.jsonl')}
    screen=read(ROOT/'screen.jsonl');assert len({r['key'] for r in screen})==len(screen)
    last_other=last_prefix=0
    for r in screen:
        assert r['id'] in data and 0<=r['layer']<36 and r['role'] in ROLES
        for name in ['counterfactual','recipient','unpatched_counterfactual','unpatched_recipient']:
            s=r[name];lp=s['token_log_probabilities']
            assert len(lp)==len(s['target_ids']) and all(math.isfinite(x) and x<=1e-6 for x in lp)
            assert abs(sum(lp)-s['log_probability'])<=1e-9
            assert abs(sum(lp[:-1])-s['digits_log_probability'])<=1e-9
        expected=r['counterfactual']['log_probability']-r['recipient']['log_probability']-r['unpatched_counterfactual']['log_probability']+r['unpatched_recipient']['log_probability']
        assert abs(expected-r['delta_log_ratio'])<=1e-8
        assert r['counterfactual']['target']==data[r['id']]['predictions']['counterfactual']
        assert r['recipient']['target']==data[r['id']]['recipient']['y']
        if r['layer']==35:
            for key in ['counterfactual','recipient']:
                start=1 if r['role']=='prompt_end' else 0
                assert r[key]['token_log_probabilities'][start:]==r['unpatched_'+key]['token_log_probabilities'][start:],'Final-block causal invariant failed'
            if r['role']=='prompt_end':last_prefix+=1
            else:last_other+=1
    report=dict(stage='partial' if a.partial else 'completed',screen_records=len(screen),unique_keys=True,
        score_arithmetic_and_finiteness=True,final_block_no_effect_at_nonanswer_nodes=last_other,
        final_block_prefix_write_only_changes_first_conditional_token=last_prefix)
    if not a.partial:
        assert len(screen)==5184
        combos={(r['id'],r['layer'],r['role']) for r in screen}
        assert combos=={(id,l,role) for id in data for l in range(36) for role in ROLES}
        controls=read(ROOT/'controls.jsonl');sources=read(ROOT/'source-baselines.jsonl')
        assert len(controls)==1080 and len(sources)==432
        assert len({r['key'] for r in controls})==1080 and len({r['key'] for r in sources})==432
        import torch
        vectors=torch.load(ROOT/'control-vectors.pt',map_location='cpu',weights_only=True)
        source_hashes=0;identity=0;high_probability=0
        baseline={r['key']:r for r in read(ROOT/'baselines.jsonl')};lookup={(r['id'],r['layer'],r['role']):r for r in screen}
        for r in controls:
            if r['source_hash']:
                v=vectors[r['key']]
                assert hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==r['source_hash']
                assert r['write']['write_error']==0 and r['write']['other_positions_max_change']==0
                source_hashes+=1
            if r['condition'] in ['unpatched','same_example']:
                assert r['output_ids']==baseline[r['id']+'/recipient_full']['output_ids'];identity+=1
            if r['condition']=='donor':
                s=lookup[r['id'],r['layer'],r['role']]
                if s['counterfactual']['log_probability']>math.log(.5):
                    assert r['counterfactual_success'],'A >.5 complete sequence probability must give that greedy sequence'
                    high_probability+=1
        assert source_hashes==864 and identity==216
        report.update(control_records=len(controls),source_baseline_records=len(sources),saved_vector_hashes=source_hashes,
            identity_generations_equal_baseline=identity,high_probability_implies_greedy_counterfactual=high_probability,
            reserve_evaluated=False)
    name=f'screen-integrity-partial-{len(screen)}.json' if a.partial else 'completed-result-audit.json'
    dump(ROOT/name,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
