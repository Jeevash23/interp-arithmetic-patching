"""Check source freezes, expected record counts and completed-study decisions."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    manifests=['competence-v3/freeze-qwen4b-fewshot.json','competence-v3/eligible-sites-freeze.json',
        'subspace-v1/source-freeze.json','subspace-v1/value-digit-diagnostic/freeze.json',
        'subspace-v1/value-digit-no-borrow/freeze.json','subspace-v1/value-digit-qualified-sources/freeze.json',
        'component-v1/freeze.json']
    checks=[]
    for name in manifests:
        obj=json.loads((HERE/name).read_text())
        for f,h in obj['files'].items():assert sha(HERE/f)==h,(name,f)
        checks.append(dict(manifest=name,files=len(obj['files']),sha256=sha(HERE/name)))
    expected={
        'competence-v3/results/qwen4b-fewshot/baselines.jsonl':216,
        'competence-v3/results/qwen4b-fewshot/screen.jsonl':5184,
        'competence-v3/results/qwen4b-fewshot/source-baselines.jsonl':432,
        'competence-v3/results/qwen4b-fewshot/controls.jsonl':1080,
        'competence-v3/results/qwen4b-fewshot/actual-donor-digit-control.jsonl':108,
        'competence-v3/results/eligible-sites/controls.jsonl':1080,
        'subspace-v1/results/training-baselines.jsonl':1512,
        'subspace-v1/results/controls.jsonl':504,
        'subspace-v1/value-digit-diagnostic/results/baselines.jsonl':486,
        'subspace-v1/value-digit-diagnostic/results/source-baselines.jsonl':486,
        'subspace-v1/value-digit-no-borrow/results/baselines.jsonl':486,
        'subspace-v1/value-digit-no-borrow/results/source-baselines.jsonl':486,
        'subspace-v1/value-digit-qualified-sources/results/baselines.jsonl':486,
        'subspace-v1/value-digit-qualified-sources/results/source-baselines.jsonl':243,
        'subspace-v1/value-digit-qualified-sources/results/controls.jsonl':567,
        'component-v1/results/attention/controls.jsonl':360,
        'component-v1/results/mlp/controls.jsonl':360}
    journals=[]
    for name,n in expected.items():
        rr=read(HERE/name)
        assert len(rr)==n and len({r['key'] for r in rr})==n,name
        journals.append(dict(path=name,records=n,unique_keys=True,sha256=sha(HERE/name)))
    for failed in ['value-digit-diagnostic','value-digit-no-borrow']:
        root=HERE/'subspace-v1'/failed/'results'
        assert not json.loads((root/'competence.json').read_text())['pass_check']
        assert (root/'stopped-at-competence.json').exists() and not (root/'controls.jsonl').exists()
    root=HERE/'subspace-v1/value-digit-qualified-sources/results'
    assert json.loads((root/'competence.json').read_text())['pass_check']
    q=json.loads((root/'summary.json').read_text())
    assert all(q['conditions'][c]['discriminative']['full']==0 for c in ['donor','same_value_coupled'])
    assert q['joint_sources']['all']['coarse']==q['joint_sources']['all']['full']==0
    for c in ['qwen4b-fewshot','eligible-sites']:
        assert not json.loads((HERE/'competence-v3/results'/c/'analysis/summary.json').read_text())['promising_sites']
    assert not json.loads((HERE/'subspace-v1/results/analysis/summary.json').read_text())['all_progression_pass']
    assert json.loads((HERE/'subspace-v1/results/completed-audit.json').read_text())['pass_check']
    for c in ['attention','mlp']:
        root=HERE/'component-v1/results'/c
        assert json.loads((root/'instrumentation.json').read_text())['pass_check']
        assert json.loads((root/'completed.json').read_text())['completed']
        assert not json.loads((root/'analysis/summary.json').read_text())['promising_sites']
    result=dict(client_date='2026-10-02',pass_check=True,source_freezes=checks,journals=journals,
        failed_diagnostic_gates_honored=True,qualified_source_comparison_complete=True,
        no_intervention_passed_full_progression=True,learned_artifact_sha256=sha(HERE/'subspace-v1/results/learned-bases.pt'),
        manuscript_sha256=sha(HERE/'paper/manuscript.md'),reserve_evaluated=False,multilingual_interventions_evaluated=False,
        caution='Record counts include explicitly reused outputs and repeated donors; do not sum them as independent observations.')
    (HERE/'completed-round-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_freezes','journals']},indent=2))
    print('Verified',len(checks),'source freezes and',len(journals),'unique-key journals.')


if __name__=='__main__':main()
