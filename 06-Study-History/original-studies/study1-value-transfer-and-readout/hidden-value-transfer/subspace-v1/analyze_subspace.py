"""Complete-answer outcomes and unchanged development progression heuristics."""
import collections,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;V3=HERE.parent/'competence-v3';ROOT=HERE/'results'
sys.path.insert(0,str(HERE.parent/'competence-v2'));sys.path.insert(0,str(HERE.parent))
from analyze import interval
from tasks import dump,sha


def read(p):return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    assert json.loads((ROOT/'completed.json').read_text())['completed']
    data={r['id']:r for r in read(V3/'development-with-source-controls.jsonl')}
    raw=read(ROOT/'controls.jsonl');assert len(raw)==504
    by={(r['id'],r['condition']):r for r in raw};assert len(by)==504
    original=V3/'results/qwen4b-fewshot'
    base={r['key']:r for r in read(original/'baselines.jsonl')}
    sources={r['key']:r for r in read(original/'source-baselines.jsonl')}
    competence=json.loads((original/'competence.json').read_text());good=set(competence['triple_competent'])
    summaries={}
    for condition in dict.fromkeys(r['condition'] for r in raw):
        rr=[r for r in raw if r['condition']==condition]
        assert len(rr)==36
        counts=dict(n=len(rr),original_counterfactual=sum(r['counterfactual_success'] for r in rr),
            actual_source_counterfactual=sum(bool(r['actual_source_success']) for r in rr),
            unchanged=sum(r['answer']==r['recipient_task']['y'] for r in rr),
            invalid=sum(r['answer'] is None for r in rr),truncated=sum(r['truncated'] for r in rr),
            copied_source_answer=sum(r['answer']==r['comparison_reference']['donor_answer'] for r in rr),
            copied_source_z=sum(r['answer']==r['comparison_reference']['donor_z'] for r in rr),
            first_operand_only=sum('first_operand_only' in r['alternative_matches'] for r in rr),
            both_operand_transfer=sum('operand_transfer' in r['alternative_matches'] for r in rr),
            alternative_matches=dict(collections.Counter(k for r in rr for k in r['alternative_matches'])),
            original_triple_competent_n=len(good),original_triple_competent_cf=sum(r['counterfactual_success'] for r in rr if r['id'] in good),
            reused_records=sum(r['reused_from_key'] is not None for r in rr))
        norms=[r['write']['delta_norm'] for r in rr if r['source_hash'] is not None]
        if norms:counts['mean_actual_displacement_norm']=sum(norms)/len(norms)
        if condition in next(iter(data.values()))['controls']:
            eligible=[r for r in rr if sources[r['id']+'/'+condition+'/full']['correct'] and sources[r['id']+'/'+condition+'/target_remaining']['correct'] and base[r['id']+'/recipient_full']['correct']]
            counts.update(source_triple_competent_n=len(eligible),source_competent_cf=sum(bool(r['actual_source_success']) for r in eligible))
        summaries[condition]=counts
    donor=summaries['donor'];n=len(data);rate=donor['original_counterfactual']/n
    operations={op:dict(n=12,correct=sum(by[id,'donor']['counterfactual_success'] for id,r in data.items() if r['recipient']['remaining']==op)) for op in ['add','sub','mul']}
    joint=sum(all(by[id,c]['counterfactual_success'] for c in ['donor','same_z_decomposition','same_z_different_answer']) for id in data)
    both=sum(by[id,'donor']['counterfactual_success'] and by[id,'same_leading_different_value']['actual_source_success'] for id in data)
    matched=[id for id,r in data.items() if str(r['donor']['y'])[0]==str(r['controls']['same_leading_different_value']['y'])[0]]
    gates=dict(all_planned_controls_present=len(raw)==504,primary_at_least_20pct=rate>=.2,
        noise_advantage_10pp=rate-summaries['isotropic_delta']['original_counterfactual']/n>=.1,
        digit_advantage_10pp=all(rate-summaries[c]['original_counterfactual']/n>=.1 for c in ['text_first_answer_digit','text_actual_first_token']),
        random_subspace_advantage_10pp=rate-summaries['random_subspace']['original_counterfactual']/n>=.1,
        shuffled_labels_advantage_10pp=rate-summaries['label_shuffled']['original_counterfactual']/n>=.1,
        all_recipient_operations=all(v['correct']>0 for v in operations.values()),
        same_z_robustness=all(summaries[c]['original_counterfactual']/n>=max(.2,.8*rate) for c in ['same_z_decomposition','same_z_different_answer']),
        different_z_same_answer=summaries['different_z_same_answer']['actual_source_counterfactual']/n>=.2 and (summaries['different_z_same_answer']['actual_source_counterfactual']-summaries['different_z_same_answer']['copied_source_answer'])/n>=.1 and summaries['different_z_same_answer']['source_triple_competent_n']/n>=.8,
        matched_first_digit_both_succeed=both/n>=.2 and summaries['same_leading_different_value']['source_triple_competent_n']/n>=.8)
    comparisons={condition:interval([(r['group'],int(by[id,'donor']['counterfactual_success'])-int(by[id,condition]['counterfactual_success'])) for id,r in data.items()])
                 for condition in ['isotropic_delta','random_subspace','label_shuffled','text_actual_first_token','whole_vector']}
    manifest=json.loads((ROOT/'learned-freeze.json').read_text())
    assert sha(ROOT/'learned-bases.pt')==manifest['artifact_sha256']
    result=dict(stage='exploratory development; fitted intervention',model='Qwen3-4B-Instruct-2507-FP8',layer=23,role='prompt_end',rank=8,strength=1,
        n_groups=12,n_pairs=36,conditions=summaries,operations=operations,
        joint_original_and_both_same_z=dict(n=36,correct=joint),
        matched_leading_z=dict(n=36,both_correct=both,exact_output_agreement=sum(by[id,'donor']['output_ids']==by[id,'same_leading_different_value']['output_ids'] for id in data)),
        matched_leading_z_and_y=dict(n=len(matched),both_correct=sum(by[id,'donor']['counterfactual_success'] and by[id,'same_leading_different_value']['actual_source_success'] for id in matched),
            exact_output_agreement=sum(by[id,'donor']['output_ids']==by[id,'same_leading_different_value']['output_ids'] for id in matched)),
        paired_differences=comparisons,progression=gates,all_progression_pass=all(gates.values()),
        training_competence=json.loads((ROOT/'training-competence.json').read_text()),
        training_value_variance_explained=manifest['explained_between_value_variance'],reserve_evaluated=False)
    out=ROOT/'analysis';out.mkdir(exist_ok=True);dump(out/'summary.json',result)
    flat=[dict(condition=c,**{k:v for k,v in x.items() if not isinstance(v,dict)}) for c,x in summaries.items()]
    with (out/'conditions.csv').open('w') as f:
        fields=list(dict.fromkeys(k for r in flat for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    print(json.dumps({k:result[k] for k in ['training_competence','operations','progression','all_progression_pass']},indent=2))
    print('COUNTS',json.dumps({k:v['original_counterfactual'] for k,v in summaries.items()}))


if __name__=='__main__':main()
