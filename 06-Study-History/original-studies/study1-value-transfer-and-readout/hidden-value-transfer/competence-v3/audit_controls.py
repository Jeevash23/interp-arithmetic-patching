"""Supplement frozen outcomes with explicit origins and alternative predictions."""
import collections,hashlib,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from tasks import calc,dump
from prompt_formats import EXAMPLES


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    root=HERE/'results/qwen4b-fewshot';data={r['id']:r for r in rows(HERE/'development-with-source-controls.jsonl')}
    raw=rows(root/'controls.jsonl');out=root/'analysis';out.mkdir(exist_ok=True)
    manifest=json.loads((HERE/'control-interpretation-freeze.json').read_text())
    for name,digest in manifest['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    actual_sources={'donor','random_source',*next(iter(data.values()))['controls']}
    example_numbers={int(x) for inp,ans in EXAMPLES for x in re.findall(r'\d+',inp+' '+ans)}
    ledger=[];sites=[]
    for item in raw:
        d=item['source_task'];r=item['recipient_task'];g=lambda z:calc(r['remaining'],z,r['k'])
        alternatives=dict(first_operand_only=g(calc(r['first'],d['a'],r['b'])),
            second_operand_only=g(calc(r['first'],r['a'],d['b'])),
            first_operator_only=g(calc(d['first'],r['a'],r['b'])),
            downstream_operator_only=calc(d['remaining'],r['z'],r['k']),
            downstream_constant_only=calc(r['remaining'],r['z'],d['k']),
            downstream_program=calc(d['remaining'],r['z'],d['k']))
        condition=item['condition']
        origin=d if condition in actual_sources else r if condition=='same_example' else None
        ledger.append(dict(key=item['key'],condition=condition,
            origin_kind='activation donor' if condition in actual_sources else 'recipient self' if condition=='same_example' else 'no transferred donor activation',
            actual_activation_origin=origin,comparison_donor_reference=d,
            actual_origin_counterfactual=g(origin['z']) if origin else None,
            alternatives=alternatives,alternative_matches=[k for k,v in alternatives.items() if item['answer']==v],
            example_number_copy=item['answer'] in example_numbers,
            hypothetical_expression_transport_equals_value_transport=g(calc(d['first'],d['a'],d['b']))==item['actual_source_predictions']['counterfactual']))
    (out/'audited-control-origins.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    for layer,role in sorted({(r['layer'],r['role']) for r in raw}):
        rr=[r for r in raw if r['layer']==layer and r['role']==role];by={(r['id'],r['condition']):r for r in rr}
        joint=sum(all(by[id,c]['counterfactual_success'] for c in ['donor','same_z_decomposition','same_z_different_answer']) for id in data)
        matched=[id for id,r in data.items() if str(r['donor']['y'])[0]==str(r['controls']['same_leading_different_value']['y'])[0]]
        both=sum(by[id,'donor']['counterfactual_success'] and by[id,'same_leading_different_value']['actual_source_success'] for id in matched)
        agreement=sum(by[id,'donor']['output_ids']==by[id,'same_leading_different_value']['output_ids'] for id in matched)
        counts={}
        for cond in sorted({r['condition'] for r in rr}):
            kk={r['key'] for r in rr if r['condition']==cond};ll=[r for r in ledger if r['key'] in kk]
            counts[cond]=dict(n=len(ll),matches=dict(collections.Counter(k for r in ll for k in r['alternative_matches'])),example_number_copy=sum(r['example_number_copy'] for r in ll))
        sites.append(dict(layer=layer,role=role,n_pairs=len(data),joint_original_and_both_same_z=joint,
            same_leading_z_and_final_answer=dict(n=len(matched),both_respective_counterfactuals=both,exact_output_agreement=agreement),alternative_counts=counts))
    dump(out/'supplemental-interpretation.json',dict(stage='exploratory development',raw_records=len(raw),sites=sites,
        example_numbers=sorted(example_numbers),primary_cf_example_answer_collisions=sum(r['predictions']['counterfactual'] in {int(a) for _,a in EXAMPLES} for r in data.values()),
        expression_versus_computed_value='not distinguished by final-output predictions alone'))
    print(json.dumps({'raw_records':len(raw),'sites':[{k:v for k,v in s.items() if k!='alternative_counts'} for s in sites]},indent=2))


if __name__=='__main__':main()
