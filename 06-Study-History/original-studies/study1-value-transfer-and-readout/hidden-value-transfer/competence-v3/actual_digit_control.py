"""Compare activation effects to the donor model's actual first generated token."""
import hashlib,json,sys,time
from pathlib import Path
import study_formats as formats
from tasks import dump,prompt
HERE=Path(__file__).resolve().parent;root=HERE/'results/qwen4b-fewshot';core=formats.core


def read(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    freeze=json.loads((HERE/'actual-digit-freeze.json').read_text())
    for p,h in freeze['files'].items():assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h
    deadline=time.monotonic()+3*3600
    while not (root/'completed-controls.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Primary controls did not complete')
        time.sleep(10)
    core.previous.RESULTS=root
    engine=core.Engine('qwen4b','actual-donor-digit-control');engine.format_name='fewshot'
    data=read(HERE/'development-with-source-controls.jsonl');baseline={r['key']:r for r in read(root/'baselines.jsonl')}
    controls={r['key']:r for r in read(root/'controls.jsonl')};sites=json.loads((root/'screen-summary.json').read_text())['selected']
    journal=core.previous.Journal('actual-donor-digit-control');reuse={}
    for row in data:
        donor=baseline[row['id']+'/donor_full'];forced=[donor['output_ids'][0]]
        assert engine.tok.decode(forced)==str(donor['answer'])[0]
        encoded=engine.encode(*prompt(row['recipient'],row['family']))
        assert encoded['prompt_ids']==baseline[row['id']+'/recipient_full']['prompt_ids']
        for site in sites:
            key=f'{row["id"]}/{site["layer"]}/{site["role"]}/text_first_generated_donor_digit'
            if key in journal.rows:continue
            original=controls[f'{row["id"]}/{site["layer"]}/{site["role"]}/text_first_answer_digit']
            cache_key=(tuple(encoded['prompt_ids']),tuple(forced));old=None
            if forced==original['forced_output_ids']:old=original
            elif cache_key in reuse:old=reuse[cache_key]
            output=engine.generate(encoded,forced=forced) if old is None else {k:old[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}
            saved=journal.put(key,dict(id=row['id'],group=row['group'],layer=site['layer'],role=site['role'],
                condition='text_first_generated_donor_digit',source_baseline_key=donor['key'],
                source_output_ids=donor['output_ids'],counterfactual=row['predictions']['counterfactual'],
                counterfactual_success=output['answer']==row['predictions']['counterfactual'],
                reused_from=old['key'] if old else None,**encoded,**output))
            reuse[cache_key]=saved
    competent=set(json.loads((root/'competence.json').read_text())['triple_competent']);summaries=[]
    for site in sites:
        rr=[r for r in journal.rows.values() if r['layer']==site['layer'] and r['role']==site['role']]
        ds=[controls[f'{r["id"]}/{site["layer"]}/{site["role"]}/donor'] for r in rr]
        summaries.append(dict(layer=site['layer'],role=site['role'],n=len(rr),donor_activation_correct=sum(r['counterfactual_success'] for r in ds),
            actual_digit_text_correct=sum(r['counterfactual_success'] for r in rr),
            exact_generation_agreement=sum(a['output_ids']==b['output_ids'] for a,b in zip(ds,rr)),
            triple_competent_n=sum(r['id'] in competent for r in rr),
            triple_competent_activation_correct=sum(r['id'] in competent and r['counterfactual_success'] for r in ds),
            triple_competent_actual_digit_correct=sum(r['id'] in competent and r['counterfactual_success'] for r in rr)))
    result=dict(stage='supplementary exploratory control',records=len(journal.rows),new_generations=sum(r['reused_from'] is None for r in journal.rows.values()),sites=summaries)
    dump(root/'analysis/actual-donor-digit-summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
