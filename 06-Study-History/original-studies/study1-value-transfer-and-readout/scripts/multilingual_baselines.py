"""Exploratory draft-translation competence only; no cross-language activation inference."""
import json
from run import ROOT,RESULTS,Engine,Journal,load_data,dump
from baselines_batched import fill_jobs
from value_transfer.multilingual import translated_prompt

def main():
    engine=Engine();rows=load_data('development')[:20]
    output=RESULTS/'multilingual-behavioral';output.mkdir(exist_ok=True)
    journal=Journal(output/'baselines.jsonl');english=Journal(RESULTS/'development/baselines.jsonl')
    specs=[('donor_full','donor','full'),('recipient_full','recipient','full'),('donor_first','donor','first'),('recipient_first','recipient','first'),('donor_remaining','donor','remaining'),('recipient_remaining','recipient','remaining'),('textual_counterfactual','recipient','remaining')]
    plan={'n_pairs':20,'languages':['en','es','ta'],'translation_status':'[redacted]-reviewed development drafts; no independent human verification','source':'matched first 20 development pairs','activation_interventions':False,'generation_config':engine.config,'labels':'Intermediate and Final retained in English; answers numeric; reasoning language unconstrained','amendment':'007'}
    dump(output/'plan.json',plan)
    for language in ['en','es','ta']:
        tasks=[]
        for row in rows:
            for label,side,kind in specs:
                key=f'{language}/{row["id"]}/{label}'
                if key in journal.rows:continue
                if language=='en':
                    source=english.rows[f'{row["id"]}/{label}'];journal.put(key,{**source,'language':'en','reused_english_baseline_key':source['key']});continue
                supplied=row['donor']['z'] if label=='textual_counterfactual' else None
                target=row['predictions']['counterfactual'] if supplied is not None else row[side]['z' if kind=='first' else 'y']
                text=translated_prompt(row[side],row['family'],language,kind,supplied)
                tasks.append({'key':key,'id':row['id'],'condition':label,'language':language,'prompt':text,'target':target,'task':row[side],'prompt_ids':engine.encode(text)})
        fill_jobs(engine,tasks,journal);print(f'Completed language {language}',flush=True)
    table=[]
    for language in ['en','es','ta']:
        for label,_,_ in specs:
            rr=[journal.rows[f'{language}/{row["id"]}/{label}'] for row in rows]
            table.append({'language':language,'condition':label,'n':len(rr),'correct':sum(r['correct'] for r in rr),'percent':100*sum(r['correct'] for r in rr)/len(rr),'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr),'mean_prompt_tokens':sum(len(r['prompt_ids']) for r in rr)/len(rr),'mean_output_tokens':sum(r['total_output_tokens'] for r in rr)/len(rr)})
    dump(output/'summary.json',table);print(json.dumps(table,indent=2),flush=True)
if __name__=='__main__':main()
