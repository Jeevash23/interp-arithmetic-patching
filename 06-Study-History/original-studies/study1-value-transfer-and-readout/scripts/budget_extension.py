"""Separate 1024-token diagnostic, extending only outputs truncated at 512 tokens."""
import json
from run import ROOT,RESULTS,Engine,Journal,dump
from value_transfer.batch import generate_batch

def main():
    config=json.loads((ROOT/'configs/pilot.json').read_text());config.update(max_new_tokens=1024,run_label='v3-budget1024')
    source=Journal(RESULTS/'multilingual-behavioral/baselines.jsonl');assert len(source.rows)==420
    directory=RESULTS/'multilingual-budget1024';journal=Journal(directory/'baselines.jsonl')
    dump(directory/'plan.json',{'amendment':'009','source_budget':512,'diagnostic_budget':1024,'config':config,'selection':'extend every truncated original output; retain all EOS outputs unchanged','interpretation':'paired budget continuation, not independent samples or language-specific localization'})
    truncated=[r for r in source.rows.values() if r['truncated']]
    engine=Engine(config)
    # One complete native-generation check establishes that resume preserves this stream.
    if truncated:
        ref=truncated[0];native=engine.native_generate(ref['prompt_ids'])
        resumed=engine.generate(ref['prompt_ids'],ref['prompt_ids']+ref['output_ids'])
        check={'key':ref['key'],'original_prefix_equal':native[:len(ref['output_ids'])]==ref['output_ids'],'full_output_equal':native==resumed['output_ids'],'native_ids':native,'resumed':resumed}
        check['pass']=check['original_prefix_equal'] and check['full_output_equal'];dump(directory/'native-resume-validation.json',check)
        if not check['pass']:raise RuntimeError('Budget-extension native equivalence failed')
    pending=[]
    for r in source.rows.values():
        if r['key'] in journal.rows:continue
        if not r['truncated']:
            journal.put(r['key'],{**r,'original_512_key':r['key'],'original_truncated':False,'budget_extension_applied':False,'new_extension_tokens':0})
        else:pending.append(r)
    for offset in range(0,len(pending),4):
        chunk=pending[offset:offset+4];outs=generate_batch(engine,[{'prompt_ids':r['prompt_ids'],'prefix_ids':r['prompt_ids']+r['output_ids']} for r in chunk])
        if offset==0 and chunk[0]['key']==ref['key']:
            check['batch_extension_equal']=outs[0]['output_ids']==native;check['pass']=check['pass'] and check['batch_extension_equal'];dump(directory/'native-resume-validation.json',check)
            if not check['pass']:raise RuntimeError('Batched budget extension differs from native continuation')
        for r,out in zip(chunk,outs):
            journal.put(r['key'],{**{k:r[k] for k in ['id','condition','language','prompt','target']},**out,'correct':out['parsed']['answer']==r['target'],'original_512_key':r['key'],'original_output_ids':r['output_ids'],'original_truncated':True,'budget_extension_applied':True,'new_extension_tokens':len(out['fresh_output_ids'])})
        print(f'Budget extension {min(offset+4,len(pending))}/{len(pending)}',flush=True)
    table=[]
    for lang in ['en','es','ta']:
        for c in sorted({r['condition'] for r in source.rows.values()}):
            rr=[r for r in journal.rows.values() if r['language']==lang and r['condition']==c]
            table.append({'language':lang,'condition':c,'n':len(rr),'correct':sum(r['correct'] for r in rr),'parser_failures':sum(r['parsed']['parser_failure'] for r in rr),'truncations':sum(r['truncated'] for r in rr),'extended':sum(r['budget_extension_applied'] for r in rr),'newly_correct_after_extension':sum(r['correct'] and not source.rows[r['key']]['correct'] for r in rr)})
    dump(directory/'summary.json',table);print(json.dumps(table,indent=2))
if __name__=='__main__':main()
