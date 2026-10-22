"""Complete baselines in validated batches, reusing identical seeded prompts explicitly."""
import argparse
import json
from run import ROOT,RESULTS,Engine,Journal,load_data,prompt,baseline,dump
from value_transfer.batch import generate_batch

def baseline_batched(engine,rows,journal):
    specs=[('donor_full','donor','full'),('recipient_full','recipient','full'),('donor_first','donor','first'),('recipient_first','recipient','first'),('donor_remaining','donor','remaining'),('recipient_remaining','recipient','remaining'),('textual_counterfactual','recipient','remaining')]
    tasks=[]
    for row in rows:
        for label,side,kind in specs:
            key=f'{row["id"]}/{label}'
            if key in journal.rows:continue
            supplied=row['donor']['z'] if label=='textual_counterfactual' else None
            target=row['predictions']['counterfactual'] if supplied is not None else row[side]['z' if kind=='first' else 'y']
            text=prompt(row[side],row['family'],kind,supplied)
            tasks.append({'key':key,'id':row['id'],'condition':label,'prompt':text,'target':target,'task':row[side],'prompt_ids':engine.encode(text)})
    fill_jobs(engine,tasks,journal)
    return baseline(engine,rows,journal)

def fill_jobs(engine,tasks,journal):
    complete_prompt={tuple(r['prompt_ids']):r for r in journal.rows.values()}
    pending={}
    for task in tasks:pending.setdefault(tuple(task['prompt_ids']),[]).append(task)
    total=len(tasks);done=0
    def store(out,duplicates,reused=None):
        nonlocal done
        result={k:v for k,v in out.items() if k not in ['key','id','condition','prompt','target','correct','task']}
        for task in duplicates:
            row={**result,**{k:v for k,v in task.items() if k!='key'},'correct':out['parsed']['answer']==task['target'],'reused_exact_prompt_from':reused}
            saved=journal.put(task['key'],row);done+=1
            if reused is None:reused=saved['key']
    for ids in list(pending):
        if ids in complete_prompt:
            store(complete_prompt[ids],pending.pop(ids),complete_prompt[ids]['key'])
    items=sorted(pending.items(),key=lambda item:(item[1][0]['condition'],len(item[0])))
    for offset in range(0,len(items),4):
        chunk=items[offset:offset+4]
        outs=generate_batch(engine,[{'prompt_ids':list(ids)} for ids,_ in chunk])
        for (_,duplicates),out in zip(chunk,outs):store(out,duplicates)
        print(f'Baseline records filled {done}/{total}; unique batch {offset//4+1}/{(len(items)+3)//4}',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=100);args=parser.parse_args()
    assert json.loads((RESULTS/'smoke/extension-validation.json').read_text())['pass']
    engine=Engine();rows=load_data('development')[:args.n];j=Journal(RESULTS/'development/baselines.jsonl')
    summary=baseline_batched(engine,rows,j);dump(RESULTS/'development/baselines-summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
