"""Audit textual exposure before the labelled output sites without model inference."""
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import load_data
from transformers import AutoTokenizer

def main():
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-0.6B',local_files_only=True)
    data={r['id']:r for r in load_data('development')}
    rows=[json.loads(x) for x in (ROOT/'results/v3/development/baselines.jsonl').read_text().splitlines()]
    audit=[]
    for row in rows:
        if row['condition'] not in ['donor_full','recipient_full']:continue
        side=row['condition'].split('_')[0];t=data[row['id']][side]
        text=row['text'];thinking=text.split('</think>')[0]
        zd=re.findall(r'(?<!\d)'+str(t['z'])+r'(?!\d)',thinking)
        yd=re.findall(r'(?<!\d)'+str(t['y'])+r'(?!\d)',thinking)
        audit.append({'id':row['id'],'side':side,'z':t['z'],'y':t['y'],'z_literal_in_thinking':bool(zd),'y_literal_in_thinking':bool(yd),'z_count':len(zd),'y_count':len(yd),'contains_think_close':'</think>' in text,'intermediate_label_present':'Intermediate:' in text.split('</think>')[-1]})
    out=ROOT/'results/v3/analysis';out.mkdir(parents=True,exist_ok=True)
    (out/'prefix-exposure.json').write_text(json.dumps(audit,indent=2))
    summary={'full_generations':len(audit),'z_literal_in_thinking':sum(r['z_literal_in_thinking'] for r in audit),'final_answer_literal_in_thinking':sum(r['y_literal_in_thinking'] for r in audit),'note':'Literal occurrences are exposure checks, not semantic identification of the first calculation or its correctness.'}
    (out/'prefix-exposure-summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
