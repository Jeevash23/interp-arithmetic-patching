import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import load_data
from value_transfer.timing import equation_cuts
from transformers import AutoTokenizer

def main():
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-0.6B',local_files_only=True)
    data={r['id']:r for r in load_data('development')}
    records=[json.loads(x) for x in (ROOT/'results/v3/development/baselines.jsonl').read_text().splitlines()]
    out=[]
    for r in records:
        if r['condition'] not in ['donor_full','recipient_full']:continue
        task=data[r['id']][r['condition'].split('_')[0]];cuts,info=equation_cuts(tokenizer,r,task)
        out.append({'id':r['id'],'condition':r['condition'],**info,'prefixes':cuts})
    (ROOT/'results/v3/analysis/equation-locator-audit.json').write_text(json.dumps(out,indent=2))
    print('available',sum(r['available'] for r in out),'/',len(out))
    print('paired availability',sum(all(r['available'] for r in out if r['id']==i) for i in sorted({r['id'] for r in out})))
    print('earlier numeric z absent',sum(r['available'] and r['earlier_z_literal_count']==0 for r in out))
    for r in out:print(r['id'],r['condition'],r.get('matched_text',r.get('reason')),r.get('earlier_z_literal_count'))

if __name__=='__main__':main()
