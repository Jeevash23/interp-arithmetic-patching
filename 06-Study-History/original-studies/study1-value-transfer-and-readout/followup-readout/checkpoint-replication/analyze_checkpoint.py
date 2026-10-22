"""Report a completed checkpoint branch without converting feasibility failures to evidence."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=PARENT.parent
sys.path.insert(0,str(PARENT))
from analyze_replication import ci
from value_transfer.tasks import parse,classify

def read(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def main():
    complete=json.loads((HERE/'results/complete.json').read_text())
    freeze=json.loads((HERE/'freeze.json').read_text())
    source_ok=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in freeze['files'].items())
    b=read(HERE/'results/baselines.jsonl');raw=read(HERE/'results/interventions.jsonl')
    parser_ok=all(parse(r['text'])==r['parsed'] for r in b+[r for r in raw if r['status']=='generated'])
    instrumentation=json.loads((HERE/'results/instrumentation.json').read_text())
    cases=[{k:r[k] for k in ['id','pass','native_donor','native_recipient','timings_available','replay_identity','same_example_identity','cached_recomputed_exact','max_cache_logit_abs','cache_logits_pass','write_pass'] if k in r} for r in instrumentation['checks']]
    result={'complete':complete,'frozen_sources_unchanged':source_ok,'parser_audit_pass':parser_ok,'baseline_records':len(b),'intervention_records':len(raw),'instrumentation':{'pass':instrumentation['pass'],'cases':cases}}
    if complete['mechanism_evaluated']:
        idx={(r['id'],r['condition']):r for r in raw};table=[]
        for c in sorted({r['condition'] for r in raw}):
            for width in [0,1,2]:
                ss=[r for r in raw if r['condition']==c and (width==0 or r['width']==width)]
                table.append({'condition':c,'width':width,'n':len(ss),'available':sum(r['status']=='generated' for r in ss),'counterfactual':sum(r['success'] for r in ss),'invalid':sum(r['category']=='invalid' for r in ss),'truncated':sum(r['truncated'] for r in ss)})
        joint=[(r,idx[r['id'],'text_first_digit']) for r in raw if r['condition']=='activation' and r['status']=='generated' and idx[r['id'],'text_first_digit']['status']=='generated']
        two=[r for r in raw if r['condition']=='activation' and r['width']==2]
        delta=[int(idx[r['id'],'text_full_value']['success'])-int(r['success']) for r in two]
        result.update(conditions=table,digit_sequence_agreement={'exact':sum(a['output_ids']==b['output_ids'] for a,b in joint),'joint_available':len(joint)},two_digit_text_advantage=ci(delta,[r['group'] for r in two]))
        result['count_audit_pass']=len(b)==180 and len(raw)==480
    result['audit_pass']=source_ok and parser_ok and result.get('count_audit_pass',True)
    (HERE/'results/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['audit_pass']

if __name__=='__main__':main()
