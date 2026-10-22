"""Deterministic fresh paired programs and a pre-outcome source/config freeze."""
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import task,predictions,prompt as original_prompt,FORMAT,FINAL_FORMAT,op,load_data

def prompt(t,family,kind='full',supplied=None):
    if family=='box':return original_prompt(t,'dev_box',kind,supplied)
    first=f"A register initially contains {t['a']}. First, "+(f"increase its contents by {t['b']}." if t['first']=='add' else f"decrease its contents by {t['b']}.")
    second={'add':f"add {t['k']} to the intermediate value",'sub':f"subtract {t['k']} from the intermediate value",'mul':f"multiply the intermediate value by {t['k']}"}[t['remaining']]
    if kind=='full':return f"{first} The value after this update is the intermediate value. Next, {second}. Give the intermediate and final values.\n{FORMAT}"
    if kind=='remaining':return f"The intermediate value in a register is {t['z'] if supplied is None else supplied}. Next, {second}. Give the final value.\n{FINAL_FORMAT}"
    raise ValueError(kind)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def make_data():
    rng=random.Random(20264909);rows=[];seen=set()
    prior={original_prompt(r['donor'],r['family']) for r in load_data('development')}
    for g in range(30):
        for attempt in range(10000):
            short=5+g%5;long=10*short+rng.randint(1,9)
            fr=['add','sub'][g%2];fd='sub' if fr=='add' else 'add'
            gr=['add','sub','mul'][g%3];gd=rng.choice([v for v in ['add','sub','mul'] if v!=gr])
            r=task(rng.randint(45,89),fr,gr,rng.randint(2,4),rng)
            ds=task(short,fd,gd,rng.randint(2,4),rng)
            dl={**ds,'a':ds['a']+long-short,'z':long,'y':op(long,ds['k'],gd)}
            if str(r['z'])[0]==str(short) or r['z']==long:continue
            if any(prompt(d,'box') in prior for d in [ds,dl]):continue
            if json.dumps(r,sort_keys=True) in seen:continue
            if not all(len(set(predictions(d,r)[k] for k in ['counterfactual','recipient_answer','donor_answer','donor_z','recipient_z','operand_transfer']))==6 and min(d['y'],r['y'],predictions(d,r)['counterfactual'])>0 for d in [ds,dl]):continue
            break
        else:raise RuntimeError('Could not construct a valid group')
        seen.add(json.dumps(r,sort_keys=True))
        for family in ['box','register']:
            for width,d in [(1,ds),(2,dl)]:
                rows.append({'id':f'readout-{g:02}-{family}-w{width}','group':g,'family':family,'width':width,'leading_digit':short,'donor':d,'recipient':r,'predictions':predictions(d,r)})
    assert len(rows)==120 and len(seen)==30
    return rows

def main():
    if (HERE/'freeze.json').exists():raise RuntimeError('Freeze already exists; do not overwrite it')
    rows=make_data()
    (HERE/'data.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    cfg=json.loads((ROOT/'configs/pilot.json').read_text())
    cfg.update(seed=20262909,run_label='readout-replication',layers=[26],timings=['pre_z'])
    (HERE/'config.json').write_text(json.dumps({'engine':cfg,'sampling_seeds':[20262909,20263909],'groups':30,'pair_seed_records':240},indent=2)+'\n')
    required=[HERE/n for n in ['PROTOCOL.md','prepare.py','run_replication.py','analyze_replication.py','data.jsonl','config.json']]
    required += [ROOT/n for n in ['src/value_transfer/engine.py','src/value_transfer/batch.py','src/value_transfer/tasks.py','src/value_transfer/timing.py','scripts/run.py','scripts/baselines_batched.py','scripts/clamp_mediation.py','scripts/controls.py','data/model-manifest.json']]
    assert all(p.exists() for p in required)
    freeze={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'fixed before any follow-up model evaluation','original_validation':'reserved and not evaluated','files':{str(p.relative_to(ROOT)):digest(p) for p in required}}
    (HERE/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps({'pairs':len(rows),'program_groups':30,'seeds':2,'frozen_files':len(required)}))

if __name__=='__main__':main()
