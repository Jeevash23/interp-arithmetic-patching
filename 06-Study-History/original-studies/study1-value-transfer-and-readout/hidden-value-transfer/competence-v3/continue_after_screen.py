"""Finish the current research pipeline after its already-running screen exits."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
root=HERE/'results/qwen4b-fewshot';deadline=time.monotonic()+3*3600
print('Waiting for successful completion of the current screen.',flush=True)
while not (root/'completed-screen.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Current screen did not finish within the pipeline deadline')
    time.sleep(10)
records=[json.loads(s) for s in (root/'screen.jsonl').read_text().splitlines()]
assert len(records)==5184 and len({r['key'] for r in records})==5184
assert len(json.loads((root/'screen-summary.json').read_text())['selected'])==3
steps=[('controls-qwen4b-fewshot',['study_formats.py','--candidate','qwen4b','--format','fewshot','--phase','controls']),
       ('analysis-qwen4b-fewshot',['analyze_formats.py','--candidate','qwen4b-fewshot']),
       ('supplemental-audit',['audit_controls.py'])]
for name,args in steps:
    print('Starting',name,flush=True)
    log=HERE/'logs'/(name+'.log')
    assert not log.exists(),'Refusing to overwrite an existing execution log'
    with log.open('w') as stream:
        subprocess.run([sys.executable,'-u',*args],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Completed',name,flush=True)
(HERE/'pipeline-completed.json').write_text(json.dumps(dict(completed=True,client_date='2026-10-02',stage='development',reserve_evaluated=False),indent=2)+'\n')
