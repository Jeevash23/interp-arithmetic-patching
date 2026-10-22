import time,subprocess,sys
from ai04.common import ROOT,readj,writej,utc
gate=ROOT/'04-Results/audits/deepseek-format-v3/checks.json'
while not gate.exists() or not readj(gate).get('complete'):time.sleep(10)
if not readj(gate)['passed']:
 writej(ROOT/'06-Study-History/second-sequence-status.json',dict(state='qualification_boundary',gate=str(gate.relative_to(ROOT)),finished_utc=utc()))
 raise SystemExit('Format alternative did not qualify; no mechanistic panel')
with (ROOT/'04-Results/development/second-panel-v1.log').open('w') as f:
 p=subprocess.run([sys.executable,str(ROOT/'02-Code/run_second_panel.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
done=ROOT/'04-Results/development/deepseek-panel-v1/completion.json'
writej(ROOT/'06-Study-History/second-sequence-status.json',dict(state=readj(done)['status'] if done.exists() else 'execution_failed',exit_code=p.returncode,finished_utc=utc()))
