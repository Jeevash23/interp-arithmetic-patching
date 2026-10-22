"""Run the declared, dependent local stages; stop on any failed gate."""
import subprocess,sys,time
from ai04.common import ROOT,readj,writej,utc
status=ROOT/'06-Study-History/sequence-status.json'
def run(script,args,log):
 writej(status,dict(stage=script,args=args,started_utc=utc(),state='running'))
 with (ROOT/log).open('w') as f:subprocess.run([sys.executable,str(ROOT/'02-Code'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def verify(name,slug):
 target='04-Results/analysis/'+slug+'-integrity-v1.json'
 run('verify_run.py',[name,'--output',target],'04-Results/analysis/'+slug+'-integrity-v1.log')
 assert readj(ROOT/target)['passed'],target
try:
 while not (ROOT/'04-Results/development/qwen06-development-v2/completion.json').exists():time.sleep(10)
 assert readj(ROOT/'04-Results/development/qwen06-development-v2/completion.json')['status']=='completed'
 verify('development/qwen06-development-v2','development')
 if not (ROOT/'05-Notes/protocols/frozen-selection-v1.json').exists():run('select_sites.py',[],'04-Results/development/selection-v1.log')
 for config,slug in [('final-main-v1','main'),('final-controls-v1','controls')]:
  c=readj(ROOT/'05-Notes/protocols'/f'{config}.json');dest=ROOT/'04-Results'/c['run']/'completion.json'
  if not dest.exists():run('run_experiment.py',['05-Notes/protocols/'+config+'.json'],'04-Results/final/'+config+'.log')
  assert readj(dest)['status']=='completed';verify(c['run'],slug)
 run('final_analysis.py',['--output','04-Results/analysis/final-v1'],'04-Results/analysis/final-v1.log')
 if not (ROOT/'04-Results/development/qwen06-donor-controls-v1/completion.json').exists():run('donor_controls.py',[],'04-Results/development/donor-controls-v1.log')
 if not (ROOT/'04-Results/development/qwen06-symbolic-v1/completion.json').exists():run('run_experiment.py',['05-Notes/protocols/symbolic-v1.json'],'04-Results/development/symbolic-v1.log')
 verify('development/qwen06-symbolic-v1','symbolic')
 writej(status,dict(state='complete',finished_utc=utc()))
except Exception as exc:
 old=readj(status) if status.exists() else {};writej(status,old|dict(state='failed',error=repr(exc),finished_utc=utc()));raise
