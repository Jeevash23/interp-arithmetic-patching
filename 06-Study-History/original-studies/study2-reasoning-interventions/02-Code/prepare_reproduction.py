"""Prepare a new local workspace from an executed snapshot; never alter a journal."""
import argparse,shutil,json,shlex
from ai04.common import ROOT,readj,writej,digest
a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--name',required=True);args=a.parse_args()
if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('Use a simple alphanumeric workspace name')
p=ROOT/'04-Results'/args.run;fr=readj(p/'freeze.json');src=p/'source';dest=ROOT/'04-Results/reproductions'/args.name
if dest.exists():raise RuntimeError('Refusing an existing reproduction workspace')
for path,h in fr['source'].items():
 if not (src/path).exists() or digest((src/path).read_bytes())!=h:raise RuntimeError('Missing or changed snapshot: '+path)
shutil.copytree(src,dest)
c=fr['config'].copy();c['run']='reproduced/'+args.name;c['baseline_source']=None
model=ROOT/c.get('engine',{}).get('model_path','../study1-value-transfer-and-readout/models/Qwen3-0.6B')
(dest/'models').mkdir(exist_ok=True);(dest/'models/checkpoint').symlink_to(model.resolve(),target_is_directory=True)
c['engine']=c.get('engine',{})|dict(model_path='models/checkpoint')
for path in [c['data'],c['protocol'],c['gate'],*c.get('freeze_extra',[])]:
 if not (dest/path).exists():(dest/path).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest/path)
if c.get('selection') and not (dest/c['selection']).exists():shutil.copy2(ROOT/c['selection'],dest/c['selection'])
config='05-Notes/protocols/reproduction.json';writej(dest/config,c)
writej(dest/'reproduction-origin.json',dict(original_run=args.run,original_freeze_hash=fr['hash'],model_read_only_target=str(model.resolve()),changes=['new run ID','local checkpoint symlink path','regenerate baselines rather than reuse earlier observations'],inference_executed=False))
print('Prepared only. No model inference has run.')
if 'deepseek' in args.run:
 writej(dest/'05-Notes/protocols/second-panel-v1.json',c)
 command=[str(ROOT/'.venv/bin/python'),str(dest/'02-Code/run_second_panel.py')]
else:command=[str(ROOT/'.venv/bin/python'),str(dest/'02-Code/run_experiment.py'),config]
print(' '.join(map(shlex.quote,command)))
