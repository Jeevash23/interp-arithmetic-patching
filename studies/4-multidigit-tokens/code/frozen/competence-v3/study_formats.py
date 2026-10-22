"""Prospectively bounded prompt calibration after both system-only candidates fail."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'competence-v2'
sys.path.insert(0,str(V2));sys.path.insert(0,str(HERE.parent))
import study as core
from tasks import prompt as original_prompt,sha,dump
from prompt_formats import EXAMPLES


def compact_prompt(t,family='box',kind='full',z=None):
    if kind in ['remaining','counterfactual']:return original_prompt(t,family,kind,z)
    text='Start with '+str(t['a']);spans={'a_end':len(text)}
    text+='. First, '+('add ' if t['first']=='add' else 'subtract ')+str(t['b'])
    spans['b_end']=len(text);text+='.';spans['first_stage_end']=len(text)
    if kind=='full':
        text+=' Then, '+{'add':f"add {t['k']} to the result.",'sub':f"subtract {t['k']} from the result.",'mul':f"multiply the result by {t['k']}."}[t['remaining']]
    return text+' Return only the final integer, with no explanation.',spans


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['qwen4b','qwen25'],required=True)
    ap.add_argument('--format',choices=['fewshot','compact'],required=True)
    ap.add_argument('--phase',choices=['baseline','screen','controls'],required=True);a=ap.parse_args()
    for c in ['qwen4b','qwen25']:
        assert not json.loads((V2/'results'/c/'competence.json').read_text())['pass_check'],'System-only candidate passed; contingency is not authorized by its protocol'
    key=a.candidate+'-'+a.format
    order=['qwen4b-fewshot','qwen25-fewshot','qwen4b-compact','qwen25-compact']
    for earlier in order[:order.index(key)]:
        assert not json.loads((HERE/'results'/earlier/'competence.json').read_text())['pass_check'],'Earlier candidate passed; stop calibration'
    paths=[HERE/'PROTOCOL.md',HERE/'study_formats.py',HERE/'analyze_formats.py',HERE/'development-with-source-controls.jsonl',
        V2/'PROTOCOL.md',V2/'study.py',V2/'analyze.py',V2/'AMENDMENT-DENSE-LOADER.md',V2/'dependency-manifest.json',
        V2/f'model-manifest-{a.candidate}.json',HERE.parent/'tasks.py',HERE.parent/'run.py',HERE.parent/'prompt_formats.py']
    freeze=dict(candidate=a.candidate,format=a.format,stage='development',client_date='2026-10-01',files={str(p.relative_to(HERE.parent)):sha(p) for p in paths})
    target=HERE/f'freeze-{key}.json'
    if target.exists():assert json.loads(target.read_text())==freeze,'Frozen calibration source changed'
    else:dump(target,freeze)
    core.previous.RESULTS=HERE/'results'/key
    core.prompt=core.previous.prompt=compact_prompt if a.format=='compact' else original_prompt
    rows=[json.loads(s) for s in (HERE/'development-with-source-controls.jsonl').read_text().splitlines()]
    if a.phase!='baseline':
        assert json.loads((core.previous.RESULTS/'competence.json').read_text())['pass_check']
        assert json.loads((core.previous.RESULTS/'instrumentation.json').read_text())['pass_check']
        chosen=dict(candidate=a.candidate,format=a.format,source_freeze_sha256=sha(target))
        selection=HERE/'selected-configuration.json'
        if selection.exists():assert json.loads(selection.read_text())==chosen
        else:dump(selection,chosen)
    engine=core.Engine(a.candidate,a.phase)
    engine.format_name='fewshot' if a.format=='fewshot' else 'system'
    rp=core.previous.RESULTS/f'runtime-{a.phase}.json';runtime=json.loads(rp.read_text())
    runtime.update(calibration_format=a.format,examples=EXAMPLES if a.format=='fewshot' else [],data_sha256=sha(HERE/'development-with-source-controls.jsonl'))
    dump(rp,runtime)
    if a.phase=='baseline':
        assert core.instrumentation(engine,rows),'Instrumentation failed'
        core.previous.baselines(engine,rows)
    elif a.phase=='screen':core.screen(engine,rows)
    else:
        core.source_baselines(engine,rows);core.controls(engine,rows)
    dump(core.previous.RESULTS/f'completed-{a.phase}.json',dict(completed=True,candidate=a.candidate,format=a.format,
        phase=a.phase,reserve_evaluated=False,max_cuda_allocated=core.torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
