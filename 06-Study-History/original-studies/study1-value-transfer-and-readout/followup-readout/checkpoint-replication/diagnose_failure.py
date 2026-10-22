"""Preserve the exact failed identity replay; this is not a relaxed replication."""
import json
from run_checkpoint import HERE,LocalEngine,Journal,dump

def main():
    status=json.loads((HERE/'results/complete.json').read_text())
    assert not status['mechanism_evaluated']
    checks=json.loads((HERE/'results/instrumentation.json').read_text())['checks']
    failed=[r['id'] for r in checks if r.get('replay_identity') is False]
    engine=LocalEngine();base=Journal(HERE/'results/baselines.jsonl');details=[]
    for rid in failed:
        r=base.rows[f'{rid}/recipient_full'];prefix=engine.cuts(r)['pre_z'];source=engine.capture(prefix,[26])[26]
        replay=engine.generate(r['prompt_ids'],prefix)
        same=engine.generate(r['prompt_ids'],prefix,26,source)
        details.append({'id':rid,'baseline':r,'unpatched_replay':replay,'same_example_replay':same,'reproduces_original_failure':replay['output_ids']!=r['output_ids'],'replay_and_same_example_equal':replay['output_ids']==same['output_ids'],'final_answer_changed':replay['parsed']['answer']!=r['parsed']['answer'],'scope':'reconstruction of a known identity failure, not new mechanism evidence'})
    dump(HERE/'results/identity-failure-detail.json',details)
    print(json.dumps([{'id':r['id'],'original_final_text':r['baseline']['text'].split('</think>')[-1],'replay_final_text':r['unpatched_replay']['text'].split('</think>')[-1],'failure_reproduced':r['reproduces_original_failure'],'same_example_matches_replay':r['replay_and_same_example_equal'],'final_answer_changed':r['final_answer_changed']} for r in details],indent=2))

if __name__=='__main__':main()
