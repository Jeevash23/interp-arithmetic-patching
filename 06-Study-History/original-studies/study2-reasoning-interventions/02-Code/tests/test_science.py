from ai04.tasks import parse,generate_groups,op

def test_parser():
 assert parse('<think>Final: 9</think>\nFinal: 10<|im_end|>')['final']==10
 assert parse('<think>x</think>\\boxed{56}')['final']==56
 assert parse('<think>x</think>Final: 9\n\\boxed{56}')['final'] is None
 assert parse('<think>Final: 9')['final'] is None
 assert parse('<think>x</think>Final: 12cats')['final'] is None
 assert parse('<think>x</think>Intermediate: 6\nFinal: -12')['intermediate']==6

def test_data_and_leakage():
 allrows=[]
 for s,n,z in [('audit',4,940001),('pilot',8,940021),('development',24,940031),('final',80,940041)]:
  r=generate_groups(s,n,z);assert r==generate_groups(s,n,z)
  for x in r:
   for p in [x['donor'],x['recipient']]:assert op(p['a'],p['first'],p['b'])==p['z']
   h=x['hypotheses'];assert len({h['recipient'],h['donor_answer'],h['counterfactual']})==3
  allrows+=r
 for k in ['recipient_prompt','donor_prompt']:assert len({r[k] for r in allrows})==len(allrows)
 assert set(r['template'] for r in allrows if r['split']=='development').isdisjoint(set(r['template'] for r in allrows if r['split']=='final'))

def test_denominators_and_grouping():
 import importlib.util
 from pathlib import Path
 p=Path(__file__).parents[1]/'analyze.py';spec=importlib.util.spec_from_file_location('ai04_analysis',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 # Four nested branches from a single program are one independent unit.
 v=m.interval([.25,.75],B=2000)
 assert v['mean']==.5 and v['n_groups']==2
 assert 0<=v['lo']<=.5<=v['hi']<=1

def test_parser_rejects_multiple_and_thinking_only():
 assert parse('<think>\\boxed{4}</think>Final: 7\nFinal: 8')['final'] is None
 assert parse('<think>Final: 9</think>Intermediate: 9')['final'] is None
 assert parse('<think>Final: 9</think>Final: 12.5')['final'] is None

def test_paired_stratified_calculation():
 import importlib.util,sys
 from pathlib import Path
 p=Path(__file__).parents[1];sys.path.insert(0,str(p))
 from final_analysis import stratified_interval
 a=stratified_interval([1,-1,.5,-.5],[0,0,1,1],B=1000)
 assert a['mean']==0 and a['n_groups']==4 and a['lo']<0<a['hi']
 b=stratified_interval([0]*4,[0,0,1,1],B=100)
 assert b['mean']==b['lo']==b['hi']==0 and b['degenerate']

def test_semantic_agreement_requires_valid_natural_answers():
 from final_analysis_v2 import valid_answer_agreement
 def row(value,stop='eos'):
  return dict(available=True,stop_reason=stop,parsed=dict(final=value))
 assert not valid_answer_agreement(row(None),row(None))
 assert not valid_answer_agreement(row(7,'max_tokens'),row(7))
 assert valid_answer_agreement(row(7),row(7))
 assert not valid_answer_agreement(row(7),row(8))
