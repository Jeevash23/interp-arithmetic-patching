import re,random
from .common import ROOT,writej,digest

def op(x,s,k):return {'+':lambda:x+k,'-':lambda:x-k,'*':lambda:x*k}[s]()
def program(z,first,b,second,k):
 a=z-b if first=='+' else z+b
 return dict(a=a,b=b,first=first,z=z,second=second,k=k,y=op(z,second,k))
TEMPLATES=[
 'Execute two steps. First calculate z = {a} {first} {b}. Then calculate y = z {second} {k}.',
 'Start with {a}. Apply {first} {b} to obtain the intermediate z. Apply {second} {k} to z to obtain the final y.',
 'A register starts at {a}. Update it by {first} {b}; call the new value z. Set y to z {second} {k}.',
 'A calculator program assigns z := {a} {first} {b}, followed by y := z {second} {k}. Execute the assignments in order.'
]
FORMAT='\nReport your result using exactly two final lines:\nIntermediate: <integer>\nFinal: <integer>'
def prompt(p,template=0):return TEMPLATES[template].format(**p)+FORMAT
def supplied(p,z):return f'Use the supplied intermediate z = {z}. Compute y = z {p["second"]} {p["k"]}. Do not perform any earlier steps.\nReport exactly two final lines:\nIntermediate: <integer>\nFinal: <integer>'
def parse(text):
 if '</think>' not in text:return dict(final=None,intermediate=None,format='unclosed_thinking')
 tail=text.rsplit('</think>',1)[1]
 norm=tail.replace('**','').replace('<|im_end|>','').replace('<|endoftext|>','')
 vals=[int(x) for x in re.findall(r'^\s*Final:\s*(?:\\\(?\s*)?(-?\d+)\s*(?:\\\)?)?\s*$',norm,re.M)]
 boxes=[int(x) for x in re.findall(r'\\boxed\{\s*(-?\d+)\s*\}',norm)]
 ints=[int(x) for x in re.findall(r'^\s*Intermediate:\s*(-?\d+)\s*$',norm,re.M)]
 candidates=vals+boxes
 if not candidates and re.fullmatch(r'\s*-?\d+\s*',norm):candidates=[int(norm)]
 final=candidates[-1] if candidates and len(set(candidates))==1 else None
 return dict(final=final,intermediate=ints[-1] if ints and len(set(ints))==1 else None,format='label' if vals and final is not None else 'boxed' if boxes and final is not None else 'bare' if final is not None else 'invalid_or_conflict')
def classify(result,g):
 p=result['parsed'];a=p['final']
 hypotheses=g['hypotheses'];matches=[k for k,v in hypotheses.items() if a is not None and a==v]
 cat='noncompletion' if result['stop_reason']!='eos' else 'invalid' if a is None else next((k for k in ['counterfactual','recipient','donor_answer','donor_first_digit','operand_a','operand_b','donor_operation'] if k in matches),'other_valid')
 return dict(category=cat,matches=matches,counterfactual=cat=='counterfactual',valid=a is not None,completed=result['stop_reason']=='eos')
def generate_groups(split,n,seed):
 rng=random.Random(seed);rows=[]
 ranges={'audit':(31,39,71,79),'pilot':(42,59,82,99),'development':(61,79,101,129),'final':(132,169,202,239)}
 dlo,dhi,rlo,rhi=ranges[split]
 seen=set()
 while len(rows)<n:
  i=len(rows);zd=rng.randint(dlo,dhi);zr=rng.randint(rlo,rhi)
  dr=program(zd,'+' if i%2 else '-',rng.randint(3,16),['+','-','*'][i%3],rng.randint(2,8))
  rr=program(zr,'-' if i%2 else '+',rng.randint(3,16),['-','*','+'][i%3],rng.randint(2,8))
  ycf=op(zd,rr['second'],rr['k'])
  if len({dr['y'],rr['y'],ycf})<3 or min(dr['y'],rr['y'],ycf)<=0:continue
  if any(p['z'] in [p['a'],p['b'],p['k'],p['y']] for p in [dr,rr]):continue
  if str(zd)[0]==str(zr)[0]:continue
  key=(tuple(dr.values()),tuple(rr.values()))
  if key in seen:continue
  seen.add(key)
  hs=dict(counterfactual=ycf,recipient=rr['y'],donor_answer=dr['y'],donor_first_digit=op(int(str(zd)[0]),rr['second'],rr['k']),operand_a=op(op(dr['a'],rr['first'],rr['b']),rr['second'],rr['k']),operand_b=op(op(rr['a'],rr['first'],dr['b']),rr['second'],rr['k']),donor_operation=op(zr,dr['second'],dr['k']),complete_expression=ycf)
  template=(2+i%2) if split=='final' else i%2
  rows.append(dict(id=f'{split}-{i:03d}',split=split,template=template,donor=dr,recipient=rr,hypotheses=hs,collisions={k:[q for q,v in hs.items() if v==a and q!=k] for k,a in hs.items()},donor_prompt=prompt(dr,template),recipient_prompt=prompt(rr,template),supplied_prompt=supplied(rr,zd)))
 return rows
if __name__=='__main__':
 for split,n,seed in [('audit',4,940001),('pilot',8,940021),('development',24,940031),('final',80,940041)]:
  rows=generate_groups(split,n,seed)
  writej(ROOT/'03-Data/generated'/f'{split}.json',rows)
  writej(ROOT/'03-Data/splits'/f'{split}.json',dict(split=split,groups=n,seed=seed,ids=[r['id'] for r in rows],hash=digest(rows),status='generated without model outcomes'))
