import random,json
from ai04.common import ROOT,writej,digest
rng=random.Random(949001);rows=[]
for i in range(8):
 vals=rng.sample(range(31,90),6);zd,zr,unused=vals[:3];yr,yd,ycf=vals[3:]
 # Two lookup compositions. State IDs are labels, not quantities to add.
 dstart=101+i;rstart=201+i
 dr=dict(z=zd,y=yd);rr=dict(z=zr,y=yr)
 dtable=f'{dstart} -> {zd}; {rstart} -> {unused}'
 rtable=f'{rstart} -> {zr}; {dstart} -> {unused}'
 dsecond=f'{zd} -> {yd}; {zr} -> {ycf}; {unused} -> {yr}'
 rsecond=f'{zd} -> {ycf}; {zr} -> {yr}; {unused} -> {yd}'
 fmt='\nThese numbers are state labels. Follow the lookup arrows, without arithmetic. Report exactly two final lines:\nIntermediate: <integer>\nFinal: <integer>'
 dp=f'Apply mapping A, then mapping B. Start at {dstart}. Mapping A: {dtable}. Mapping B: {dsecond}.'+fmt
 rp=f'Apply mapping A, then mapping B. Start at {rstart}. Mapping A: {rtable}. Mapping B: {rsecond}.'+fmt
 sp=f'The intermediate state is supplied as {zd}. Apply only mapping B: {rsecond}.'+fmt
 rows.append(dict(id=f'symbolic-{i:03d}',split='symbolic_qualification',template=i%2,donor=dr,recipient=rr,donor_prompt=dp,recipient_prompt=rp,supplied_prompt=sp,hypotheses=dict(counterfactual=ycf,recipient=yr,donor_answer=yd),collisions={},task_family='two_lookup_composition'))
writej(ROOT/'03-Data/generated/symbolic.json',rows);writej(ROOT/'03-Data/splits/symbolic.json',dict(seed=949001,groups=8,hash=digest(rows),status='unexamined model outcomes'))
