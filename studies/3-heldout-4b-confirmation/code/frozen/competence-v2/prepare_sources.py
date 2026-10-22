"""Add a same-leading-digit donor contrast without changing any original task."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from tasks import task,predictions,calc
rows=[json.loads(s) for s in (HERE.parent/'data/development.jsonl').read_text().splitlines()]
for row in rows:
    d=row['donor'];r=row['recipient'];z=d['z']+1 if d['z']%10<9 else d['z']-1
    variant=task(z,d['first'],d['b'],d['remaining'],d['k'])
    assert z!=d['z'] and str(z)[0]==str(d['z'])[0]
    assert predictions(variant,r)['counterfactual']!=row['predictions']['counterfactual']
    row['controls']['same_leading_different_value']=variant
    splice=int(str(d['z'])[0]+str(r['z'])[1:])
    row['digit_alternatives']={
        'intermediate_leading_digit_splice':calc(r['remaining'],splice,r['k']),
        'donor_answer_leading_digit_splice':int(str(d['y'])[0]+str(r['y'])[1:])}
path=HERE/'development-with-source-controls.jsonl'
text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
if path.exists():assert path.read_text()==text
else:path.write_text(text)
print('36 original pairs retained; one fixed same-leading-digit donor added per group.')
