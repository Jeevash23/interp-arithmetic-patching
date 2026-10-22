"""Display every fitted-intervention control with its relevant source target."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE.parent/'competence-v2/.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=HERE/'results/analysis';s=json.loads((root/'summary.json').read_text())
items=[('donor','Learned value projection'),('same_z_decomposition','Same value, new decomposition'),
       ('same_z_different_answer','Same value, new donor answer'),('different_z_same_answer','New value, same donor answer'),
       ('same_leading_different_value','New value, same leading digit'),('isotropic_delta','Norm-matched noise'),
       ('random_subspace','Rank- and norm-matched random projection'),('label_shuffled','Projection fitted to shuffled labels'),
       ('random_source','Unrelated source'),('whole_vector','Whole-vector write at the same site'),
       ('text_first_answer_digit','First digit of correct donor answer'),('text_actual_first_token','Actual first donor token'),
       ('unpatched','Unpatched recipient'),('same_example','Recipient self projection')]
values=[s['conditions'][c]['actual_source_counterfactual' if c in ['different_z_same_answer','same_leading_different_value','random_source'] else 'original_counterfactual'] for c,l in items]
colors=['#376a93' if c in ['donor','same_z_decomposition','same_z_different_answer','different_z_same_answer','same_leading_different_value'] else '#9ca3af' for c,l in items]
fig,ax=plt.subplots(figsize=(10,7.4),layout='constrained')
ax.barh(range(len(items)),values,color=colors,height=.65)
ax.set_yticks(range(len(items)),[l for c,l in items]);ax.invert_yaxis()
ax.set_xlim(0,36);ax.set_xticks([0,6,12,18,24,30,36]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
for i,v in enumerate(values):ax.text(v+.25,i,f'{v}/36',va='center',fontsize=10)
ax.set_xlabel('Complete counterfactual generations for the relevant source')
ax.set_title('Fixed rank-eight value projection: full development sample\nBlock 23 / answer prefix · 12 donor groups · no held-out evaluation',loc='left',pad=12)
fig.savefig(root/'subspace-controls.png',dpi=180);fig.savefig(root/'subspace-controls.svg');plt.close(fig)
print(root/'subspace-controls.png')
