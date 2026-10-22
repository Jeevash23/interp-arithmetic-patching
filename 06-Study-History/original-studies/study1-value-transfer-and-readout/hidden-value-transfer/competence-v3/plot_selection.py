"""Compare the two discovery statistics without implying generated success."""
import json,math,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE.parent/'competence-v2/.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
root=HERE/'results/qwen4b-fewshot';out=root/'analysis'
rows=[json.loads(s) for s in (root/'screen.jsonl').read_text().splitlines()]
roles={'a_end':('First operand','#9a3b37'),'b_end':('Second operand','#376aa6'),
       'first_stage_end':('First-step end','#4e8660'),'prompt_end':('Answer prefix','#875caa')}
fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
for role,(label,color) in roles.items():
    rr=[[r for r in rows if r['role']==role and r['layer']==l] for l in range(36)]
    axes[0].plot(range(36),[sum(r['delta_log_ratio'] for r in x)/36 for x in rr],color=color,label=label,lw=2)
    axes[1].plot(range(36),[sum(math.exp(r['counterfactual']['log_probability']) for r in x)/36 for x in rr],color=color,lw=2)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_xlim(0,35)
axes[0].set_ylabel('Mean log-ratio change (nats)');axes[0].legend(ncol=2,frameon=False,loc='lower left',bbox_to_anchor=(.01,.08))
axes[0].set_title('Discovery statistics answer different questions\n36 development pairs in 12 donor groups',loc='left')
axes[1].set_ylabel('Mean probability of complete\ncounterfactual, including EOS')
axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylim(-.002,.065)
axes[1].set_xlabel('Decoder block output (zero based)')
axes[1].annotate('Block 23 / answer prefix: 5.78%\nFresh generation: 2/36; source controls fail',
                 xy=(23,.05782943916128455),xytext=(2,.050),fontsize=10,
                 arrowprops=dict(arrowstyle='->',color='#555555'))
fig.savefig(out/'selection-metrics.png',dpi=180);fig.savefig(out/'selection-metrics.svg');plt.close(fig)
print(out/'selection-metrics.png')
