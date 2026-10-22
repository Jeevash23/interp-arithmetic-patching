"""Plot complete-answer counts, including both declared digit controls."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE/'results/qwen4b-fewshot';OUT=ROOT/'analysis'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    report=json.loads((OUT/'summary.json').read_text())
    extra=json.loads((OUT/'actual-donor-digit-summary.json').read_text())
    lookup={(s['layer'],s['role']):s for s in extra['sites']}
    names=['donor','same_z_decomposition','same_z_different_answer','isotropic_delta','text_first_answer_digit','actual_digit','unpatched','same_example']
    labels=['Donor activation','Same value, new decomposition','Same value, new donor answer','Matched noise','Digit of correct donor answer','Actual first donor token','Unpatched','Recipient self write']
    fig,axes=plt.subplots(1,3,figsize=(14,5.6),sharex=True,sharey=True,layout='constrained')
    for ax,site in zip(axes,report['sites']):
        s=site['site'];ex=lookup[s['layer'],s['role']]
        values=[ex['actual_digit_text_correct'] if name=='actual_digit' else site['conditions'][name]['counterfactual'] for name in names]
        ax.barh(range(len(names)),values,color=['#16697a']*3+['#9ca3af']*5,height=.65)
        ax.set_title(f"Block {s['layer']} · {s['role'].replace('_',' ')}",fontsize=11)
        ax.set_yticks(range(len(names)),labels,fontsize=10);ax.invert_yaxis();ax.set_xlim(0,41)
        ax.set_xticks([0,9,18,27,36]);ax.set_xlabel('Correct complete counterfactuals / 36')
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        for y,value in enumerate(values):ax.text(value+.6,y,str(value),va='center',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fresh generation after the selected development interventions\nQwen3-4B-Instruct-2507-FP8 · three examples · 12 donor groups',fontsize=13)
    fig.savefig(OUT/'fresh-generation-controls.png',dpi=180)
    fig.savefig(OUT/'fresh-generation-controls.svg')
    plt.close(fig)
    print(OUT/'fresh-generation-controls.png')


if __name__=='__main__':main()
