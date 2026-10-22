"""Paired exploratory competence analysis; language-family effects are not identified."""
import json
import numpy as np
from analyze import ROOT,read,write_csv,clustered_interval,plt
from value_transfer.tasks import load_data

def main():
    source=ROOT/'results/v3/multilingual-behavioral';out=ROOT/'results/v3/analysis'
    raw=read(source/'baselines.jsonl');table=json.loads((source/'summary.json').read_text())
    data={r['id']:r for r in load_data('development')};paired=[]
    conditions=['donor_full','recipient_full','donor_first','recipient_first','donor_remaining','recipient_remaining','textual_counterfactual']
    for lang in ['es','ta']:
        for c in conditions:
            en={r['id']:r for r in raw if r['condition']==c and r['language']=='en'};other={r['id']:r for r in raw if r['condition']==c and r['language']==lang};ids=sorted(en.keys()&other.keys())
            delta=[int(other[i]['correct'])-int(en[i]['correct']) for i in ids]
            paired.append({'language_minus_english':lang,'condition':c,'n':len(ids),'paired_difference':float(np.mean(delta)),'donor_value_cluster_sensitivity':clustered_interval(delta,[data[i]['donor']['z'] for i in ids]),'recipient_value_cluster_sensitivity':clustered_interval(delta,[data[i]['recipient']['z'] for i in ids]),'note':'Only four donor-value clusters and two fixed templates. Degenerate bootstrap intervals at zero are not evidence of equivalence.'})
    write_csv(out/'multilingual-baselines.csv',table);(out/'multilingual-paired.json').write_text(json.dumps(paired,indent=2))
    matrix=np.array([[next(r['percent'] for r in table if r['condition']==c and r['language']==lang) for lang in ['en','es','ta']] for c in conditions])
    labels=['Donor full task','Recipient full task','Donor first step','Recipient first step','Donor remaining operation','Recipient remaining operation','Supplied donor z continuation']
    fig,ax=plt.subplots(figsize=(7.5,4.8),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(3),['English','Spanish','Tamil']);ax.set_yticks(range(7),labels)
    for i in range(7):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]/5:.0f}/20',ha='center',va='center',color='white' if matrix[i,j]>60 else 'black')
    fig.colorbar(im,ax=ax,label='Accuracy (%)');ax.set_title('Exploratory behavior — 512-token ceiling\nSpanish/Tamil translations await independent review',fontsize=11)
    fig.savefig(out/'multilingual-baselines.png',dpi=180);fig.savefig(out/'multilingual-baselines.svg');plt.close(fig)
    extended_path=ROOT/'results/v3/multilingual-budget1024/summary.json'
    if extended_path.exists():
        extended=json.loads(extended_path.read_text());write_csv(out/'multilingual-budget1024.csv',extended)
        comparison=[]
        for r in table:
            e=next(x for x in extended if x['language']==r['language'] and x['condition']==r['condition'])
            comparison.append({'language':r['language'],'condition':r['condition'],'n':r['n'],'correct_512':r['correct'],'correct_1024':e['correct'],'truncated_512':r['truncations'],'truncated_1024':e['truncations'],'newly_correct':e['newly_correct_after_extension']})
        write_csv(out/'language-budget-comparison.csv',comparison)
    print(f'Exploratory language analysis saved to {out}')
if __name__=='__main__':main()
