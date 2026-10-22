from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/"tmp/matplotlib"))
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from ai04.common import ROOT,readj,writej,digest

out=ROOT/'04-Results/figures';out.mkdir(exist_ok=True)
f=readj(ROOT/'04-Results/analysis/final-v2/results.json');s=readj(ROOT/'04-Results/analysis/secondary-v1/secondary.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
blue='#176894';orange='#D36B24';green='#29836B';grey='#79838D'
def save(fig,name):
 fig.savefig(out/(name+'.png'),dpi=230,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
dev=s['runs']['development/qwen06-development-v2']['summary']['conditions'];keys=[b+'-L'+str(l)+'/activation' for b in ['first_z','label_z'] for l in [13,20,26]];labels=['First / 13','First / 20','First / 26','Label / 13','Label / 20','Label / 26']
fig,axs=plt.subplots(1,3,figsize=(10.5,3.9),layout='constrained')
for ax,field,title,unit in zip(axs,['mean_A_delta','mean_B_delta','success'],['A: immediate digit margin','B: replayed answer margin','C: complete target answer'],['Change in logits','Change in log probability (nats)','Fraction of continuation attempts']):
 vals=[dev[k][field]/dev[k]['n'] if field=='success' else dev[k][field] for k in keys]
 ax.bar(np.arange(6),vals,color=[blue]*3+[orange]*3,width=.7);ax.set_xticks(np.arange(6),labels,rotation=65,ha='right');ax.set_title(title,fontsize=11);ax.set_ylabel(unit);ax.axhline(0,color='#444444',lw=.7)
 if field=='success':
  ax.set_ylim(0,1);ax.text(.5,.94,'24 programs; 2 streams per site',transform=ax.transAxes,ha='center',fontsize=10)
  for i,k in enumerate(keys):ax.text(i,vals[i]+.035,f'{dev[k]["success"]}/{dev[k]["n"]}',ha='center',fontsize=10)
save(fig,'01-assay-comparison')

ctrl=f['controls']['conditions'];conds=['activation','text_digit','text_full','random_0','random_1','random_2'];labs=['Activation','First digit','Full value','Random 1','Random 2','Random 3']
fig,axs=plt.subplots(1,2,figsize=(10,3.9),sharey=True,layout='constrained')
for ax,b,title in zip(axs,['first_z','label_z'],['Before first printed intermediate','Before labelled intermediate']):
 rs=[ctrl[b+'-L26/'+c] for c in conds];vals=[r['success']/r['n'] for r in rs];colors=[blue,orange,green,grey,grey,grey];ax.bar(range(6),vals,color=colors);ax.set_xticks(range(6),labs,rotation=40,ha='right');ax.set_ylim(0,1);ax.set_title(title,fontsize=11)
 for i,r in enumerate(rs):
  lo=r['cluster_interval']['lo'];hi=r['cluster_interval']['hi'];ax.errorbar(i,vals[i],yerr=[[vals[i]-lo],[hi-vals[i]]],color='#222222',capsize=3,lw=1);ax.text(i,min(.95,max(hi,vals[i])+.045),f'{r["success"]}/{r["n"]}',ha='center',fontsize=10)
axs[0].set_ylabel('Complete counterfactual answer / all attempts');fig.supxlabel('Fixed block 26; first 16 held-out programs; two streams; descriptive program-bootstrap intervals',fontsize=10)
save(fig,'02-control-comparison')

fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
nodes=[('Changed intermediate','51','Leading digit becomes 5'),('Uses changed value','46','Visible calculation: 51 - 5'),('Recomputes original','91','Returns to 75 + 16'),('Correct final answer','86','Recalculates 91 - 5')]
for i,(title,num,detail) in enumerate(nodes):
 x=.01+i*.25;ax.add_patch(FancyBboxPatch((x,.34),.225,.5,boxstyle='round,pad=0.008',transform=ax.transAxes,fc='#EAF2F6' if i<2 else '#E9F3EE',ec=blue if i<2 else green,lw=1.2));ax.text(x+.1125,.74,title,ha='center',transform=ax.transAxes,fontsize=10,weight='bold');ax.text(x+.1125,.53,num,ha='center',transform=ax.transAxes,fontsize=26,color=blue if i<2 else green);ax.text(x+.1125,.40,detail,ha='center',transform=ax.transAxes,fontsize=10)
 if i<3:ax.annotate('',xy=(x+.247,.59),xytext=(x+.228,.59),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.2,color='#555555'))
ax.text(.5,.18,'One audited pilot trajectory (pilot-000, first_z-L26, stream 0).\nThis shows visible recomputation in this trace; it does not identify a private belief or complete-value transfer.',ha='center',va='center',transform=ax.transAxes,fontsize=10)
save(fig,'03-observed-trajectory')

fig,axs=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained');labels2=[];natural=[];bad=[];missing=[];tokens=[]
for b,short in [('first_z','First'),('label_z','Label')]:
 for c,lab in zip(conds[:3],labs[:3]):
  r=ctrl[b+'-L26/'+c];n=r['n'];cat=r['categories'];labels2.append(short+' / '+lab);missing.append(cat.get('unavailable',0)/n);bad.append(cat.get('noncompletion',0)/n);natural.append(1-bad[-1]-missing[-1]);tokens.append(r['fresh_tokens']/n)
y=np.arange(len(labels2));axs[0].barh(y,natural,label='Natural termination',color=green);axs[0].barh(y,bad,left=natural,label='Budget exhausted',color=orange);axs[0].barh(y,missing,left=np.array(natural)+bad,label='Unavailable',color=grey);axs[0].set_yticks(y,labels2);axs[0].invert_yaxis();axs[0].set_xlim(0,1);axs[0].set_xlabel('Fraction of intended attempts');axs[0].set_title('Termination is separate from correctness');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=10,frameon=False)
axs[1].barh(y,tokens,color=blue);axs[1].set_yticks(y,labels2);axs[1].invert_yaxis();axs[1].set_xlabel('Fresh generated tokens per intended attempt');axs[1].set_title('Continuation work after the same cut')
save(fig,'04-completion-and-cost')

d=s['donor']['conditions'];variants=['same_z_other_answer','same_z_decomposition','different_z_same_answer','same_leading_wide'];vlabs=['Same z, other answer','Same z, other decomposition','Other z, same answer','Same first digit, wider z']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.3),sharex=True,layout='constrained')
for ax,b,title in zip(axs,['first_z','label_z'],['First intermediate boundary','Labelled intermediate boundary']):
 rs=[d[v+'/'+b] for v in variants];y=np.arange(4);ax.barh(y-.16,[r['first_digit_copied']/r['n'] for r in rs],height=.3,color=blue,label='Source first digit');ax.barh(y+.16,[r['own_cf']/r['n'] for r in rs],height=.3,color=green,label='Complete target answer');ax.set_yticks(y,vlabs);ax.invert_yaxis();ax.set_xlim(0,1.05);ax.set_xlabel('Fraction of 8 intended attempts');ax.set_title(title,fontsize=11)
fig.legend(*axs[1].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=10,frameon=False);save(fig,'05-donor-semantics')
writej(out/'manifest.json',dict(inputs={'final':digest((ROOT/'04-Results/analysis/final-v2/results.json').read_bytes()),'secondary':digest((ROOT/'04-Results/analysis/secondary-v1/secondary.json').read_bytes())},figures=[p.name for p in sorted(out.glob('*.png'))],source=digest(Path(__file__).read_bytes())))
print('Saved five figures to',out)
