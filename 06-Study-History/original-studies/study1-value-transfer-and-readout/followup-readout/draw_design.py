"""Scientific design schematic; values are illustrative, not model results."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig,ax=plt.subplots(figsize=(10.5,3.9),layout='constrained');ax.set_xlim(0,10.5);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',linewidth=1,edgecolor=color,facecolor='white'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#172033')
def arrow(a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526274',lw=1.4))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,fontsize=9,ha='center')
box(.1,2.65,2.15,.8,'Donor A\nz = 2 + 4 = 6','#2563eb')
box(.1,1.2,2.15,.8,'Donor B\nz = 60 + 4 = 64','#2563eb')
box(3.6,2.65,2.9,.8,'Complete-value hypothesis\nDistinct recipient targets','#0f766e')
box(7.8,2.65,2.45,.8,'g(6) = 9\ng(64) = 67','#0f766e')
box(3.6,1.2,2.9,.8,'Next-digit hypothesis\nShared first digit: 6','#d97706')
box(7.8,1.2,2.45,.8,'Fresh continuation\nMay append digits','#d97706')
arrow((2.3,3.05),(3.5,3.05));arrow((2.3,1.6),(3.5,2.8))
arrow((2.3,3.05),(3.5,1.85));arrow((2.3,1.6),(3.5,1.6))
arrow((6.55,3.05),(7.7,3.05));arrow((6.55,1.6),(7.7,1.6))
ax.text(5.25,3.8,'Same recipient and random stream; change only the donor source',ha='center',fontsize=13,weight='bold')
ax.text(5.25,.63,'Recipient: 50 - 3 = 47, then g(z) = z + 3. Whole-value transfer must distinguish 6 from 64.',ha='center',fontsize=11)
ax.text(5.25,.19,'Design illustration. The target answers are arithmetic predictions, not measured model outputs.',ha='center',fontsize=9,color='#526274')
out=HERE/'paper';fig.savefig(out/'paired-design.png',dpi=190);fig.savefig(out/'paired-design.svg');plt.close(fig)
