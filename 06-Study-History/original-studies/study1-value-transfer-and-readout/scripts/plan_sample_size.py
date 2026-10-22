"""Pilot-informed conditional planning simulations; no simulated model results."""
import json
import sys
import numpy as np
from scipy.stats import t
from analyze import ROOT,read,write_csv
sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import load_data

def icc(values,labels):
    groups=[np.asarray([v for v,l in zip(values,labels) if l==u],dtype=float) for u in sorted(set(labels))]
    if len(groups)<2 or len({len(g) for g in groups})!=1:return None
    m=len(groups[0]);x=np.stack(groups);means=x.mean(1)
    between=m*means.var(ddof=1);within=((x-means[:,None])**2).sum()/(len(groups)*(m-1))
    denominator=between+(m-1)*within
    return float((between-within)/denominator) if denominator>0 else None

def simulated_power(q,rho,clusters,delta=.1,reps=5000,seed=20260909):
    # D is the paired success difference, in {-1,0,+1}; E[D]=delta.
    p=np.array([(q+delta)/2,(q-delta)/2,1-q]);rng=np.random.default_rng(seed)
    if rho==0:prob=np.broadcast_to(p,(reps,clusters,3))
    else:prob=rng.dirichlet(p*(1/rho-1),size=(reps,clusters))
    counts=rng.multinomial(5,prob);means=(counts[:,:,0]-counts[:,:,1])/5
    average=means.mean(1);se=means.std(1,ddof=1)/np.sqrt(clusters)
    reject=average-t.ppf(.975,clusters-1)*se>0
    power=float(reject.mean())
    return power,float(np.sqrt(power*(1-power)/reps))

def main():
    raw=read(ROOT/'results/v3/development/controls-n100/controls.jsonl')
    data={r['id']:r for r in load_data('development')}
    groups={c:{r['id']:r for r in raw if r['condition']==c} for c in ['donor','unpatched','random_source','mismatched_z','isotropic_delta']}
    if len(groups['donor'])!=100:raise RuntimeError('Complete 100-pair controls are required for this planning analysis')
    observed=[]
    for c in ['unpatched','random_source','mismatched_z','isotropic_delta']:
        ids=sorted(groups['donor'].keys()&groups[c].keys())
        if len(ids)!=100:raise RuntimeError('Incomplete control')
        d=[int(groups['donor'][i]['success'])-int(groups[c][i]['success']) for i in ids]
        observed.append({'control':c,'n':len(ids),'donor_value_clusters':len({data[i]['donor']['z'] for i in ids}),'paired_mean':float(np.mean(d)),'paired_variance':float(np.var(d,ddof=1)),'discordance_q':float(np.mean(np.abs(d))),'donor_z_icc':icc(d,[data[i]['donor']['z'] for i in ids])})
    q=max(.15,max(r['discordance_q'] for r in observed));rho=max(0,max((r['donor_z_icc'] or 0) for r in observed));rho=min(.8,rho)
    scenarios=[('pilot_q_icc',q,rho),('pilot_q_no_clustering',q,0),('pilot_q_rho_0.2',q,.2),('pilot_q_rho_0.5',q,.5),('high_discordance_rho_0.5',max(q,.5),.5)]
    table=[];selected=[]
    for name,qq,rr in scenarios:
        choice=None
        for clusters in [20,30,40,60,80,120,160,240,320]:
            power,se=simulated_power(qq,rr,clusters)
            row={'scenario':name,'discordance_q':qq,'icc':rr,'clusters':clusters,'pairs':clusters*5,'power_at_10pp':power,'monte_carlo_se':se}
            table.append(row)
            if choice is None and power>=.8:choice=row
        selected.append({'scenario':name,'smallest_tested_design_at_80pct':choice})
    out=ROOT/'results/v3/analysis';out.mkdir(exist_ok=True)
    write_csv(out/'sample-size-simulations.csv',table)
    record={'type':'conditional design simulation, not observed language-model results','smallest_effect':.1,'power_target':.8,'alpha_two_sided':.05,'replicates':5000,'cluster_size':5,'parameter_rule':'Use maximum pilot discordance across four controls, floored at 0.15 to define a feasible 0.10-effect alternative; use maximum donor-value ICC clipped to [0,0.8], then stress-test wider dependence/discordance assumptions.','observed_pilot':observed,'candidate_designs':selected,'limitations':['20 donor-value clusters give uncertain variance and ICC estimates.','Simulation models one dependence axis and a cluster-level t statistic, not all reused structures and templates.','Power concerns one paired comparison; a simultaneous multiple-control claim needs its own multiplicity-aware design.','A broader confirmation must balance recipient operations and use substantially more template families; do not simply duplicate pilot pairs.','These are planning candidates, not an authorized/frozen confirmatory sample size.','Inference about a development-selected maximum remains selection biased.']}
    (out/'sample-size-planning.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
