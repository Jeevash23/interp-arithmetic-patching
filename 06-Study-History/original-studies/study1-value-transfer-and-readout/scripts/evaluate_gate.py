"""Compute declared numerical development gates; interpretation is a separate judgment."""
import json
from analyze import ROOT,read

def main():
    baseline=json.loads((ROOT/'results/v3/development/baselines-summary-n100.json').read_text())
    raw=read(ROOT/'results/v3/development/controls-n100/controls.jsonl')
    groups={c:[r for r in raw if r['condition']==c] for c in ['donor','unpatched','same_example','random_source','mismatched_z','isotropic_delta']}
    if any(len(rr)!=100 for rr in groups.values()):raise RuntimeError('All 100 pairs are required in every gate condition')
    counts={c:sum(r['success'] for r in rr) for c,rr in groups.items()}
    margins={c:(counts['donor']-counts[c])/100 for c in groups if c!='donor'}
    record={'stage':'exploratory English development','n':100,'site':{'layer':26,'timing':'pre_z','strength':1,'rule':'full block output replacement at one fixed historical position'},'competence_pass':all(baseline[c]['rate']>=.8 for c in ['donor_full','recipient_full','textual_counterfactual']),'counts':counts,'minimum_counterfactual_count':20,'primary_rate_pass':counts['donor']>=20,'minimum_margin':.1,'paired_margins':margins,'margins_pass':all(v>=.1-1e-12 for v in margins.values()),'interpretation_gate':'requires separate review of crossed controls, number readout and text clamping; numerical gates alone do not authorize a mechanistic claim','validation_evaluated':False}
    record['numerical_gate_pass']=record['competence_pass'] and record['primary_rate_pass'] and record['margins_pass']
    (ROOT/'results/v3/analysis/development-gate.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
