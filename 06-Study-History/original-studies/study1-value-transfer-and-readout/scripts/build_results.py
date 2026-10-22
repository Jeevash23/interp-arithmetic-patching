"""Build the review report from completed measurements and a separately written decision."""
import json
import csv
from analyze import ROOT

def js(rel):return json.loads((ROOT/rel).read_text())
def csvrows(rel):return list(csv.DictReader((ROOT/rel).open()))
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    baseline=js('results/v3/development/baselines-summary-n100.json');screen=js('results/v3/development/screen-summary.json')['table']
    controls=csvrows('results/v3/analysis/controls.csv');clamp=js('results/v3/development/clamp-mediation-summary.json');text=js('results/v3/development/textual-scratchpad-summary.json')
    languages=js('results/v3/multilingual-behavioral/summary.json');extended=js('results/v3/multilingual-budget1024/summary.json');gate=js('results/v3/analysis/development-gate.json');audit=js('results/v3/analysis/artifact-audit.json')
    decision=(ROOT/'docs/DECISION.md').read_text();assert audit['pass']
    lines=['# Compositional value-transfer experiment — measured results','',
           'Qwen3-0.6B · local 6 GB NVIDIA GPU · 9 September 2026 · exploratory development study','',decision,'',
           '## Behavioral competence','',
           'All counts below use the complete 100-pair English development set. The first 20 pairs supplied the initial instrumentation and discovery sample. Full-task failures remain included; the prespecified triple-competence subset contains 96 pairs.','',
           mdtable(['Baseline','Correct','Parser failures','Truncations'],[[k.replace('_',' '),f'{v["correct"]}/{v["n"]}',v['parser_failures'],v['truncations']] for k,v in baseline.items()]),'',
           'The earlier non-thinking configurations achieved only 6/20 donor and 6/20 recipient full-task accuracy (v1), and 2/20 and 5/20 (v2). Their failures and source snapshots remain saved. These are configuration comparisons, not an isolated effect of thinking mode.','',
           '## Bounded intervention screen','',
           'The complete 24-site search used 20 pairs (four donor-value clusters). The maximum was 6/20 at block 26 before the labelled Intermediate numeral. Both original arithmetic answers were already present in all 40 donor/recipient thinking traces.','',
           '![Complete screening counts](results/v3/analysis/screen.png)','',
           '[Every screening count and failure](results/v3/analysis/screen.csv). Equation timings were available for 16/20 pairs; unavailable cases remain unsuccessful in the complete denominator. An early equation location is not proof of first hidden computation.','',
           '## Development controls at the selected site','',
           mdtable(['Condition','Original counterfactual','Unchanged answer','Copied donor answer','Copied donor z','Invalid incl. unavailable','Unavailable','Truncated'],[[r['condition'].replace('_',' '),f'{r["counterfactual"]}/{r["n"]}',r['recipient_answer'],r['donor_answer'],r['donor_z'],r['invalid'],r['unavailable'],r['truncations']] for r in controls]),'',
           '![Development controls](results/v3/analysis/controls.png)','',
           'For crossed/random sources, [the full table](results/v3/analysis/controls.csv) also scores the counterfactual using the actual source value. Donor-answer-as-intermediate predictions, all overlapping alternatives, distractor predictions and generated parser failures are separate fields. Unavailable positions and parser failures are not conflated in that table.','',
           '## Printed-number diagnostics','',
           '![Value and digit comparators](results/v3/analysis/value-vs-digit.png)','',
           'The original text-prefix assay replaces the intermediate numeral and allows fresh continuation immediately. New digits can extend that numeral. The line-clamp assay includes the original line ending, then generates the final answer freshly, with or without the same pre-z activation write.','',
           mdtable(['Completed-line clamp condition','Counterfactual','Unchanged','Invalid','Truncated'],[[r['condition'].replace('_',' '),f'{r["counterfactual"]}/{r["n"]}',r['recipient_answer'],r['invalid'],r['truncations']] for r in clamp]),'',
           '[Paired text/activation comparison](results/v3/analysis/text-activation-comparison.json), [all numeral-prefix outcomes](results/v3/development/textual-scratchpad-summary.json), and [value-length/operation diagnostics](results/v3/analysis/mechanism-diagnostics.csv). These are exploratory controlled continuations, not natural direct/indirect-effect estimates.','',
           '## Spanish and Tamil competence supplement','',
           'The same 20 underlying development pairs were used in all languages. English outputs were reused. Spanish/Tamil prompts are [redacted]-reviewed drafts without independent fluent verification; output labels and numeric answers are shared, and thinking language is unconstrained.','',
           mdtable(['Baseline','English','Spanish','Tamil'],[[c.replace('_',' ')]+[f'{next(r["correct"] for r in languages if r["condition"]==c and r["language"]==lang)}/20' for lang in ['en','es','ta']] for c in baseline]),'',
           '![Matched language competence](results/v3/analysis/multilingual-baselines.png)','',
           '[Language failures, truncations and token lengths](results/v3/analysis/multilingual-baselines.csv). This small draft-translation comparison does not identify a language-family effect or a cross-language mechanism barrier.','',
           'A separate 1024-token diagnostic extended all 70 original truncations, preserving their generated prefixes and leaving EOS-completed runs unchanged. Native and resumed/batched continuation agreed on a development example used for instrumentation validation. English and Spanish had no original truncations, so their outcomes were reused. The Tamil comparison is:','',
           mdtable(['Tamil baseline','Correct at 512','Correct at 1024','Truncated at 512','Truncated at 1024'],[[c.replace('_',' '),f'{next(r["correct"] for r in languages if r["language"]=="ta" and r["condition"]==c)}/20',f'{next(r["correct"] for r in extended if r["language"]=="ta" and r["condition"]==c)}/20',next(r['truncations'] for r in languages if r['language']=='ta' and r['condition']==c),next(r['truncations'] for r in extended if r['language']=='ta' and r['condition']==c)] for c in baseline]),'',
           '[Paired budget diagnostic](results/v3/analysis/language-budget-comparison.csv). This is an adapted behavioral supplement, not a new independent sample or a cross-language activation experiment.','',
           '## Uncertainty, validation and audit','',
           'The full development set reuses 20 donor values, 20 recipient values, 12 arithmetic structures and two fixed templates. [Paired analysis](results/v3/analysis/paired-analysis.json) retains complete and prespecified competence-filtered samples. Separate cluster bootstraps and their envelope are sensitivity descriptions; they are not exact multiway or selection-adjusted confidence intervals.','',
           '[Pilot-informed sample-size simulations](results/v3/analysis/sample-size-planning.json) evaluate a ten-percentage-point target under explicit dependence assumptions. These are statistical design simulations, not measured model outcomes or a finalized confirmatory sample size.','',
           mdtable(['Set','Planned pairs','Evaluated pairs','Status'],[['English discovery',20,20,'exploratory'],['English full development',100,100,'includes discovery'],['Reserved English validation',100,0,'not evaluated; see decision above'],['Spanish/Tamil competence',20,20,'matched development pairs; no activation transfer claims']]),'',
           f'The declared numerical English progression gate is **{"passed" if gate["numerical_gate_pass"] else "not passed"}**. [Exact gate calculation](results/v3/analysis/development-gate.json). Interpretation requires the controls in addition to this numerical threshold.','',
           '[Artifact audit](results/v3/analysis/artifact-audit.json) passed. It checks split/model hashes, unique raw records, parser and causal scoring, token bookkeeping, identity continuations and stored activation hashes. [Measurement boundaries](docs/MEASUREMENT-BOUNDARIES.md) specify what each check establishes.','',
           '## Files for continuing the research','',
           '- [Living manuscript](paper/manuscript.md) and [prioritized next experiments](docs/NEXT-STEPS.md).',
           '- [Protocol](docs/PROTOCOL.md), amendments 001–009, [research log](docs/RESEARCH-LOG.md).',
           '- [Primary-source comparison](docs/RELATED-WORK.md), [environment](docs/ENVIRONMENT.md), [reproduction instructions](docs/REPRODUCIBILITY.md).',
           '- Deterministic datasets under `data/`; complete outputs, failures and vectors under `results/`; commands/errors under `logs/`.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));print(ROOT/'RESULTS.md')
if __name__=='__main__':main()
