#!/usr/bin/env python3
"""Recount the figures of the two original study reports from their saved journals (make verify / make analysis).

Standard library only; nothing here runs a model. The journals are read in place from the two original study
folders, which this project calls Study 1 and Study 2:

  Study 1  06-Study-History/original-studies/study1-value-transfer-and-readout   (Qwen3-0.6B readout experiment;
           Qwen3-4B answer-only experiments: screen, fresh generation, rank-8 projection, components, diagnostic)
  Study 2  06-Study-History/original-studies/study2-reasoning-interventions      (Qwen3-0.6B held-out experiment)

Each of the 50 checks compares a recount with the number stated in the original reports (the Study 1 draft of
2 October 2026 and the Study 2 report of 21 September 2026). Outputs go to paper/generated/reconciliation/:
numbers.json (every recounted quantity with its journal), one Markdown table per original table, and
RECONCILIATION.md. With --check the outputs are recomputed and compared with the saved files without writing.
Exit status is non-zero if any check fails or (with --check) any saved output differs.
"""
import collections
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORIG = ROOT / '06-Study-History/original-studies'
R2 = ORIG / 'study1-value-transfer-and-readout'           # Study 1
HV = R2 / 'hidden-value-transfer'
R4 = ORIG / 'study2-reasoning-interventions/04-Results'   # Study 2 results
D4 = ORIG / 'study2-reasoning-interventions/03-Data'      # Study 2 programs
OUT = ROOT / 'paper/generated/reconciliation'
CHECK_ONLY = '--check' in sys.argv[1:]

NUMBERS, CHECKS, TABLES = {}, [], {}


def jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def jload(path):
    with open(path) as handle:
        return json.load(handle)


def rel(path):
    return str(pathlib.Path(path).relative_to(ROOT))


def record(key, value, source):
    NUMBERS[key] = {'value': value, 'source': source if isinstance(source, str) else rel(source)}
    return value


def check(label, recount, expected, source):
    CHECKS.append({'claim': label, 'recount': recount, 'expected': expected,
                   'pass': recount == expected, 'source': source})


def table(name, header, rows, note=''):
    TABLES[name] = {'header': header, 'rows': rows, 'note': note}


def apply(op, x, y):
    return {'add': x + y, 'sub': x - y, 'mul': x * y, '+': x + y, '-': x - y, '*': x * y}[op]


# ---------------------------------------------------------------------------
# AI-02 qualification
# ---------------------------------------------------------------------------
def competence_counts(path):
    counts = collections.defaultdict(lambda: [0, 0])
    for r in jsonl(path):
        c = counts[r['condition']]
        c[1] += 1
        c[0] += bool(r.get('correct'))
    return {k: tuple(v) for k, v in counts.items()}


def triple_competent(path):
    byid = collections.defaultdict(dict)
    for r in jsonl(path):
        byid[r['id']][r['condition']] = bool(r.get('correct'))
    ok = sum(all(v.get(c) for c in ('donor_full', 'recipient_full', 'textual_counterfactual')) for v in byid.values())
    return ok, len(byid)


QUAL = [
    ('Qwen3-4B-FP8, calculator instruction only', HV / 'competence-v2/results/qwen4b/baselines.jsonl'),
    ('Qwen2.5-1.5B, calculator instruction only', HV / 'competence-v2/results/qwen25/baselines.jsonl'),
    ('Qwen3-4B-FP8, instruction plus three demonstrations', HV / 'competence-v3/results/qwen4b-fewshot/baselines.jsonl'),
]
rows = []
for label, path in QUAL:
    c = competence_counts(path)
    t, n = triple_competent(path)
    rows.append([label, '%d/%d' % c['donor_full'], '%d/%d' % c['recipient_full'],
                 '%d/%d' % c['textual_counterfactual'], '%d/%d' % (t, n)])
    record('ai02.qualification.' + label, {'donor_full': c['donor_full'], 'recipient_full': c['recipient_full'],
                                            'textual_counterfactual': c['textual_counterfactual'], 'triple': (t, n)}, path)
table('T1-ai02-qualification',
      ['Development configuration', 'Donor full', 'Recipient full', 'Supplied donor intermediate', 'Triple-competent pairs'],
      rows, 'Recomputed from the three baseline journals; every record stays in the denominator.')
c4 = competence_counts(QUAL[2][1])
c4b = competence_counts(QUAL[0][1])
c25 = competence_counts(QUAL[1][1])
check('AI-02 selected configuration donor full 33/36, recipient full 36/36',
      (c4['donor_full'], c4['recipient_full']), ((33, 36), (36, 36)), 'competence-v3 baselines.jsonl')
check('AI-02 calculator-only 4B donor 30/36, recipient 24/36',
      (c4b['donor_full'], c4b['recipient_full']), ((30, 36), (24, 36)), 'competence-v2 qwen4b baselines.jsonl')
check('AI-02 Qwen2.5-1.5B donor 6/36, recipient 4/36',
      (c25['donor_full'], c25['recipient_full']), ((6, 36), (4, 36)), 'competence-v2 qwen25 baselines.jsonl')
c17 = competence_counts(HV / 'results/baselines.jsonl')
record('ai02.qwen17b.initial', c17, HV / 'results/baselines.jsonl')
check('AI-02 Qwen3-1.7B initial donor two-step 0/36', c17.get('donor_full'), (0, 36), 'hidden-value-transfer/results/baselines.jsonl')

# ---------------------------------------------------------------------------
# AI-02 screen: 36 blocks x 4 roles
# ---------------------------------------------------------------------------
screen_path = HV / 'competence-v3/results/qwen4b-fewshot/screen.jsonl'
agg = collections.defaultdict(list)
for r in jsonl(screen_path):
    agg[(r['layer'], r['role'])].append((r['delta_log_ratio'], r['counterfactual']['log_probability'], r['group']))
site_rows = []
for (layer, role), vals in sorted(agg.items()):
    bygroup = collections.defaultdict(list)
    for dlr, lp, g in vals:
        bygroup[g].append(dlr)
    mean_dlr = statistics.mean(statistics.mean(v) for v in bygroup.values())
    mean_p = statistics.mean(math.exp(lp) for _, lp, _ in vals)
    above_half = sum(math.exp(lp) > 0.5 for _, lp, _ in vals)
    site_rows.append({'layer': layer, 'role': role, 'mean_delta_log_ratio': mean_dlr,
                      'mean_cf_probability': mean_p, 'cf_probability_above_half': above_half, 'n': len(vals)})
record('ai02.screen.records', sum(len(v) for v in agg.values()), screen_path)
check('AI-02 screen has 5,184 records', NUMBERS['ai02.screen.records']['value'], 5184, rel(screen_path))
ratio_top = sorted(site_rows, key=lambda s: (-s['mean_delta_log_ratio'], s['layer']))[:3]
record('ai02.screen.ratio_top3', [(s['layer'], s['role'], round(s['mean_delta_log_ratio'], 3)) for s in ratio_top], screen_path)
check('AI-02 ratio-selected sites are blocks 20, 9, 10 at the first operand',
      [(s['layer'], s['role']) for s in ratio_top], [(20, 'a_end'), (9, 'a_end'), (10, 'a_end')], rel(screen_path))
check('AI-02 ratio-selected mean changes 30.405, 30.159, 30.159 nats',
      [round(s['mean_delta_log_ratio'], 3) for s in ratio_top], [30.405, 30.159, 30.159], rel(screen_path))
eligible = {}
for role in ('b_end', 'first_stage_end', 'prompt_end'):
    best = max((s for s in site_rows if s['role'] == role), key=lambda s: s['mean_cf_probability'])
    eligible[role] = (best['layer'], best['mean_cf_probability'])
record('ai02.screen.eligible_sites', eligible, screen_path)
check('AI-02 eligible-site revision chooses blocks 14, 20, 23',
      [eligible[r][0] for r in ('b_end', 'first_stage_end', 'prompt_end')], [14, 20, 23], rel(screen_path))
ROLE_NAMES = {'a_end': 'first operand', 'b_end': 'second operand', 'first_stage_end': 'first-stage end', 'prompt_end': 'answer prefix'}
prof = collections.defaultdict(dict)
for s in site_rows:
    prof[s['layer']][s['role']] = s
rows = []
for layer in sorted(prof):
    row = [str(layer)]
    for role in ('a_end', 'b_end', 'first_stage_end', 'prompt_end'):
        s = prof[layer][role]
        row.append('%.1f / %.1e' % (s['mean_delta_log_ratio'], s['mean_cf_probability']))
    rows.append(row)
table('T2-ai02-layer-position-screen',
      ['Block'] + ['%s: log-ratio change (nats) / P(complete counterfactual)' % ROLE_NAMES[r] for r in ('a_end', 'b_end', 'first_stage_end', 'prompt_end')],
      rows, 'Recomputed from screen.jsonl (36 blocks x 4 roles x 36 pairs). Group-weighted ratio means; plain means of the complete-counterfactual probability.')
record('ai02.screen.max_cf_probability_blocks_0_to_11', max(s['mean_cf_probability'] for s in site_rows if s['layer'] <= 11), screen_path)
record('ai02.screen.max_cf_probability_any', max(s['mean_cf_probability'] for s in site_rows), screen_path)
record('ai02.screen.prompt_end_first_block_with_ratio_change_over_5', min(s['layer'] for s in site_rows if s['role'] == 'prompt_end' and s['mean_delta_log_ratio'] > 5), screen_path)

# ---------------------------------------------------------------------------
# AI-02 fresh-generation controls (whole-vector and native components)
# ---------------------------------------------------------------------------
def first_operand_only(rec):
    s, r = rec['source_task'], rec['recipient_task']
    return apply(r['remaining'], apply(r['first'], s['a'], r['b']), r['k'])


def control_site_summary(path, layer, role):
    recs = [r for r in jsonl(path) if r['layer'] == layer and r['role'] == role]
    out = {}
    bypair = collections.defaultdict(dict)
    for r in recs:
        bypair[r['id']][r['condition']] = r
    for cond in sorted({r['condition'] for r in recs}):
        rs = [r for r in recs if r['condition'] == cond]
        cf_key = 'actual_source_success' if cond in ('different_z_same_answer', 'same_leading_different_value', 'random_source') else 'counterfactual_success'
        out[cond] = {
            'n': len(rs),
            'counterfactual': sum(bool(r[cf_key]) for r in rs),
            'unchanged': sum(r['answer'] == r['predictions']['recipient_answer'] for r in rs),
            'copied_donor_answer': sum(r['answer'] == r['predictions']['donor_answer'] for r in rs),
            'first_operand_only': sum(r['answer'] == first_operand_only(r) for r in rs),
            'invalid': sum(r['answer'] is None or bool(r.get('truncated')) for r in rs),
        }
    joint = sum(all(bypair[p].get(c, {}).get('counterfactual_success') for c in ('donor', 'same_z_decomposition', 'same_z_different_answer')) for p in bypair)
    lead_pairs = [p for p in bypair if 'same_leading_different_value' in bypair[p]]
    same_lead_identical = sum(bypair[p]['donor']['answer'] == bypair[p]['same_leading_different_value']['answer'] for p in lead_pairs)
    ops = collections.Counter(r['recipient_operation'] for r in recs if r['condition'] == 'donor' and r['counterfactual_success'])
    return out, {'joint_original_and_both_same_z': joint, 'same_leading_identical_outputs': same_lead_identical,
                 'pairs': len(bypair), 'donor_success_by_operation': dict(ops)}


WV_SITES = [
    (HV / 'competence-v3/results/qwen4b-fewshot/controls.jsonl', 20, 'a_end', 'Block 20, first operand'),
    (HV / 'competence-v3/results/qwen4b-fewshot/controls.jsonl', 9, 'a_end', 'Block 9, first operand'),
    (HV / 'competence-v3/results/qwen4b-fewshot/controls.jsonl', 10, 'a_end', 'Block 10, first operand'),
    (HV / 'competence-v3/results/eligible-sites/controls.jsonl', 14, 'b_end', 'Block 14, second operand'),
    (HV / 'competence-v3/results/eligible-sites/controls.jsonl', 20, 'first_stage_end', 'Block 20, first-stage end'),
    (HV / 'competence-v3/results/eligible-sites/controls.jsonl', 23, 'prompt_end', 'Block 23, answer prefix'),
]
OUTCOME_ROWS = [
    ('Original donor: complete counterfactual', 'donor', 'counterfactual'),
    ('Same z, different decomposition: complete counterfactual', 'same_z_decomposition', 'counterfactual'),
    ('Same z, different donor answer: complete counterfactual', 'same_z_different_answer', 'counterfactual'),
    ("Different z, same donor answer: that source's counterfactual", 'different_z_same_answer', 'counterfactual'),
    ("Same leading digit, different z: that source's counterfactual", 'same_leading_different_value', 'counterfactual'),
    ('Matched noise: original counterfactual', 'isotropic_delta', 'counterfactual'),
    ('First-digit textual control: original counterfactual', 'text_first_answer_digit', 'counterfactual'),
    ('Original donor: first-operand-only prediction', 'donor', 'first_operand_only'),
    ('Original donor: copied donor final answer', 'donor', 'copied_donor_answer'),
    ('Original donor: unchanged recipient answer', 'donor', 'unchanged'),
    ('Matched noise: unchanged recipient answer', 'isotropic_delta', 'unchanged'),
]
wv = {}
for path, layer, role, label in WV_SITES:
    wv[label] = control_site_summary(path, layer, role)
    record('ai02.fresh.' + label, wv[label], path)
labels = [l for *_, l in WV_SITES]
rows = []
for text, cond, field in OUTCOME_ROWS:
    rows.append([text] + ['%d/%d' % (wv[l][0][cond][field], wv[l][0][cond]['n']) for l in labels])
rows.append(['Joint original and both same-z successes'] + ['%d/%d' % (wv[l][1]['joint_original_and_both_same_z'], wv[l][1]['pairs']) for l in labels])
rows.append(['Original and same-leading-digit sources give identical outputs'] + ['%d/%d' % (wv[l][1]['same_leading_identical_outputs'], wv[l][1]['pairs']) for l in labels])
table('T3-ai02-whole-vector-fresh-generation', ['Fresh-generation outcome'] + labels, rows,
      'Recomputed from controls.jsonl of competence-v3 (ratio-selected sites) and eligible-sites (revised sites). 36 development pairs per cell.')
ratio_labels = labels[:3]
rev_labels = labels[3:]
check('AI-02 ratio sites: 0/36 complete counterfactuals at blocks 20, 9, 10',
      [wv[l][0]['donor']['counterfactual'] for l in ratio_labels], [0, 0, 0], 'competence-v3 controls.jsonl')
check('AI-02 ratio sites: first-operand-only outputs 16, 19, 19',
      [wv[l][0]['donor']['first_operand_only'] for l in ratio_labels], [16, 19, 19], 'competence-v3 controls.jsonl')
check('AI-02 ratio sites: first-digit text control 1/36 at each site',
      [wv[l][0]['text_first_answer_digit']['counterfactual'] for l in ratio_labels], [1, 1, 1], 'competence-v3 controls.jsonl')
check('AI-02 noise unchanged 35, 35, 36 at ratio sites',
      [wv[l][0]['isotropic_delta']['unchanged'] for l in ratio_labels], [35, 35, 36], 'competence-v3 controls.jsonl')
check('AI-02 revised sites: complete counterfactuals 1, 0, 2',
      [wv[l][0]['donor']['counterfactual'] for l in rev_labels], [1, 0, 2], 'eligible-sites controls.jsonl')
check('AI-02 revised sites: same-leading-digit sources identical on 33, 35, 26 pairs',
      [wv[l][1]['same_leading_identical_outputs'] for l in rev_labels], [33, 35, 26], 'eligible-sites controls.jsonl')
check('AI-02 revised sites: unchanged recipient 3, 35, 5',
      [wv[l][0]['donor']['unchanged'] for l in rev_labels], [3, 35, 5], 'eligible-sites controls.jsonl')
check('AI-02 block 14 different-z/same-answer source counterfactual 4/36',
      wv['Block 14, second operand'][0]['different_z_same_answer']['counterfactual'], 4, 'eligible-sites controls.jsonl')
check('AI-02 joint original-and-both-same-z successes are zero at all six whole-vector sites',
      [wv[l][1]['joint_original_and_both_same_z'] for l in labels], [0] * 6, 'competence-v3 and eligible-sites controls.jsonl')

comp = {}
for name in ('attention', 'mlp'):
    path = HV / ('component-v1/results/%s/controls.jsonl' % name)
    comp[name] = control_site_summary(path, 23, 'prompt_end')
    record('ai02.component.' + name, comp[name], path)
rows = []
for text, cond, field in OUTCOME_ROWS[:6] + OUTCOME_ROWS[9:10]:
    rows.append([text] + ['%d/%d' % (comp[n][0][cond][field], comp[n][0][cond]['n']) for n in ('attention', 'mlp')])
rows.append(['Original donor: invalid output'] + ['%d/36' % comp[n][0]['donor']['invalid'] for n in ('attention', 'mlp')])
rows.append(['Joint original and both same-z successes'] + ['%d/36' % comp[n][1]['joint_original_and_both_same_z'] for n in ('attention', 'mlp')])
table('T6-ai02-native-components', ['Fresh-generation outcome', 'Attention contribution', 'Feed-forward contribution'], rows,
      'Block 23 / answer prefix. Recomputed from component-v1 controls.jsonl.')
check('AI-02 components: 1/36 original counterfactual each',
      [comp[n][0]['donor']['counterfactual'] for n in ('attention', 'mlp')], [1, 1], 'component-v1 controls.jsonl')
check('AI-02 components: unchanged 10 and 15',
      [comp[n][0]['donor']['unchanged'] for n in ('attention', 'mlp')], [10, 15], 'component-v1 controls.jsonl')
check('AI-02 components: successes only under multiplication',
      [comp[n][1]['donor_success_by_operation'] for n in ('attention', 'mlp')], [{'mul': 1}, {'mul': 1}], 'component-v1 controls.jsonl')

# ---------------------------------------------------------------------------
# AI-02 learned subspace and qualified-source diagnostic
# ---------------------------------------------------------------------------
sub_path = HV / 'subspace-v1/results/controls.jsonl'
sub = collections.defaultdict(list)
for r in jsonl(sub_path):
    sub[r['condition']].append(r)
sub_summary = {}
for cond, rs in sub.items():
    sub_summary[cond] = {'n': len(rs),
                         'original_counterfactual': sum(bool(r['counterfactual_success']) for r in rs),
                         'actual_source_counterfactual': sum(bool(r['actual_source_success']) for r in rs),
                         'unchanged': sum(r['answer'] == r['comparison_reference']['recipient_answer'] for r in rs),
                         'invalid': sum(r['answer'] is None or bool(r.get('truncated')) for r in rs)}
record('ai02.subspace.conditions', sub_summary, sub_path)
pairs = collections.defaultdict(dict)
for cond, rs in sub.items():
    for r in rs:
        pairs[r['id']][cond] = r
sub_joint = sum(all(pairs[p][c]['counterfactual_success'] for c in ('donor', 'same_z_decomposition', 'same_z_different_answer')) for p in pairs)
sub_lead_identical = sum(pairs[p]['donor']['answer'] == pairs[p]['same_leading_different_value']['answer'] for p in pairs)
sub_ops = collections.Counter(r['recipient_operation'] for r in sub['donor'] if r['counterfactual_success'])
record('ai02.subspace.joint_original_and_both_same_z', sub_joint, sub_path)
record('ai02.subspace.same_leading_identical_outputs', sub_lead_identical, sub_path)
record('ai02.subspace.donor_success_by_operation', dict(sub_ops), sub_path)
SUB_ORDER = ('donor', 'same_z_decomposition', 'same_z_different_answer', 'different_z_same_answer', 'same_leading_different_value',
             'isotropic_delta', 'random_subspace', 'label_shuffled', 'text_first_answer_digit', 'text_actual_first_token', 'whole_vector')
rows = [[c, '%d/36' % sub_summary[c]['original_counterfactual'], '%d/36' % sub_summary[c]['actual_source_counterfactual'], '%d/36' % sub_summary[c]['unchanged']] for c in SUB_ORDER]
table('T4-ai02-learned-subspace', ['Condition (rank-8 projection, block 23 / answer prefix)', 'Original counterfactual', "Source's own counterfactual", 'Unchanged recipient answer'],
      rows, 'Recomputed from subspace-v1 controls.jsonl. Text and whole-vector rows reuse outputs measured earlier at the same site.')
check('AI-02 subspace: 1/36 original counterfactual, 23/36 unchanged',
      (sub_summary['donor']['original_counterfactual'], sub_summary['donor']['unchanged']), (1, 23), rel(sub_path))
check('AI-02 subspace: noise 0 (35 unchanged), random subspace 0 (34 unchanged), shuffled labels 0 (6 unchanged)',
      [(sub_summary[c]['original_counterfactual'], sub_summary[c]['unchanged']) for c in ('isotropic_delta', 'random_subspace', 'label_shuffled')],
      [(0, 35), (0, 34), (0, 6)], rel(sub_path))
check('AI-02 subspace: same-z sources 2/36 and 1/36 with one joint success',
      (sub_summary['same_z_decomposition']['original_counterfactual'], sub_summary['same_z_different_answer']['original_counterfactual'], sub_joint), (2, 1, 1), rel(sub_path))
check('AI-02 subspace: same-leading-digit sources identical on 32/36', sub_lead_identical, 32, rel(sub_path))
check('AI-02 subspace: success only under multiplication', dict(sub_ops), {'mul': 1}, rel(sub_path))
tb = HV / 'subspace-v1/results/training-baselines.jsonl'
tc = collections.defaultdict(lambda: [0, 0])
for r in jsonl(tb):
    tc[r['kind']][1] += 1
    tc[r['kind']][0] += bool(r.get('correct'))
record('ai02.subspace.fitting_baselines', {k: tuple(v) for k, v in tc.items()}, tb)
check('AI-02 subspace fitting set: first step 756/756, full problem 677/756',
      (tuple(tc['first']), tuple(tc['full'])), ((756, 756), (677, 756)), rel(tb))

q_path = HV / 'subspace-v1/value-digit-qualified-sources/results/controls.jsonl'
q = collections.defaultdict(list)
for r in jsonl(q_path):
    q[r['condition']].append(r)
q_summary = {}
for cond, rs in q.items():
    disc = [r for r in rs if r['discriminative']]
    q_summary[cond] = {'n': len(rs), 'full': sum(bool(r['full_success']) for r in rs), 'coarse': sum(bool(r['coarse_success']) for r in rs),
                       'full_discriminative': sum(bool(r['full_success']) for r in disc), 'coarse_discriminative': sum(bool(r['coarse_success']) for r in disc),
                       'n_discriminative': len(disc), 'unchanged': sum(r['answer'] == r['alternatives']['recipient_answer'] for r in rs)}
record('ai02.qualified_sources.conditions', q_summary, q_path)
qpairs = collections.defaultdict(dict)
for cond, rs in q.items():
    for r in rs:
        qpairs[r['id']][cond] = r
q_joint_full = sum(bool(qpairs[p]['donor']['full_success'] and qpairs[p]['same_value_coupled']['full_success']) for p in qpairs)
q_joint_coarse = sum(bool(qpairs[p]['donor']['coarse_success'] and qpairs[p]['same_value_coupled']['coarse_success']) for p in qpairs)
record('ai02.qualified_sources.joint', {'full': q_joint_full, 'coarse': q_joint_coarse, 'pairs': len(qpairs)}, q_path)
coarse_ops = collections.Counter(r['recipient']['remaining'] for r in q['donor'] if r['coarse_success'])
record('ai02.qualified_sources.primary_coarse_by_operation', dict(coarse_ops), q_path)
cells = collections.defaultdict(list)
for r in q['donor']:
    cells[(r['donor']['z'] // 10, r['recipient']['z'], r['recipient']['remaining'])].append(r)
unit = {'cells': len(cells), 'all_full': sum(all(r['full_success'] for r in rs) for rs in cells.values()),
        'all_coarse': sum(all(r['coarse_success'] for r in rs) for rs in cells.values()),
        'identical_outputs': sum(len({r['answer'] for r in rs}) == 1 for rs in cells.values())}
record('ai02.qualified_sources.unit_triplets', unit, q_path)
cols = ('donor', 'same_value_coupled', 'isotropic_delta', 'random_subspace', 'text_actual_first_token')
rows = [['Complete intermediate-value target, all pairs'] + ['%d/81' % q_summary[c]['full'] for c in cols],
        ['Leading-digit target, all pairs'] + ['%d/81' % q_summary[c]['coarse'] for c in cols],
        ['Complete target, 72 discriminating pairs'] + ['%d/72' % q_summary[c]['full_discriminative'] for c in cols],
        ['Leading-digit target, 72 discriminating pairs'] + ['%d/72' % q_summary[c]['coarse_discriminative'] for c in cols],
        ['Unchanged recipient answer'] + ['%d/81' % q_summary[c]['unchanged'] for c in cols]]
table('T5-ai02-qualified-source-diagnostic', ['Qualified-source outcome', 'Primary source', 'Coupled same-z source', 'Noise', 'Random subspace', 'Actual-first-token text'],
      rows, 'Recomputed from value-digit-qualified-sources controls.jsonl (81 pairs, 72 of them discriminating).')
check('AI-02 qualified sources: 0/72 full-value successes for both sources',
      (q_summary['donor']['full_discriminative'], q_summary['same_value_coupled']['full_discriminative']), (0, 0), rel(q_path))
check('AI-02 qualified sources: coarse 3/81 and 7/81; text 12/81',
      (q_summary['donor']['coarse'], q_summary['same_value_coupled']['coarse'], q_summary['text_actual_first_token']['coarse']), (3, 7, 12), rel(q_path))
check('AI-02 qualified sources: no joint full or coarse success across sources', (q_joint_full, q_joint_coarse), (0, 0), rel(q_path))
check('AI-02 qualified sources: 27 cells, none all-full, one all-coarse, 19 identical',
      (unit['cells'], unit['all_full'], unit['all_coarse'], unit['identical_outputs']), (27, 0, 1, 19), rel(q_path))
check('AI-02 qualified sources: primary coarse matches all under addition', dict(coarse_ops), {'add': 3}, rel(q_path))

# ---------------------------------------------------------------------------
# AI-02 readout study (September, Qwen3-0.6B native thinking)
# ---------------------------------------------------------------------------
ro = collections.defaultdict(dict)
for seed in ('seed-20262909', 'seed-20263909'):
    for r in jsonl(R2 / ('followup-readout/results/%s/interventions.jsonl' % seed)):
        ro[(r['id'], r['seed'])][r['condition']] = r
act_success = sum(bool(v['activation']['success']) for v in ro.values())
full_avail = [v for v in ro.values() if v['text_full_value']['status'] == 'generated']
full_success = sum(bool(v['text_full_value']['success']) for v in full_avail)
digit_pairs = [v for v in ro.values() if v['text_first_digit']['status'] == 'generated']
digit_exact = sum(v['activation']['output_ids'] == v['text_first_digit']['output_ids'] for v in digit_pairs)
source_copy = sum(v['activation']['fresh_output_ids'][:1] == [v['activation']['source_next_token']] for v in ro.values())
width = collections.defaultdict(dict)
for (pid, seed), v in ro.items():
    a = v['activation']
    width[(a['group'], a['family'], seed)][a['width']] = tuple(a['fresh_output_ids'])
width_pairs = [w for w in width.values() if len(w) >= 2]
width_exact = sum(len(set(w.values())) == 1 for w in width_pairs)
record('ai02.readout', {'planned': len(ro), 'activation_success': act_success, 'full_value_text_success': full_success,
                        'full_value_text_available': len(full_avail), 'first_digit_exact_sequence': (digit_exact, len(digit_pairs)),
                        'source_first_token_copy': (source_copy, len(ro)), 'width_invariance': (width_exact, len(width_pairs))}, R2 / 'followup-readout/results')
check('Readout: activation 16/240 versus full-value text 151/240', (act_success, full_success), (16, 151), 'followup-readout interventions.jsonl')
check('Readout: first-digit text reproduces activation output 233/238', (digit_exact, len(digit_pairs)), (233, 238), 'followup-readout interventions.jsonl')
check('Readout: first emitted token matches donor 240/240', (source_copy, len(ro)), (240, 240), 'followup-readout interventions.jsonl')
check('Readout: paired short and long donors identical 117/120', (width_exact, len(width_pairs)), (117, 120), 'followup-readout interventions.jsonl')
table('T10-ai02-readout', ['Readout study measurement (Qwen3-0.6B, block 26, before the labelled intermediate)', 'Count'],
      [['Activation write produces the complete counterfactual', '%d/240' % act_success],
       ['Full-value text produces the complete counterfactual', '%d/240' % full_success],
       ['First emitted token equals the donor first digit', '%d/%d' % (source_copy, len(ro))],
       ['First-digit text reproduces the activation output token for token', '%d/%d' % (digit_exact, len(digit_pairs))],
       ['Short and long donors sharing the first digit give identical output', '%d/%d' % (width_exact, len(width_pairs))]],
      'Recomputed from both seed journals (30 program groups x 4 pairs x 2 seeds = 240 planned records).')

# ---------------------------------------------------------------------------
# AI-04 held-out comparison, controls, development, donor semantics
# ---------------------------------------------------------------------------
programs = {p['id']: p for split in ('development', 'final', 'pilot') for p in jload(D4 / ('generated/%s.json' % split))}
main = list(jsonl(R4 / 'final/qwen06-main-v1/records.jsonl'))
base = collections.defaultdict(collections.Counter)
for r in main:
    if r['kind'] != 'baseline':
        continue
    prog = programs[r['group']]
    z_expected = prog['recipient']['z'] if r['role'] == 'recipient' else prog['donor']['z']
    b = base[r['role']]
    b['programs'] += 1
    b['natural_eos'] += r['stop_reason'] == 'eos'
    b['correct_final'] += bool(r['correct'])
    inter_ok = r['parsed'].get('intermediate') == z_expected
    b['correct_intermediate'] += inter_ok
    b['both'] += bool(r['correct']) and inter_ok
record('ai04.main.baselines', {k: dict(v) for k, v in base.items()}, R4 / 'final/qwen06-main-v1/records.jsonl')
table('T7a-ai04-heldout-baselines', ['Baseline role', 'Programs', 'Natural EOS', 'Correct final', 'Correct intermediate label', 'Both stages correct'],
      [[role, base[role]['programs'], base[role]['natural_eos'], base[role]['correct_final'], base[role]['correct_intermediate'], base[role]['both']] for role in ('donor', 'recipient', 'supplied')],
      'Recomputed from the main journal; the intermediate oracle is the donor z for the donor and supplied roles and the recipient z for the recipient role.')
check('AI-04 held-out baselines donor 59/64, recipient 61/64, supplied 57/64',
      [base[r]['correct_final'] for r in ('donor', 'recipient', 'supplied')], [59, 61, 57], 'qwen06-main-v1 records.jsonl')
check('AI-04 held-out natural EOS 60, 63, 61', [base[r]['natural_eos'] for r in ('donor', 'recipient', 'supplied')], [60, 63, 61], 'qwen06-main-v1 records.jsonl')
check('AI-04 held-out correct intermediate labels 56, 60, 58',
      [base[r]['correct_intermediate'] for r in ('donor', 'recipient', 'supplied')], [56, 60, 58], 'qwen06-main-v1 records.jsonl')


def outcome_counts(recs):
    c = collections.Counter({k: 0 for k in ('branches', 'unavailable', 'target', 'invalid', 'exhausted', 'recipient', 'other_valid')})
    for r in recs:
        o = r['outcome']
        c['branches'] += 1
        if not r['available']:
            c['unavailable'] += 1
        elif o['counterfactual']:
            c['target'] += 1
        elif not o['completed']:
            c['exhausted'] += 1
        elif not o['valid']:
            c['invalid'] += 1
        elif o['category'] == 'recipient':
            c['recipient'] += 1
        else:
            c['other_valid'] += 1
    return c


sites = collections.defaultdict(list)
for r in main:
    if r['kind'] == 'intervention' and r['condition'] == 'activation':
        sites[r['site']].append(r)
main_summary = {}
for site, rs in sites.items():
    c = outcome_counts(rs)
    competent = [r for r in rs if r['competent']]
    main_summary[site] = dict(c, groups_with_any_success=len({r['group'] for r in rs if r['outcome']['counterfactual']}),
                              competent_branches=len(competent), competent_target=sum(bool(r['outcome']['counterfactual']) for r in competent))
record('ai04.main.sites', main_summary, R4 / 'final/qwen06-main-v1/records.jsonl')
SITES = ('label_z-L20', 'label_z-L26')
rows = [[('replay' if s == 'label_z-L20' else 'continuation'), s, '%d/%d' % (main_summary[s]['target'], main_summary[s]['branches']),
         '%d/64' % main_summary[s]['groups_with_any_success'], main_summary[s]['recipient'], main_summary[s]['other_valid'],
         '%d/%d/%d' % (main_summary[s]['invalid'], main_summary[s]['exhausted'], main_summary[s]['unavailable']),
         '%d/%d' % (main_summary[s]['competent_target'], main_summary[s]['competent_branches'])] for s in SITES]
table('T7b-ai04-heldout-sites', ['Selection policy', 'Frozen site', 'Target answers / branches', 'Programs with any success', 'Recipient oracle', 'Other valid number', 'Invalid / exhausted / unavailable', 'Target on triple-competent subset'],
      rows, 'Recomputed from the main journal (64 held-out programs x 2 streams per site).')
check('AI-04 held-out: 0/128 at label_z-L20 and label_z-L26', [main_summary[s]['target'] for s in SITES], [0, 0], 'qwen06-main-v1 records.jsonl')
check('AI-04 held-out: recipient answers 113/128 and 91/128; other numbers 4 and 26',
      [(main_summary[s]['recipient'], main_summary[s]['other_valid']) for s in SITES], [(113, 4), (91, 26)], 'qwen06-main-v1 records.jsonl')
check('AI-04 held-out: 8 unavailable, 2 exhausted, 1 invalid per site',
      [(main_summary[s]['unavailable'], main_summary[s]['exhausted'], main_summary[s]['invalid']) for s in SITES], [(8, 2, 1), (8, 2, 1)], 'qwen06-main-v1 records.jsonl')
check('AI-04 held-out: competent subset 0/104 at both sites',
      [(main_summary[s]['competent_target'], main_summary[s]['competent_branches']) for s in SITES], [(0, 104), (0, 104)], 'qwen06-main-v1 records.jsonl')
lead = collections.Counter()
for r in sites['label_z-L26']:
    if not r['available']:
        continue
    lead['available'] += 1
    inter = r['parsed'].get('intermediate')
    donor_z = programs[r['group']]['donor']['z']
    if inter is not None:
        lead['first_numeral_parsed'] += 1
        lead['starts_with_donor_digit'] += str(inter)[0] == str(donor_z)[0]
        lead['complete_donor_numeral'] += inter == donor_z
record('ai04.main.block26_leading_digit', dict(lead), R4 / 'final/qwen06-main-v1/records.jsonl')

ctrl = list(jsonl(R4 / 'final/qwen06-controls-v1/records.jsonl'))
cc = collections.defaultdict(list)
for r in ctrl:
    if r['kind'] == 'intervention':
        cc[(r['site'], r['condition'])].append(r)
ctrl_summary = {'%s/%s' % k: dict(outcome_counts(v)) for k, v in cc.items()}
record('ai04.controls', ctrl_summary, R4 / 'final/qwen06-controls-v1/records.jsonl')
rows = []
for site in ('first_z-L26', 'label_z-L26'):
    for cond in ('activation', 'random_0', 'random_1', 'random_2', 'text_digit', 'text_full'):
        c = ctrl_summary['%s/%s' % (site, cond)]
        rows.append(['%s/%s' % (site, cond), '%d/%d' % (c.get('target', 0), c['branches']), c.get('recipient', 0), c.get('other_valid', 0),
                     '%d/%d/%d' % (c.get('invalid', 0), c.get('exhausted', 0), c.get('unavailable', 0))])
table('T8-ai04-heldout-controls', ['Boundary / condition', 'Target', 'Recipient oracle', 'Other valid', 'Invalid / exhausted / unavailable'],
      rows, 'Recomputed from the controls journal (16 programs x 2 streams per condition).')
check('AI-04 controls: full-value text 5/32 early and 3/32 late; activation 0/32 at both',
      [ctrl_summary[k].get('target', 0) for k in ('first_z-L26/text_full', 'label_z-L26/text_full', 'first_z-L26/activation', 'label_z-L26/activation')], [5, 3, 0, 0], 'qwen06-controls-v1 records.jsonl')
check('AI-04 controls: full-value text exhausts 11/32 early; activation 7/32',
      [ctrl_summary['first_z-L26/text_full'].get('exhausted', 0), ctrl_summary['first_z-L26/activation'].get('exhausted', 0)], [11, 7], 'qwen06-controls-v1 records.jsonl')
check('AI-04 controls: random directions give 0 targets and 29 or 30 recipient answers',
      sorted({(ctrl_summary['%s/random_%d' % (s, i)].get('target', 0), ctrl_summary['%s/random_%d' % (s, i)].get('recipient', 0)) for s in ('first_z-L26', 'label_z-L26') for i in range(3)}),
      [(0, 29), (0, 30)], 'qwen06-controls-v1 records.jsonl')
dp = {}
for site in ('first_z-L26', 'label_z-L26'):
    act = {(r['group'], r['stream']): r for r in cc[(site, 'activation')]}
    dig = {(r['group'], r['stream']): r for r in cc[(site, 'text_digit')]}
    keys = [k for k in act if act[k]['available'] and dig[k]['available']]
    dp[site] = {'runnable_pairs': len(keys), 'exact_sequence': sum(act[k]['output_ids'] == dig[k]['output_ids'] for k in keys)}
record('ai04.controls.digit_pairs', dp, R4 / 'final/qwen06-controls-v1/records.jsonl')
check('AI-04 controls: first-digit text reproduces activation 30/30 early and 30/32 late',
      [(dp[s]['exact_sequence'], dp[s]['runnable_pairs']) for s in ('first_z-L26', 'label_z-L26')], [(30, 30), (30, 32)], 'qwen06-controls-v1 records.jsonl')

dev = list(jsonl(R4 / 'development/qwen06-development-v2/records.jsonl'))
ds = collections.defaultdict(list)
for r in dev:
    if r['kind'] == 'intervention' and r['condition'] == 'activation':
        ds[r['site']].append(r)
DEV_SITES = ('first_z-L13', 'first_z-L20', 'first_z-L26', 'label_z-L13', 'label_z-L20', 'label_z-L26')
dev_rows, dev_summary = [], {}
for site in DEV_SITES:
    rs = ds[site]
    a = [r['A_delta'] for r in rs if r.get('A_delta') is not None]
    b = [r['B_delta'] for r in rs if r.get('B_delta') is not None]
    c = sum(bool(r['outcome']['counterfactual']) for r in rs)
    dev_summary[site] = {'mean_A': statistics.mean(a), 'mean_B': statistics.mean(b), 'C': (c, len(rs))}
    dev_rows.append([site, '%.4f' % statistics.mean(a), '%.6f' % statistics.mean(b), '%d/%d' % (c, len(rs))])
record('ai04.development.sites', dev_summary, R4 / 'development/qwen06-development-v2/records.jsonl')
table('T9a-ai04-development', ['Site', 'Mean A change (logits)', 'Mean B change (nats)', 'C target / attempts'],
      dev_rows, 'Recomputed from the development-v2 journal (24 programs x 2 streams per site). A and B means use records where the assay was available.')
check('AI-04 development C counts 0, 2, 2, 0, 0, 4', [dev_summary[s]['C'][0] for s in DEV_SITES], [0, 2, 2, 0, 0, 4], 'qwen06-development-v2 records.jsonl')
check('AI-04 development mean A changes 0.0442 ... 28.0946',
      [round(dev_summary[s]['mean_A'], 4) for s in DEV_SITES], [0.0442, 23.5377, 28.2053, 0.1344, 12.677, 28.0946], 'qwen06-development-v2 records.jsonl')

don = list(jsonl(R4 / 'development/qwen06-donor-controls-v1/records.jsonl'))
dn = collections.defaultdict(list)
src_ok = {}
for r in don:
    if r['kind'] == 'intervention':
        dn[(r['variant'], r['boundary'])].append(r)
    elif r['kind'] == 'source_baseline':
        src_ok[(r['group'], r['variant'])] = r['parsed'].get('final') == r['task']['y']
# original activation sequences at the same boundary/layer come from the development-v2 journal
orig_seq = {}
for r in dev:
    if r['kind'] == 'intervention' and r['condition'] == 'activation' and r['layer'] == 26 and r['stream'] == 0:
        orig_seq[(r['group'], r['boundary'])] = r['output_ids']
donor_rows, donor_summary = [], {}
for (variant, boundary), rs in sorted(dn.items()):
    own = sum(bool(r['own_cf']) for r in rs)
    src_correct = sum(src_ok.get((r['group'], variant), False) for r in rs)
    exact = sum(r['output_ids'] == orig_seq.get((r['group'], boundary)) for r in rs)
    donor_summary['%s/%s' % (variant, boundary)] = {'branches': len(rs), 'source_correct': src_correct, 'own_target': own, 'exact_original_sequence': exact}
    donor_rows.append(['%s/%s' % (variant, boundary), src_correct, own, exact])
record('ai04.donor_semantics', donor_summary, R4 / 'development/qwen06-donor-controls-v1/records.jsonl')
table('T9b-ai04-donor-semantics', ['Source alternative / boundary', 'Source correct (of 8)', "Source's own target (of 8)", 'Exact original activation sequence (of 8)'],
      donor_rows, 'Recomputed from the donor-controls journal; the original sequence is the stream-0 block-26 activation run of the same program in development-v2.')
check('AI-04 donor semantics: same-leading wide sources produce none of their own targets and reproduce 8/8 original sequences',
      [(donor_summary['same_leading_wide/%s' % b]['own_target'], donor_summary['same_leading_wide/%s' % b]['exact_original_sequence']) for b in ('first_z', 'label_z')],
      [(0, 8), (0, 8)], 'qwen06-donor-controls-v1 records.jsonl')
check('AI-04 donor semantics: different-z/same-answer sources reproduce 3/8 early and 4/8 late sequences',
      [donor_summary['different_z_same_answer/%s' % b]['exact_original_sequence'] for b in ('first_z', 'label_z')], [3, 4], 'qwen06-donor-controls-v1 records.jsonl')

# ---------------------------------------------------------------------------
# Write (or compare) outputs
# ---------------------------------------------------------------------------
files = {'numbers.json': json.dumps(NUMBERS, indent=1, default=str)}
for name, t in TABLES.items():
    lines = ['| ' + ' | '.join(str(h) for h in t['header']) + ' |', '|' + '---|' * len(t['header'])]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in t['rows']]
    if t['note']:
        lines += ['', t['note']]
    files[name + '.md'] = '\n'.join(lines) + '\n'
passed = sum(c['pass'] for c in CHECKS)
lines = ['# Reconciliation of recounted numbers against the two original reports', '',
         '%d of %d checks pass. Each row recounts a figure from the journals in the original study folders (Study 1: '
         '`study1-value-transfer-and-readout`, Study 2: `study2-reasoning-interventions`, both under '
         '`06-Study-History/original-studies/`) and compares it with the number stated in the Study 1 draft '
         '(2 October 2026) or the Study 2 report (21 September 2026).' % (passed, len(CHECKS)),
         '', '| Claim | Recount | Stated | Result | Journal |', '|---|---|---|---|---|']
for c in CHECKS:
    lines.append('| %s | %s | %s | %s | %s |' % (c['claim'], c['recount'], c['expected'], 'pass' if c['pass'] else 'FAIL', c['source']))
files['RECONCILIATION.md'] = '\n'.join(lines) + '\n'
stale = []
if CHECK_ONLY:
    for name, text in files.items():
        path = OUT / name
        if not path.exists() or path.read_text() != text:
            stale.append(name)
    print('%d/%d checks pass; %d saved output(s) differ from the recount%s' % (passed, len(CHECKS), len(stale), (': ' + ', '.join(stale)) if stale else ''))
else:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text)
    print('%d/%d checks pass; %d tables and %d numbers written to %s' % (passed, len(CHECKS), len(TABLES), len(NUMBERS), rel(OUT)))
for c in CHECKS:
    if not c['pass']:
        print('  FAIL:', c['claim'], '| recount =', c['recount'], '| stated =', c['expected'])
sys.exit(0 if passed == len(CHECKS) and not stale else 1)
