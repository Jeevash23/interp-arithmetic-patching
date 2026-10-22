#!/usr/bin/env python3
"""Write generated/numbers.tex and the generated tables of the manuscript from saved outputs (make numbers).

Every number in the manuscript is a macro defined here or a cell of a generated table, except the design constants
listed in review/literal-numbers.json. Sources, all saved outputs (nothing here runs a model):

  generated/reconciliation/numbers.json   recounts of the original study reports from the saved journals, written
                                          by analysis/reconcile_reports.py (50 checks against the reports)
  ../analysis/posthoc/results/*.json      the post hoc analyses (analysis/posthoc/*.py)
  saved study outputs read directly       Study 2's timing audits, Study 1's trace-exposure audit, model configs
  ../studies/3-*/results/summary.json     Study 3 (held-out Qwen3-4B confirmation): every number by name, with
                                          its prespecified verdicts; verification.json must report all checks passed
  ../studies/4-*/results/summary.json     Study 4 (multi-digit number tokens): the competence gates of the candidates
  ../studies/4-*/results/candidates.json  Study 4: the status and error profile of each candidate
  ../studies/5-*/results/summary.json     Study 5 (added direction and span writes in Qwen3-4B): every number by name
  ../studies/6-*/results/summary.json     Study 6 (Study 4's design in OLMo-2-7B): its competence gate

Statements in the text that rest on a qualitative property of the data (for example that every first-operand write
changed the answer, or that the three tokenizer files are identical) are asserted here, so a change in the data
stops the build instead of leaving a stale sentence.

Usage: python paper/make_numbers.py          write the files
       python paper/make_numbers.py --check  recompute in memory, compare with the files, exit 1 on any difference
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
GEN = PAPER / "generated"
POSTHOC = PROJECT / "analysis/posthoc/results"
ORIG = PROJECT / "06-Study-History/original-studies"
S1 = ORIG / "study1-value-transfer-and-readout"      # Study 1 (readout experiment; Qwen3-4B answer-only experiments)
S2 = ORIG / "study2-reasoning-interventions"          # Study 2 (held-out experiment)
S3 = PROJECT / "studies/3-heldout-4b-confirmation"   # Study 3 (read only)
S4 = PROJECT / "studies/4-multidigit-tokens"         # Study 4 (read only)
S5 = PROJECT / "studies/5-direction-and-span"        # Study 5 (read only)
S6 = PROJECT / "studies/6-olmo2-7b-multidigit"       # Study 6 (read only)

sys.path.insert(0, str(PROJECT / "analysis/posthoc"))
from common import exact_interval, upper_bound  # noqa: E402  (same interval code as the post hoc analyses)


def load(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------- formatting
def pc(x, nd=1):
    """A proportion as a LaTeX percentage; exact 0 and 1 are written without decimals."""
    if x <= 0:
        return r"0\%"
    if x >= 1:
        return r"100\%"
    return f"{100 * x:.{nd}f}" + r"\%"


def br(lo, hi):
    return f"[{pc(lo)}, {pc(hi)}]"


def ci(k, n):
    return br(*exact_interval(k, n))


def le(x):
    return r"$\le$\," + pc(x)


def sci(x, sig=2):
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10 ** e, sig - 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.{sig - 1}f}\times 10^{{{e}}}}}"


def num(n):
    return f"{n:,}" if n >= 1000 else str(n)


def word(n):
    """Small counts written as words in running text."""
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n]


def frac(k, n):
    return f"{k}/{n}"


# ---------------------------------------------------------------------------------------------- inputs
R = {k: v["value"] for k, v in load(GEN / "reconciliation/numbers.json").items()}
TR = load(POSTHOC / "transfer-rates.json")
FDE = load(POSTHOC / "first-digit-equivalence.json")
OPS = load(POSTHOC / "operand-substitution.json")
SCR = load(POSTHOC / "screen-decomposition.json")
EXPO = load(POSTHOC / "trace-exposure.json")
TOK = load(POSTHOC / "tokenizer-check.json")
TIMING_MAIN = load(S2 / "04-Results/analysis/main-secondary-v2.json")["timing"]
TIMING_CTRL = load(S2 / "04-Results/analysis/control-secondary-v2.json")["timing"]
V3_EXPOSURE = load(S1 / "results/v3/analysis/prefix-exposure-summary.json")
CFG_06B = load(S1 / "models/Qwen3-0.6B/config.json")
CFG_4B = load(S1 / "hidden-value-transfer/competence-v2/models/qwen4b/config.json")

MACROS: dict[str, tuple[str, str]] = {}


def m(name, value, source):
    assert name not in MACROS, name
    MACROS[name] = (str(value), source)


# ---------------------------------------------------------------------------------------------- models
m("ZeroSixBlocks", CFG_06B["num_hidden_layers"], "Qwen3-0.6B config.json")
m("FourBBlocks", CFG_4B["num_hidden_layers"], "Qwen3-4B-Instruct-2507-FP8 config.json")
for name, f in TOK["files"].items():                       # results text: digits isolated, only the ten digits
    assert f["digits_isolated"] and f["entries_containing_a_digit"] == [str(d) for d in range(10)], name
    assert f["multi_digit_entries"] == 0, name
assert TOK["identical"] and len(TOK["files"]) == 3

# ---------------------------------------------------------------------------------------------- readout experiment
ro = TR["readout_qwen3_06b"]
act, full, dig = ro["activation"], ro["text_full_value"], ro["text_first_digit"]
SRC = "transfer-rates.json (readout)"
m("RoRecords", act["n"], SRC)
m("RoGroups", act["clusters"]["count"], SRC)
m("RoActSuccess", act["k"], SRC)
m("RoActRate", pc(act["p"]), SRC)
m("RoActCI", br(*act["exact95"]), SRC)
m("RoActUnits", act["clusters"]["with_success"], SRC + ", program groups with a success")
m("RoActUnitsCI", br(*exact_interval(act["clusters"]["with_success"], act["clusters"]["count"])),
  SRC + ", exact interval on program groups with a success")
m("RoFullSuccess", full["k"], SRC)
m("RoFullRate", pc(full["p"]), SRC)
m("RoFullCI", br(*full["exact95"]), SRC)
rd = FDE["qwen3_06b_readout"]
assert rd["activation_successes"] == act["k"] and rd["first_digit_text_successes"] == dig["k"]
width = rd["activation_successes_by_donor_width"]
assert int(width["1"]) + int(width["2"]) == act["k"]
m("RoOneDigitSuccess", width["1"], "first-digit-equivalence.json (readout)")
m("RoTwoDigitSuccess", width["2"], "first-digit-equivalence.json (readout)")
assert rd["activation_successes_two_digit_also_text"] == int(width["2"])   # text: "each of the ... two-digit successes"
m("RoBothSuccess", rd["successes_under_both"], "first-digit-equivalence.json (readout)")
rr = R["ai02.readout"]
assert rr["planned"] == act["n"] and rr["activation_success"] == act["k"] and rr["full_value_text_success"] == full["k"]
assert rr["source_first_token_copy"][1] == act["n"]
m("RoFirstTokCopy", rr["source_first_token_copy"][0], "reconciliation numbers.json ai02.readout")
m("RoDigitExact", rr["first_digit_exact_sequence"][0], "reconciliation numbers.json ai02.readout")
m("RoDigitPairs", rr["first_digit_exact_sequence"][1], "reconciliation numbers.json ai02.readout")
m("RoDigitExactCI", ci(*rr["first_digit_exact_sequence"]), "exact interval")
m("RoWidthIdentical", rr["width_invariance"][0], "reconciliation numbers.json ai02.readout")
m("RoWidthPairs", rr["width_invariance"][1], "reconciliation numbers.json ai02.readout")
dt = EXPO["readout_distinct_traces"]
m("RoRecipientClosed", dt["recipient_closed"], "trace-exposure.json, distinct traces")
m("RoRecipientZ", dt["recipient_z_in_thinking"], "trace-exposure.json, distinct traces")
m("RoDonorClosed", dt["donor_closed"], "trace-exposure.json, distinct traces")
m("RoDonorZ", dt["donor_z_in_thinking"], "trace-exposure.json, distinct traces")
m("VthreeTraces", V3_EXPOSURE["full_generations"], "Study 1 results/v3/analysis/prefix-exposure-summary.json")
m("VthreeTracesZ", V3_EXPOSURE["z_literal_in_thinking"], "Study 1 results/v3/analysis/prefix-exposure-summary.json")

# ---------------------------------------------------------------------------------------------- held-out experiment
s2 = TR["study2_qwen3_06b"]
h20, h26 = s2["heldout/label_z-L20"], s2["heldout/label_z-L26"]
assert (h20["k"], h20["n"]) == (h26["k"], h26["n"]) and h26["k"] == 0
SRC = "transfer-rates.json (held-out)"
m("HoSuccess", h26["k"], SRC)
m("HoBranches", h26["n"], SRC)
m("HoPrograms", h26["clusters"]["count"], SRC)
m("HoBoundBranch", pc(h26["upper95_one_sided"]), SRC)
m("HoBoundProgram", pc(h26["clusters"]["upper95_one_sided_any_success"]), SRC + ", per program")
site26 = R["ai04.main.sites"]["label_z-L26"]
m("HoRecipient", site26["recipient"], "reconciliation numbers.json ai04.main.sites")
lead = R["ai04.main.block26_leading_digit"]
m("HoLeadParsed", lead["first_numeral_parsed"], "reconciliation numbers.json ai04.main.block26_leading_digit")
m("HoLeadDonor", lead["starts_with_donor_digit"], "reconciliation numbers.json ai04.main.block26_leading_digit")
m("HoLeadComplete", lead["complete_donor_numeral"], "reconciliation numbers.json ai04.main.block26_leading_digit")
tl = TIMING_MAIN["label_z-L26"]
m("HoPriorZ", tl["prior_literal_z"], "Study 2 04-Results/analysis/main-secondary-v2.json timing")
m("HoTimingAvail", tl["groups_available"], "Study 2 04-Results/analysis/main-secondary-v2.json timing")
tf = TIMING_CTRL["first_z-L26"]
assert tf["in_thinking"] == tf["groups_available"] and tf["prior_literal_z"] == 0   # inside thinking, before the value
m("HcFirstAvail", tf["groups_available"], "Study 2 04-Results/analysis/control-secondary-v2.json timing")
m("HcPrograms", tf["groups_total"], "Study 2 04-Results/analysis/control-secondary-v2.json timing")
assert s2["controls/first_z-L26/activation"]["clusters"]["count"] == tf["groups_total"]
c = R["ai04.controls"]
assert c["first_z-L26/activation"]["target"] == c["label_z-L26/activation"]["target"]
m("HcBranches", c["first_z-L26/activation"]["branches"], "reconciliation numbers.json ai04.controls")
m("HcActEarly", c["first_z-L26/activation"]["target"], "reconciliation numbers.json ai04.controls")
m("HcFullEarly", c["first_z-L26/text_full"]["target"], "reconciliation numbers.json ai04.controls")
m("HcFullLate", c["label_z-L26/text_full"]["target"], "reconciliation numbers.json ai04.controls")
m("HcExhaustedEarly", c["first_z-L26/activation"]["exhausted"], "reconciliation numbers.json ai04.controls")
dp = R["ai04.controls.digit_pairs"]
m("HcDigitEarly", dp["first_z-L26"]["exact_sequence"], "reconciliation numbers.json ai04.controls.digit_pairs")
m("HcDigitEarlyN", dp["first_z-L26"]["runnable_pairs"], "reconciliation numbers.json ai04.controls.digit_pairs")
m("HcDigitLate", dp["label_z-L26"]["exact_sequence"], "reconciliation numbers.json ai04.controls.digit_pairs")
m("HcDigitLateN", dp["label_z-L26"]["runnable_pairs"], "reconciliation numbers.json ai04.controls.digit_pairs")
dev = R["ai04.development.sites"]
assert dev["first_z-L26"]["C"][1] == dev["label_z-L26"]["C"][1]
m("HdPrograms", s2["development/first_z-L26"]["clusters"]["count"], "transfer-rates.json (development)")
m("HdAttempts", dev["first_z-L26"]["C"][1], "reconciliation numbers.json ai04.development.sites")
m("HdAEarly", f"{dev['first_z-L26']['mean_A']:.1f}", "reconciliation numbers.json ai04.development.sites")
m("HdALate", f"{dev['label_z-L26']['mean_A']:.1f}", "reconciliation numbers.json ai04.development.sites")
m("HdAEarlyRound", round(min(dev["first_z-L26"]["mean_A"], dev["label_z-L26"]["mean_A"])),
  "reconciliation numbers.json ai04.development.sites, rounded")
m("HdCEarly", dev["first_z-L26"]["C"][0], "reconciliation numbers.json ai04.development.sites")
m("HdCLate", dev["label_z-L26"]["C"][0], "reconciliation numbers.json ai04.development.sites")
ds = R["ai04.donor_semantics"]
wide = [ds["same_leading_wide/first_z"], ds["same_leading_wide/label_z"]]
assert all(w["exact_original_sequence"] == w["branches"] == wide[0]["branches"] for w in wide)   # "in all ... programs"
m("DpPrograms", wide[0]["branches"], "reconciliation numbers.json ai04.donor_semantics")
m("DpOwnTarget", sum(w["own_target"] for w in wide), "reconciliation numbers.json ai04.donor_semantics")
base = R["ai04.main.baselines"]
for role in ("donor", "recipient", "supplied"):
    m("HoBase" + role.capitalize(), base[role]["correct_final"], "reconciliation numbers.json ai04.main.baselines")

# ---------------------------------------------------------------------------------------------- Qwen3-4B
qsel = R["ai02.qualification.Qwen3-4B-FP8, instruction plus three demonstrations"]
m("FbPairs", qsel["donor_full"][1], "reconciliation numbers.json ai02.qualification")
m("FbDonorCorrect", qsel["donor_full"][0], "reconciliation numbers.json ai02.qualification")
m("FbRecipientCorrect", qsel["recipient_full"][0], "reconciliation numbers.json ai02.qualification")
m("FbPromptBaseline", qsel["textual_counterfactual"][0], "reconciliation numbers.json ai02.qualification")
m("FbFitPrompts", R["ai02.subspace.fitting_baselines"]["first"][1], "reconciliation numbers.json ai02.subspace")
fb = TR["qwen3_4b"]
WV = ["Block 20, first operand", "Block 9, first operand", "Block 10, first operand", "Block 14, second operand",
      "Block 20, first-stage end", "Block 23, answer prefix"]
b23 = fb["whole_vector/Block 23, answer prefix"]
assert b23["k"] == max(fb["whole_vector/" + s]["k"] for s in WV)                      # "the best whole-vector site"
assert all(fb[w + "/Block 23, answer prefix"]["k"] == 1 for w in ("rank8_projection", "attention_output", "mlp_output"))
m("FbDonors", b23["clusters"]["count"], "transfer-rates.json (Qwen3-4B), donors")
m("FbBestSuccess", b23["k"], "transfer-rates.json (Qwen3-4B)")
m("FbBestCI", br(*b23["exact95"]), "transfer-rates.json (Qwen3-4B)")
FD = {(s["write"], s["site"]): s for s in FDE["qwen3_4b"]}
a = FD[("whole_vector", "Block 23, answer prefix")]
assert a["first_token_follows_then_identical"] == a["identical_to_text_edit"]
assert a["first_token_follows_then_identical"] + a["first_token_follows_then_diverges"] == a["first_token_is_donor_digit"]
SRC = "first-digit-equivalence.json (block 23)"
m("FbFirstTok", a["first_token_is_donor_digit"], SRC)
m("FbDiffer", a["donor_digit_differs_from_recipient_digit"], SRC)
m("FbFirstTokDiffer", a["first_token_is_donor_digit_where_digits_differ"], SRC)
m("FbIdentical", a["identical_to_text_edit"], SRC)
m("FbIdenticalCI", ci(a["identical_to_text_edit"], a["pairs"]), SRC)
m("FbDiverge", a["first_token_follows_then_diverges"], SRC)
m("FbSecondDonor", a["diverges_with_second_token_of_donor_answer"], SRC)
m("FbActChanged", a["answer_changed"], SRC)
m("FbNoiseChanged", a["noise_answer_changed"], SRC)
m("FbNoiseChangedCI", ci(a["noise_answer_changed"], a["noise_pairs"]), SRC)
others = [FD[("whole_vector", s)] for s in WV if s != "Block 23, answer prefix"]
m("FbNoiseChangedOther", max(s["noise_answer_changed"] for s in others), "first-digit-equivalence.json (other sites)")
first_op = [FD[("whole_vector", s)]["answer_changed"] for s in WV[:3]]
assert len(set(first_op)) == 1
m("FbOperandChanged", first_op[0], "first-digit-equivalence.json (first-operand sites)")
fresh = {s: R["ai02.fresh." + s] for s in WV}
m("FbSameLead", fresh["Block 23, answer prefix"][1]["same_leading_identical_outputs"], "reconciliation ai02.fresh")
m("FbSubSameLead", R["ai02.subspace.same_leading_identical_outputs"], "reconciliation ai02.subspace")
m("FbFseUnchanged", fresh["Block 20, first-stage end"][0]["donor"]["unchanged"], "reconciliation ai02.fresh")

# diagnostic (value-digit qualified sources)
dd, dc = fb["diagnostic/donor/discriminating"], fb["diagnostic/same_value_coupled/discriminating"]
assert dd["k"] == dc["k"] == 0 and dd["n"] == dc["n"]
SRC = "transfer-rates.json (diagnostic)"
m("DgDisc", dd["n"], SRC)
m("DgFullDisc", dd["k"], SRC)
m("DgBoundPair", pc(dd["upper95_one_sided"]), SRC)
m("DgDonors", dd["clusters"]["count"], SRC + ", donor values")
m("DgBoundDonor", pc(dd["clusters"]["upper95_one_sided_any_success"]), SRC + ", donor values")
qs = R["ai02.qualified_sources.conditions"]
m("DgPairs", qs["donor"]["n"], "reconciliation ai02.qualified_sources")
m("DgCoarseDonor", qs["donor"]["coarse"], "reconciliation ai02.qualified_sources")
m("DgCoarseCoupled", qs["same_value_coupled"]["coarse"], "reconciliation ai02.qualified_sources")
m("DgCoarseText", qs["text_actual_first_token"]["coarse"], "reconciliation ai02.qualified_sources")
m("DgCoarseJoint", R["ai02.qualified_sources.joint"]["coarse"], "reconciliation ai02.qualified_sources.joint")

# screen decomposition
sm = SCR["summary"]
top = sm["top3"]
assert [(t["layer"], t["role"]) for t in top] == [(20, "a_end"), (9, "a_end"), (10, "a_end")]
assert all(t["recipient_suppression"] / t["delta_log_ratio"] > 0.8 for t in top)           # "split the same way"
assert R["ai02.screen.records"] == 5184 and abs(top[0]["delta_log_ratio"] - R["ai02.screen.ratio_top3"][0][2]) < 1e-3
SRC = "screen-decomposition.json"
m("ScSites", sm["sites"], SRC)
m("ScRecords", num(R["ai02.screen.records"]), "reconciliation ai02.screen.records")
m("ScTopRatio", f"{top[0]['delta_log_ratio']:.1f}", SRC)
m("ScTopSupp", f"{top[0]['recipient_suppression']:.1f}", SRC)
m("ScTopGain", f"{top[0]['target_gain']:.1f}", SRC)
m("ScTopPTarget", sci(top[0]["mean_p_target"], 3), SRC)
m("ScBigSites", sm["sites_with_ratio_above_10"], SRC)
m("ScMinSuppShare", f"{math.floor(100 * sm['min_suppression_share_above_10'])}" + r"\%", SRC + ", rounded down")
m("ScMaxPEarly", sci(sm["max_mean_p_target_blocks_0_11"]), SRC)
peak = max(SCR["sites"], key=lambda s: s["mean_p_target"])
assert (peak["layer"], peak["role"]) == (23, "prompt_end")
m("ScMaxP", f"{peak['mean_p_target']:.3f}", SRC)

# operand substitution
OP = {s["site"]: s for s in OPS["sites"]}
SOURCES = ["donor", "same_z_decomposition", "same_z_different_answer", "different_z_same_answer",
           "same_leading_different_value", "random_source"]
CONTROLS = ["isotropic_delta", "unpatched", "same_example"]
A_SITES, FOUR = WV[:3], WV[:4]
donor_k = [OP[s]["sources"]["donor"]["k"] for s in A_SITES]
assert all(OP[s]["sources"][c]["n"] == 36 for s in A_SITES for c in SOURCES)
SRC = "operand-substitution.json"
m("OpDonorMin", min(donor_k), SRC)
m("OpDonorMax", max(donor_k), SRC)
m("OpDonorMinCI", ci(min(donor_k), 36), SRC)
m("OpDonorMaxCI", ci(max(donor_k), 36), SRC)
other_k = [OP[s]["sources"][c]["k"] for s in A_SITES for c in SOURCES[1:]]
m("OpOtherMin", min(other_k), SRC)
m("OpOtherMax", max(other_k), SRC)
m("OpControlMatches", sum(OP[s]["controls"][c]["k"] for s in FOUR for c in CONTROLS), SRC)
m("OpControlAttempts", sum(OP[s]["controls"][c]["n"] for s in FOUR for c in CONTROLS), SRC)
b14 = OP["Block 14, second operand"]["sources"]
m("OpBDonor", b14["donor"]["k"], SRC)
m("OpBDonorN", b14["donor"]["n"], SRC)
m("OpBSameZ", b14["same_z_decomposition"]["k"], SRC)
m("OpBSameZN", b14["same_z_decomposition"]["n"], SRC)
m("OpMatches", sum(OP[s]["sources"][c]["k"] for s in FOUR for c in SOURCES), SRC)
m("OpBeyond", sum(OP[s]["sources"][c]["matches_differing_beyond_first_digit"] for s in FOUR for c in SOURCES), SRC)

# ---------------------------------------------------------------------------------------------- Study 3
# Held-out confirmation in Qwen3-4B (studies/3-heldout-4b-confirmation; read only). Every number by name from
# results/summary.json; "lower"/"upper" are the study's conservative bounds (the smaller of the exact and
# donor-cluster lower bounds, the larger of the two upper bounds), which its decision rules use.
S3SUM = load(S3 / "results/summary.json")
S3VER = load(S3 / "results/verification.json")
assert S3VER["all_passed"] and all(c["ok"] for c in S3VER["checks"].values())
N3 = S3SUM["numbers"]
D3 = S3SUM["decisions"]
SITES3 = ["block23_answer_prefix", "block20_first_operand", "block9_first_operand", "block10_first_operand",
          "block14_second_operand", "block20_first_step_end"]
SRC = "studies/3-heldout-4b-confirmation/results/summary.json"


def n3(key):
    return N3[key]


def kn3(run, site, outcome):
    return n3(f"{run}.sites.{site}.{outcome}.k"), n3(f"{run}.sites.{site}.{outcome}.n")


def cons3(run, site, outcome):
    return br(n3(f"{run}.sites.{site}.{outcome}.lower"), n3(f"{run}.sites.{site}.{outcome}.upper"))


rc = S3SUM["reserve_checks"]
assert rc["never_evaluated"] and rc["no_overlap_with_development"] and rc["pair_overlap_with_development"] == 0
m("SthPairs", rc["reserve_pairs"], SRC + " reserve_checks")
m("SthDonors", rc["reserve_groups"], SRC + " reserve_checks")
for run, tag in (("box", ""),):
    assert n3("machine.competence.pass_check")                       # text: both runs passed the gate
    comp = f"{run}.competence.conditions"
    assert n3(f"{run}.competence.pass_check") and n3(f"{run}.competence.n_pairs") == rc["reserve_pairs"]
    m(f"Sth{tag}DonorCorrect", n3(f"{comp}.donor_full.correct"), SRC)
    m(f"Sth{tag}RecipientCorrect", n3(f"{comp}.recipient_full.correct"), SRC)
    m(f"Sth{tag}SuppliedCorrect", n3(f"{comp}.textual_counterfactual.correct"), SRC)
# P1: complete transfer at every site
cf = [kn3("box", s, "complete_counterfactual") for s in SITES3]
m("SthMaxCf", max(k for k, _ in cf), SRC + " (P1)")
m("SthMaxCfUpper", pc(max(n3(f"box.sites.{s}.complete_counterfactual.upper") for s in SITES3)), SRC + " (P1)")
m("SthResMaxCf", max(kn3("machine", s, "complete_counterfactual")[0] for s in SITES3), SRC + " (P1, reserve wording)")
assert D3["box"]["P1"]["verdict"] == "supported"
ops = {o: n3(f"box.sites.block23_answer_prefix.complete_counterfactual_by_operation.{o}") for o in ("add", "sub", "mul")}
assert ops["add"] == cf[0][0] and ops["sub"] == ops["mul"] == 0     # text: both block-23 successes under addition
# P2: first token follows the donor at the answer prefix
k, n = kn3("box", "block23_answer_prefix", "first_token_follows_donor")
m("SthFirstTok", k, SRC + " (P2)")
m("SthFirstTokRate", pc(k / n), SRC + " (P2)")
m("SthFirstTokCons", cons3("box", "block23_answer_prefix", "first_token_follows_donor"), SRC + " (P2)")
diff = D3["box"]["P2"]
m("SthAboveNoise", round(100 * diff["minus_noise"]), SRC + " (P2), percentage points")
m("SthAboveNoiseCI", "[" + ", ".join(str(round(100 * x)) for x in diff["minus_noise_cluster95"]) + "]",
  SRC + " (P2), donor-cluster interval, percentage points")
# P3 and P3s
k, n = kn3("box", "block23_answer_prefix", "equals_first_digit_text_edit")
m("SthIdentical", k, SRC + " (P3)")
m("SthIdenticalCons", cons3("box", "block23_answer_prefix", "equals_first_digit_text_edit"), SRC + " (P3)")
k, n = kn3("box", "block23_answer_prefix", "equals_own_first_token_text_edit")
m("SthOwnTok", k, SRC + " (P3s)")
m("SthOwnTokCons", cons3("box", "block23_answer_prefix", "equals_own_first_token_text_edit"), SRC + " (P3s)")
m("SthOwnTokOther", n - k, SRC + " (P3s), pairs whose later tokens differ")
k, n = kn3("box", "block23_answer_prefix", "diverges_given_first_token_follows")
# P4: operand substitution
for s, tag in (("block20_first_operand", "Twenty"), ("block9_first_operand", "Nine"), ("block10_first_operand", "Ten")):
    k, n = kn3("box", s, "first_operand_substitution_informative")
    m(f"SthOp{tag}", k, SRC + " (P4)")
    m(f"SthOp{tag}N", n, SRC + " (P4)")
    if tag == "Twenty":
        m(f"SthOp{tag}Lower", pc(n3(f"box.sites.{s}.first_operand_substitution_informative.lower")), SRC + " (P4)")
k, n = kn3("box", "block14_second_operand", "second_operand_substitution_informative")
m("SthOpFourteen", k, SRC + " (P4)")
m("SthOpFourteenN", n, SRC + " (P4)")
m("SthOpFourteenLower", pc(n3("box.sites.block14_second_operand.second_operand_substitution_informative.lower")),
  SRC + " (P4)")
p4 = D3["box"]["P4"]["sites"]
m("SthOpSitesWord", word(len(p4)), SRC + " (P4), operand sites")
m("SthOpSitesSupportedWord", word(sum(s["verdict"] == "supported" for s in p4.values())), SRC + " (P4), sites supported")
m("SthRulesWord", word(len(D3["box"])), SRC + ", number of decision rules")
assert [p4[s]["verdict"] for s in ("block20_first_operand", "block9_first_operand", "block10_first_operand",
                                   "block14_second_operand")] == ["supported", "inconclusive", "inconclusive", "supported"]
# P5 and P6
k, n = kn3("box", "block20_first_step_end", "recipient_answer_unchanged")
m("SthFseUnchanged", k, SRC + " (P5)")
k, n = kn3("box", "block23_answer_prefix", "same_leading_source_identical_output")
m("SthSameLead", k, SRC + " (P6)")
m("SthSameLeadCons", cons3("box", "block23_answer_prefix", "same_leading_source_identical_output"), SRC + " (P6)")
m("SthResSameLead", kn3("machine", "block23_answer_prefix", "same_leading_source_identical_output")[0],
  SRC + " (P6, reserve wording)")
# descriptive (not prespecified): which source gives the donor write's own output at block 23
c23 = "box.sites.block23_answer_prefix.conditions"
m("SthSameAnswerIdentical", n3(f"{c23}.different_z_same_answer.identical_to_donor_write"), SRC + ", descriptive")
m("SthSameZIdentical", n3(f"{c23}.same_z_different_answer.identical_to_donor_write"), SRC + ", descriptive")
m("SthDonorCopy", kn3("box", "block23_answer_prefix", "donor_answer_copy")[0], SRC + ", donor's answer as the output")
VERDICT = {"supported": "supported", "refuted": "refuted", "inconclusive": "inconclusive"}


def verdict(run, rule):
    v = D3[run][rule]["verdict"]
    if rule == "P4":                                                 # site-level summary of the operand rule
        sites = [s["verdict"] for s in D3[run]["P4"]["sites"].values()]
        return f"{sites.count('supported')} of {len(sites)} sites"
    assert v in VERDICT, v
    return v


rows3 = []
p1_up = max(n3(f"box.sites.{s}.complete_counterfactual.upper") for s in SITES3)
k23 = kn3("box", "block23_answer_prefix", "first_token_follows_donor")
rows3.append(["P1", "Complete transfer is rare at all six sites",
              f"at most {max(k for k, _ in cf)}/{cf[0][1]} per site, upper bounds " + r"$\le$\," + pc(p1_up)])
rows3.append(["P2", "First token follows the donor in most pairs, above noise (block 23)",
              f"{k23[0]}/{k23[1]} " + cons3("box", "block23_answer_prefix", "first_token_follows_donor")])
for rule, label, outcome in (("P3", "Output equals the first-digit text edit (block 23)", "equals_first_digit_text_edit"),
                             ("P3s", "Output equals the text edit of the write's own first token", "equals_own_first_token_text_edit")):
    k, n = kn3("box", "block23_answer_prefix", outcome)
    rows3.append([rule, label, f"{k}/{n} " + cons3("box", "block23_answer_prefix", outcome)])
ops4 = [kn3("box", s, "first_operand_substitution_informative") for s in SITES3[1:4]]
ops4.append(kn3("box", "block14_second_operand", "second_operand_substitution_informative"))
rows3.append(["P4", "Operand writes substitute the operand (blocks 20, 9, 10; 14)",
              ", ".join(f"{k}/{n}" for k, n in ops4)])
k, n = kn3("box", "block20_first_step_end", "recipient_answer_unchanged")
rows3.append(["P5", "End-of-first-step write leaves the answer unchanged",
              f"{k}/{n} " + cons3("box", "block20_first_step_end", "recipient_answer_unchanged")])
k, n = kn3("box", "block23_answer_prefix", "same_leading_source_identical_output")
rows3.append(["P6", "Same-leading-digit source gives the identical output (block 23)",
              f"{k}/{n} " + cons3("box", "block23_answer_prefix", "same_leading_source_identical_output")])
TABLE3 = [" & ".join([r, label, res, verdict("development", r), verdict("box", r), verdict("machine", r)]) + r" \\"
          for r, label, res in rows3]

# ---------------------------------------------------------------------------------------------- Study 4
# Multi-digit number tokens (studies/4-multidigit-tokens; read only). Candidate statuses from results/candidates.json,
# gate counts from results/summary.json: candidates 1 and 2 failed the competence gate, candidate 3 was not tried.
S4SUM = load(S4 / "results/summary.json")
S4CAND = load(S4 / "results/candidates.json")
SRC = "studies/4-multidigit-tokens/results/summary.json"
cand, order = S4CAND["candidates"], S4CAND["order"]
assert order == S4SUM["candidates"]["order"] and S4CAND["selected"] is None
assert all(S4SUM["tokenizer_check"][cn] for cn in order)
TRIED = [cn for cn in order if cand[cn]["status"].startswith("failed: competence gate")]
assert TRIED == order[:2] and all(cand[cn]["status"].startswith("not tried") for cn in order[2:])
m("SfoCandidatesWord", word(len(order)), SRC + ", number of candidates")
m("SfoTriedWord", word(len(TRIED)), "studies/4-multidigit-tokens/results/candidates.json, candidates tried")
import re as _re                                                     # noqa: E402
PROFILE = _re.compile(r"(\d+) of (\d+)(?: wrong)? with a multiplication second step, (\d+) of (\d+) with addition, "
                      r"(\d+) of (\d+) with subtraction")
TH = cand[order[0]]["gate"]["thresholds"]
assert all(cand[cn]["gate"]["thresholds"] == TH for cn in TRIED)
m("SfoGate", pc(TH["central"], 0), SRC + ", gate on each central baseline")
TABLE4 = {}
for cn, tag in zip(TRIED, ("", "Smol")):
    g = S4SUM[f"{cn}_gate"]
    assert not g["pass_check"] and g["instrumentation_pass"] and g["max_null_logit_difference"] == 0.0
    cnt = g["counts"]
    n = cnt["donor_full"]["n"]
    single = [cnt[k]["correct"] for k in ("donor_first", "recipient_first", "recipient_remaining", "textual_counterfactual")]
    assert len(set(single)) == 1 and single[0] == n                # text: every step asked alone was answered correctly
    assert all(cnt[k]["invalid"] == 0 for k in cnt)                 # every output parsed as an integer
    m(f"Sfo{tag}DonorCorrect", cnt["donor_full"]["correct"], SRC + f" ({cn})")
    m(f"Sfo{tag}DonorRate", pc(g["rates"]["donor_full"]), SRC + f" ({cn})")
    m(f"Sfo{tag}RecipientCorrect", cnt["recipient_full"]["correct"], SRC + f" ({cn})")
    m(f"Sfo{tag}RecipientRate", pc(g["rates"]["recipient_full"]), SRC + f" ({cn})")
    prof = PROFILE.search(cand[cn]["error_profile"])
    assert prof, cand[cn]["error_profile"]
    mul, muln, add, addn, sub, subn = map(int, prof.groups())
    assert mul / muln > max(add / addn, sub / subn)                 # text: errors most frequent with a multiplication
    for name, v in (("MulWrong", mul), ("MulN", muln)):               # the full profile is in table-study4.tex
        m(f"Sfo{tag}{name}", v, "studies/4-multidigit-tokens/results/candidates.json, error profile of " + cn)
    TABLE4[cn] = {"donor": (cnt["donor_full"]["correct"], n, g["rates"]["donor_full"]),
                 "recipient": (cnt["recipient_full"]["correct"], n, g["rates"]["recipient_full"]),
                 "supplied": (cnt["textual_counterfactual"]["correct"], n),
                 "first": (cnt["donor_first"]["correct"], n),
                 "triple": (g["triple_competent"]["k"], g["triple_competent"]["n"]),
                 "profile": ((mul, muln), (add, addn), (sub, subn))}
ns = {TABLE4[cn]["donor"][1] for cn in TRIED}
assert len(ns) == 1
m("SfoCalPairs", ns.pop(), SRC + ", calibration pairs")

# ---------------------------------------------------------------------------------------------- Study 6
# Study 4's design with OLMo-2-1124-7B-Instruct (studies/6-olmo2-7b-multidigit; read only): it ended at the
# competence gate. Same calibration split and gate thresholds as Study 4.
S6SUM = load(S6 / "results/summary.json")
SRC = "studies/6-olmo2-7b-multidigit/results/summary.json"
g6 = S6SUM["gate"]
assert S6SUM["tokenizer_check"] and S6SUM["weights_check"] and S6SUM["instrumentation"]["pass_check"]
assert S6SUM["instrumentation"]["largest_null_write_logit_change"] == 0.0 and not g6["pass_check"]
assert g6["thresholds"] == TH and g6["pairs"] == TABLE4[TRIED[0]]["donor"][1]
cnt6 = g6["counts"]
n6 = cnt6["donor_full"]["n"]
assert all(cnt6[k]["correct"] == n6 for k in ("donor_first", "recipient_first", "recipient_remaining", "textual_counterfactual"))
assert all(cnt6[k]["invalid"] == 0 for k in cnt6)
m("SsiDonorCorrect", cnt6["donor_full"]["correct"], SRC)
m("SsiDonorRate", pc(cnt6["donor_full"]["correct"] / n6), SRC)
m("SsiRecipientCorrect", cnt6["recipient_full"]["correct"], SRC)
m("SsiRecipientRate", pc(cnt6["recipient_full"]["correct"] / n6), SRC)
m("SsiTriple", g6["triple_competent"], SRC + ", pairs correct on all three central measures")
prof6 = S6SUM["error_profile"]["by_second_operation"]
assert prof6["mul"]["wrong"] / prof6["mul"]["distinct"] > max(prof6[o]["wrong"] / prof6[o]["distinct"] for o in ("add", "sub"))
m("SsiMulWrong", prof6["mul"]["wrong"], SRC + ", distinct problems wrong with a multiplication second step")
m("SsiMulN", prof6["mul"]["distinct"], SRC)
m("SsiGateSeconds", round(S6SUM["gate_seconds"]), SRC + ", gate run time on the L4")
TABLE4["olmo2-7b"] = {"donor": (cnt6["donor_full"]["correct"], n6, cnt6["donor_full"]["correct"] / n6),
                      "recipient": (cnt6["recipient_full"]["correct"], n6, cnt6["recipient_full"]["correct"] / n6),
                      "supplied": (cnt6["textual_counterfactual"]["correct"], n6),
                      "first": (cnt6["donor_first"]["correct"], n6),
                      "triple": (g6["triple_competent"], n6),
                      "profile": tuple((prof6[o]["wrong"], prof6[o]["distinct"]) for o in ("mul", "add", "sub"))}
GATED = TRIED + ["olmo2-7b"]

# ---------------------------------------------------------------------------------------------- Study 5
# Added mean-difference direction and span writes in Qwen3-4B (studies/5-direction-and-span; read only). Every
# number from results/summary.json, which collects the first-stage checks, the development selection and analysis,
# and the test analysis; the test's verify.py must have passed.
S5SUM = load(S5 / "results/summary.json")
S5VER = load(S5 / "results/verification.json")
assert S5VER["passed"] == S5VER["total"], "Study 5 verification has failures"
SRC = "studies/5-direction-and-span/results/summary.json"
d5, f5, sel5 = S5SUM["design"], S5SUM["first_stage"], S5SUM["selection"]
dev5, test5 = S5SUM["development"], S5SUM["test"]
assert f5["instrumentation"] and f5["reproduction"]["passed"] and f5["span_null_max"] == 0.0
assert f5["reproduction"]["tokens_equal"] == f5["reproduction"]["records"]
assert f5["reproduction"]["vectors_bytes_equal"] == f5["reproduction"]["vectors_checked"]
m("SfiFitPerValue", d5["fit_per_value"], SRC + ", fitting problems per intermediate value")
m("SfiFitProblems", num(d5["fit_problems"]), SRC + ", fitting problems")
m("SfiPostSpanLen", word(d5["span_lengths"]["post_operand"]), SRC + ", tokens in the post-operand span")
m("SfiFullSpanLen", d5["span_lengths"]["first_step"], SRC + ", tokens in the first-step span")
m("SfiReproRecords", f5["reproduction"]["records"], SRC + ", reproduced Study 3 records")
m("SfiReproVectors", f5["reproduction"]["vectors_checked"], SRC + ", reproduced Study 3 donor vectors")
assert str(test5["pairs"]) == MACROS["SthPairs"][0] and str(dev5["pairs"]) == MACROS["FbPairs"][0]   # Study 3's held-out and the development pairs
SITE5 = {d["role"]: d for d in sel5["direction_sites"]}
SPAN5 = {x["span"]: x for x in sel5["span_sites"]}


def kn(p):
    return p["k"], p["n"]


def cons(p):
    return br(p["lower"], p["upper"])


TD, TS, TR5 = test5["sites"]["direction"], test5["sites"]["span"], test5["rules"]
DD, DS = dev5["sites"]["direction"], dev5["sites"]["span"]
for role, tag in (("first_stage_end", "Fse"), ("prompt_end", "Pe"), ("a_end", "Ae"), ("b_end", "Be")):
    st = SITE5[role]
    m(f"Sfi{tag}Block", st["layer"], SRC + f", selected direction block, {role}")
    m(f"Sfi{tag}Alpha", word(st["alpha"]) if st["alpha"] != 1 else "one", SRC + f", selected multiple of the difference, {role}")
    r = TD[role]
    m(f"Sfi{tag}Cf", r["donor"]["k"], SRC + f", test, direction at the selected {role} site, complete counterfactual")
    m(f"Sfi{tag}CfCons", cons(r["donor"]), SRC + f", test, conservative range")
    m(f"Sfi{tag}Unchanged", r["donor_unchanged"]["k"], SRC + f", test, unchanged")
    m(f"Sfi{tag}NoiseCf", r["noise"]["k"], SRC + f", test, noise of the same norm, complete counterfactual")
    m(f"Sfi{tag}NoiseUnchanged", r["noise_unchanged"]["k"], SRC + f", test, noise, unchanged")
    m(f"Sfi{tag}SameLead", r["same_leading"]["k"], SRC + f", test, same-leading-digit direction, its own counterfactual")
    m(f"Sfi{tag}DevCf", DD[role]["donor"]["k"], SRC + f", development, selected {role} site")
    if "gurnee" in r:
        m(f"Sfi{tag}Flips", r["gurnee"]["flips"]["k"], SRC + f", test, counterfactual more likely than the recipient's answer")
        m(f"Sfi{tag}Change", f"{r['gurnee']['mean_change']:.1f}", SRC + f", test, mean change of the log-probability difference (nats)")
        m(f"Sfi{tag}DevFlips", DD[role]["gurnee"]["flips"]["k"], SRC + f", development, flips")
for span, tag in (("post_operand", "Post"), ("first_step", "Full")):
    r = TS[span]
    m(f"Sfi{tag}Block", SPAN5[span]["layer"], SRC + f", selected span block, {span}")
    m(f"Sfi{tag}Cf", r["donor"]["k"], SRC + f", test, span {span}, complete counterfactual")
    m(f"Sfi{tag}CfCons", cons(r["donor"]), SRC + ", test, conservative range")
    m(f"Sfi{tag}Unchanged", r["donor_unchanged"]["k"], SRC + ", test, unchanged")
    m(f"Sfi{tag}SourcesCf", sum(v["k"] for v in r["sources"].values()), SRC + ", test, five other sources, own counterfactuals")
    m(f"Sfi{tag}SourcesN", sum(v["n"] for v in r["sources"].values()), SRC + ", test, five other sources, writes")
    rows = test5["sweeps"]["span"][span]
    m(f"Sfi{tag}SweepMaxCf", max(x["cf"] for x in rows), SRC + f", test, largest complete counterfactual count over blocks, {span}")
    m(f"Sfi{tag}SweepMinUnchanged", min(x["unchanged"] for x in rows), SRC + f", test, smallest unchanged count over blocks, {span}")
full_rows = test5["sweeps"]["span"]["first_step"]
early = [x for x in full_rows if x["layer"] <= 11]
late = [x for x in full_rows if x["layer"] >= 18]
m("SfiFullFirstOpMin", min(x["first_op"] for x in early), SRC + ", test, first-step span blocks 0-11, fewest first-operation substitutions")
m("SfiFullFirstOpMax", max(x["first_op"] for x in early), SRC + ", test, first-step span blocks 0-11, most first-operation substitutions")
m("SfiFullLateMinUnchanged", min(x["unchanged"] for x in late), SRC + ", test, first-step span from block 18, fewest unchanged")
m("SfiFullLateFirstOpMax", max(x["first_op"] for x in late), SRC + ", test, first-step span from block 18, most first-operation substitutions")
for role, tag in (("first_stage_end", "Fse"), ("prompt_end", "Pe")):
    rows = test5["sweeps"]["direction"][f"{role}/1"]
    m(f"Sfi{tag}SweepMaxCf", max(x["cf"] for x in rows), SRC + f", test, alpha 1 sweep, largest complete counterfactual count, {role}")
    m(f"Sfi{tag}SweepMinUnchanged", min(x["unchanged"] for x in rows), SRC + f", test, alpha 1 sweep, smallest unchanged count, {role}")
    m(f"Sfi{tag}SweepMaxZsplice", max(x["z_splice"] for x in rows), SRC + f", test, alpha 1 sweep, largest intermediate-splice count, {role}")
m("SfiTestPairs", test5["pairs"], SRC + ", test pairs")
m("SfiTestDonors", test5["donor_groups"], SRC + ", test donor groups")
VERDICT_WORDS = {"supported": "supported", "not supported": "not supported", "inconclusive": "inconclusive"}
m("SfiROne", VERDICT_WORDS[TR5["R1"]["overall"]], SRC + ", R1 overall")
m("SfiRTwo", VERDICT_WORDS[TR5["R2"]["overall"]], SRC + ", R2 overall")
m("SfiRFour", TR5["R4"]["result"], SRC + ", R4")
m("SfiVerdict", TR5["verdict"], SRC + ", overall verdict")
m("SfiVerifyChecks", S5VER["total"], "studies/5-direction-and-span/results/verification.json")
P5 = load(S5 / "results/posthoc.json")
SRCP = "studies/5-direction-and-span/results/posthoc.json"
EXP = P5["descriptive_expectations"]
r4b = P5["r4_blocks_with_upper_at_least_020"]
assert r4b and all(b["kind"] == "direction" and b["where"] == "prompt_end" for b in r4b)
assert [b["layer"] for b in r4b] == list(range(r4b[0]["layer"], r4b[-1]["layer"] + 1))
m("SfiRfourFirst", r4b[0]["layer"], SRCP + ", first block whose upper bound reached 0.20")
m("SfiRfourLast", r4b[-1]["layer"], SRCP + ", last such block")
m("SfiRfourMinK", min(b["k"] for b in r4b), SRCP)
m("SfiRfourMaxK", max(b["k"] for b in r4b), SRCP)
m("SfiRfourMaxUpper", pc(max(b["upper"] for b in r4b)), SRCP)
e4 = EXP["4_answer_prefix_blocks_25_30"]
assert e4["holds"]
m("SfiBandChangedMin", min(e4["changed"]), SRCP + ", blocks 25-30, answer changed")
m("SfiBandChangedMax", max(e4["changed"]), SRCP)
m("SfiBandZspliceMin", min(e4["z_splice"]), SRCP + ", blocks 25-30, intermediate leading-digit splice")
m("SfiBandZspliceMax", max(e4["z_splice"]), SRCP)
m("SfiBandCfMax", max(e4["cf"]), SRCP + ", blocks 25-30, complete counterfactual")
m("SfiBandAnsSpliceMax", max(e4["answer_splice"]), SRCP + ", blocks 25-30, answer leading-digit splice")
byop = P5["posthoc"]["answer_prefix_z_splice_by_operation"]
m("SfiMulZspliceMax", max(byop["mul"]["z_splice_by_block"].values()), SRCP + ", post hoc, multiplication recipients")
m("SfiOpN", byop["mul"]["n"], SRCP + ", recipients per second operation")
coin = P5["posthoc"]["coincident_source_matches"]
assert all(v["own_counterfactual"] == v["equal_to_recipient_answer"] for v in coin.values())
m("SfiCoincident", sum(v["own_counterfactual"] for v in coin.values()), SRCP + ", post hoc, span-source matches that equal the recipient's answer")
m("SfiNoiseHits", word(len(P5["posthoc"]["answer_prefix_noise_counterfactuals"])), SRCP + ", post hoc")
e2c = EXP["2c_first_step_unchanged_90_from_18"]
m("SfiFullQuietBlock", e2c["first_block_meeting_it"], SRCP + ", first block from which the first-step span leaves at least 90% unchanged")
quiet = [x for x in full_rows if x["layer"] >= e2c["first_block_meeting_it"]]
m("SfiFullQuietMin", min(x["unchanged"] for x in quiet), "studies/5-direction-and-span/results/summary.json, first-step span, fewest unchanged from that block")
assert TD["first_stage_end"]["gurnee"]["flips"]["k"] == 0                 # text: never ranked above at the selected block
assert all(x["cf"] == 0 for sp in ("post_operand", "first_step") for x in test5["sweeps"]["span"][sp])   # text: no span write
assert all(x["unchanged"] == 0 for x in full_rows if x["layer"] <= 11)          # text: changed every answer at early blocks
m("SfiEarlyLast", 11, "PREDICTIONS.md item 2: blocks 0 to 11 (early blocks of the first-step span)")
m("SfiLateFirst", 18, "PREDICTIONS.md item 2: from block 18")
m("SfiBandFirst", 25, "PREDICTIONS.md item 4: blocks 25 to 30 of the answer prefix")
m("SfiBandLast", 30, "PREDICTIONS.md item 4: blocks 25 to 30 of the answer prefix")
assert (PROJECT / "studies/5-direction-and-span/PREDICTIONS.md").read_text().count("blocks 0 to 11") >= 1
m("SfiFullEighteenUnchanged", e2c["min_unchanged"], SRCP + ", first-step span, fewest unchanged from block 18")
assert EXP["1_post_operand_unchanged_90"]["holds"] and EXP["2a_first_step_first_op_70_blocks_0_11"]["holds"]
assert EXP["2b_first_step_first_op_10_from_18"]["holds"] and not e2c["holds"] and EXP["3_first_stage_end_unchanged_90"]["holds"]
assert EXP["5_gurnee_flips_exceed_generated"]["holds"]
m("SfiExpHeldWord", word(sum(v["holds"] for v in EXP.values()) - 2), SRCP + ", expectations that held in full (items 1-5; item 2 counted once)")


def alpha_label(a):
    return "" if a == 1 else rf", {a}$\times$"


DR5 = dev5["rules"]
rows5 = []
for role, label in (("first_stage_end", "Direction, end of first step"), ("prompt_end", "Direction, answer prefix")):
    st = SITE5[role]
    rows5.append(" & ".join([f"R1", f"{label} (block {st['layer']}{alpha_label(st['alpha'])})",
                             "{}/{}".format(*kn(DD[role]["donor"])), "{}/{} {}".format(*kn(TD[role]["donor"]), cons(TD[role]["donor"])),
                             DR5["R1"]["sites"][role], TR5["R1"]["sites"][role]]) + r" \\")
for span, label in (("post_operand", "Span after the second operand"), ("first_step", "Span of the whole first step")):
    rows5.append(" & ".join(["R2", f"{label} (block {SPAN5[span]['layer']})",
                             "{}/{}".format(*kn(DS[span]["donor"])), "{}/{} {}".format(*kn(TS[span]["donor"]), cons(TS[span]["donor"])),
                             DR5["R2"]["sites"][span], TR5["R2"]["sites"][span]]) + r" \\")
sweep_max = max(max(x["cf"] for x in test5["sweeps"]["direction"][f"{r}/1"]) for r in ("first_stage_end", "prompt_end"))
sweep_max = max(sweep_max, max(max(x["cf"] for x in test5["sweeps"]["span"][sp]) for sp in ("post_operand", "first_step")))
dev_max = max(max(x["cf"] for x in dev5["sweeps"]["direction"][f"{r}/1"]) for r in ("first_stage_end", "prompt_end"))
dev_max = max(dev_max, max(max(x["cf"] for x in dev5["sweeps"]["span"][sp]) for sp in ("post_operand", "first_step")))
rows5.append(" & ".join(["R4", "Every block, both positions and both spans", f"at most {dev_max}/{dev5['pairs']}",
                         f"at most {sweep_max}/{test5['pairs']}", DR5["R4"]["result"], TR5["R4"]["result"]]) + r" \\")
rows5.append(r"\addlinespace")
rows5.append(" & ".join(["", r"\emph{Verdict}", "", "", DR5["verdict"], TR5["verdict"]]) + r" \\")
TABLE5 = rows5

# Appendix: Study 5 at the selected sites (held-out pairs); span-source successes that equal the recipient's own
# answer are counted separately in the text (results/summary.json, coincident_source_matches).
ROLES5 = ("a_end", "b_end", "first_stage_end", "prompt_end")
SPANS5 = ("post_operand", "first_step")


def c5(p):
    return "--" if p is None else frac(p["k"], p["n"])


rows = [
    "Donor: complete counterfactual & " + " & ".join(c5(TD[r]["donor"]) for r in ROLES5) + " & " + " & ".join(c5(TS[s]["donor"]) for s in SPANS5) + r" \\",
    "Donor: recipient's answer unchanged & " + " & ".join(c5(TD[r]["donor_unchanged"]) for r in ROLES5) + " & " + " & ".join(c5(TS[s]["donor_unchanged"]) for s in SPANS5) + r" \\",
    "Donor: donor's first operation & " + " & ".join("--" for r in ROLES5) + " & " + " & ".join(c5(TS[s]["donor_first_op"]) for s in SPANS5) + r" \\",
    "Noise: complete counterfactual & " + " & ".join(c5(TD[r]["noise"]) for r in ROLES5) + " & " + " & ".join(c5(TS[s]["noise"]) for s in SPANS5) + r" \\",
    "Noise: recipient's answer unchanged & " + " & ".join(c5(TD[r]["noise_unchanged"]) for r in ROLES5) + " & " + " & ".join(c5(TS[s]["noise_unchanged"]) for s in SPANS5) + r" \\",
    "Same leading digit, other value$^{a}$ & " + " & ".join(c5(TD[r]["same_leading"]) for r in ROLES5) + " & " + " & ".join(c5(TS[s]["sources"]["same_leading_different_value"]) for s in SPANS5) + r" \\",
    "Same intermediate, other decomposition$^{a}$ & " + " & ".join("--" for r in ROLES5) + " & " + " & ".join(c5(TS[s]["sources"]["same_z_decomposition"]) for s in SPANS5) + r" \\",
    "Same intermediate, other answer$^{a}$ & " + " & ".join("--" for r in ROLES5) + " & " + " & ".join(c5(TS[s]["sources"]["same_z_different_answer"]) for s in SPANS5) + r" \\",
    "Other intermediate, same answer$^{a,b}$ & " + " & ".join("--" for r in ROLES5) + " & " + " & ".join(c5(TS[s]["sources"]["different_z_same_answer"]) for s in SPANS5) + r" \\",
    "Unrelated problem$^{a,b}$ & " + " & ".join("--" for r in ROLES5) + " & " + " & ".join(c5(TS[s]["sources"]["random_source"]) for s in SPANS5) + r" \\",
    "Counterfactual more likely, teacher-forced & " + " & ".join(c5(TD[r]["gurnee"]["flips"]) if "gurnee" in TD[r] else "--" for r in ROLES5) + " & -- & --" + r" \\",
]
TABLE5SITES = rows


def kpc(k, n, rate=None):
    return f"{k}/{n} ({pc(k / n if rate is None else rate)})"


rows4 = [
    "Donor full problem & " + " & ".join(kpc(*TABLE4[cn]["donor"]) for cn in GATED) + r" & $\ge$\," + pc(TH["central"], 0) + r" \\",
    "Recipient full problem & " + " & ".join(kpc(*TABLE4[cn]["recipient"]) for cn in GATED) + r" & $\ge$\," + pc(TH["central"], 0) + r" \\",
    "Second step with the donor's intermediate given & " + " & ".join("{}/{}".format(*TABLE4[cn]["supplied"]) for cn in GATED)
    + r" & $\ge$\," + pc(TH["central"], 0) + r" \\",
    "First step alone, donor and recipient & " + " & ".join("{}/{} each".format(*TABLE4[cn]["first"]) for cn in GATED)
    + r" & $\ge$\," + pc(TH["first_step"], 0) + r" \\",
    "Pairs correct on all three central measures & " + " & ".join(kpc(*TABLE4[cn]["triple"]) for cn in GATED)
    + r" & $\ge$\," + pc(TH["triple"], 0) + r" \\",
    r"\addlinespace",
]
for i, label in enumerate(("multiplication", "addition", "subtraction")):
    rows4.append(f"Distinct problems wrong, {label} second step & "
                 + " & ".join("{}/{}".format(*TABLE4[cn]["profile"][i]) for cn in GATED) + r" & \\")

# appendix: interval scale for counts out of the 36 development pairs
m("AppZeroBound", pc(upper_bound(0, 36)), "exact one-sided bound for 0 of 36")
m("AppHalf", 18, "half of the 36 development pairs")
m("AppHalfCI", ci(18, 36), "exact interval for 18 of 36")

# ---------------------------------------------------------------------------------------------- tables
def zero_or_interval(r):
    return le(r["upper95_one_sided"]) if r["k"] == 0 else br(*r["exact95"])


def by_unit(r):
    """Units (programs, program groups or donors) with at least one success: exact interval, or bound for zero."""
    cl = r["clusters"]
    if r["k"] == 0:
        return f"0/{cl['count']} " + le(cl["upper95_one_sided_any_success"])
    return f"{cl['with_success']}/{cl['count']} " + br(*exact_interval(cl["with_success"], cl["count"]))


def transfer_row(group, label, r):
    return f"{group} & {label} & {frac(r['k'], r['n'])} & {zero_or_interval(r)} & {by_unit(r)} \\\\"


rows = []
rows.append(transfer_row(r"Held-out, Qwen3-0.6B", "Block 20, labelled boundary", s2["heldout/label_z-L20"]))
rows.append(transfer_row("", "Block 26, labelled boundary", s2["heldout/label_z-L26"]))
rows.append(r"\addlinespace")
rows.append(transfer_row("Held-out control panel", "Block 26, first literal", s2["controls/first_z-L26/activation"]))
rows.append(transfer_row("", "Block 26, labelled boundary", s2["controls/label_z-L26/activation"]))
rows.append(transfer_row("", "Value as text, first literal", s2["controls/first_z-L26/text_full"]))
rows.append(transfer_row("", "Value as text, labelled boundary", s2["controls/label_z-L26/text_full"]))
rows.append(r"\addlinespace")
rows.append(transfer_row("Readout, Qwen3-0.6B", "Block 26", ro["activation"]))
rows.append(transfer_row("", "First digit as text", ro["text_first_digit"]))
rows.append(transfer_row("", "Value as text", ro["text_full_value"]))
rows.append(r"\addlinespace")
names = {"Block 20, first operand": "Block 20, first operand", "Block 9, first operand": "Block 9, first operand",
         "Block 10, first operand": "Block 10, first operand", "Block 14, second operand": "Block 14, second operand",
         "Block 20, first-stage end": "Block 20, end of first step", "Block 23, answer prefix": "Block 23, answer prefix"}
for i, s in enumerate(WV):
    rows.append(transfer_row("Answer-only, Qwen3-4B" if i == 0 else "", names[s], fb["whole_vector/" + s]))
rows.append(transfer_row("", "Block 23, rank-8 projection", fb["rank8_projection/Block 23, answer prefix"]))
rows.append(transfer_row("", "Block 23, attention output", fb["attention_output/Block 23, answer prefix"]))
rows.append(transfer_row("", "Block 23, MLP output", fb["mlp_output/Block 23, answer prefix"]))
rows.append(transfer_row("", "Rank-8 diagnostic, donor", fb["diagnostic/donor/discriminating"]))
rows.append(transfer_row("", "Rank-8 diagnostic, same value", fb["diagnostic/same_value_coupled/discriminating"]))
TABLES = {"table-transfer.tex": rows, "table-study3.tex": TABLE3, "table-study4.tex": rows4, "table-study5.tex": TABLE5, "table-study5-sites.tex": TABLE5SITES}

SOURCE_LABELS = {"donor": "Donor", "same_z_decomposition": "Same intermediate, other decomposition",
                 "same_z_different_answer": "Same intermediate, other answer",
                 "different_z_same_answer": "Other intermediate, same answer",
                 "same_leading_different_value": "Same leading digit, other value",
                 "random_source": "Unrelated problem"}
rows = []
for cnd in SOURCES:
    rows.append(SOURCE_LABELS[cnd] + " & " + " & ".join(
        frac(OP[s]["sources"][cnd]["k"], OP[s]["sources"][cnd]["n"]) for s in FOUR) + r" \\")
rows.append(r"Controls (noise, unpatched, own vector) & " + " & ".join(
    frac(sum(OP[s]["controls"][c]["k"] for c in CONTROLS), sum(OP[s]["controls"][c]["n"] for c in CONTROLS))
    for s in FOUR) + r" \\")
TABLES["table-operand.tex"] = rows

# Appendix: Qwen3-4B whole-vector sites
F0 = {s: fresh[s][0] for s in WV}
F1 = {s: fresh[s][1] for s in WV}


def cells(fn):
    return " & ".join(fn(s) for s in WV)


rows = [
    r"Donor: complete counterfactual & " + cells(lambda s: frac(F0[s]["donor"]["counterfactual"], F0[s]["donor"]["n"])) + r" \\",
    r"Same intermediate, other decomposition & " + cells(lambda s: frac(F0[s]["same_z_decomposition"]["counterfactual"], 36)) + r" \\",
    r"Same intermediate, other answer & " + cells(lambda s: frac(F0[s]["same_z_different_answer"]["counterfactual"], 36)) + r" \\",
    r"Other intermediate, same answer$^{a}$ & " + cells(lambda s: frac(F0[s]["different_z_same_answer"]["counterfactual"], 36)) + r" \\",
    r"Same leading digit, other value$^{a}$ & " + cells(lambda s: frac(F0[s]["same_leading_different_value"]["counterfactual"], 36)) + r" \\",
    r"Matched noise & " + cells(lambda s: frac(F0[s]["isotropic_delta"]["counterfactual"], 36)) + r" \\",
    r"First digit as text & " + cells(lambda s: frac(F0[s]["text_first_answer_digit"]["counterfactual"], 36)) + r" \\",
    r"\addlinespace",
    r"Donor: program run on donor's first operand & " + cells(lambda s: frac(F0[s]["donor"]["first_operand_only"], 36)) + r" \\",
    r"Donor: donor's answer & " + cells(lambda s: frac(F0[s]["donor"]["copied_donor_answer"], 36)) + r" \\",
    r"Donor: recipient's answer unchanged & " + cells(lambda s: frac(F0[s]["donor"]["unchanged"], 36)) + r" \\",
    r"Matched noise: recipient's answer unchanged & " + cells(lambda s: frac(F0[s]["isotropic_delta"]["unchanged"], 36)) + r" \\",
    r"\addlinespace",
    r"Donor and both same-intermediate sources succeed & " + cells(lambda s: frac(F1[s]["joint_original_and_both_same_z"], F1[s]["pairs"])) + r" \\",
    r"Donor and same-leading-digit source agree & " + cells(lambda s: frac(F1[s]["same_leading_identical_outputs"], F1[s]["pairs"])) + r" \\",
    r"Donor output equal to first-digit text output$^{b}$ & " + cells(lambda s: frac(FD[("whole_vector", s)]["identical_to_text_edit"], FD[("whole_vector", s)]["pairs"])) + r" \\",
]
assert all(F0[s][c]["n"] == 36 for s in WV for c in F0[s])
TABLES["table-fourb-sites.tex"] = rows

# Appendix: other writes at block 23
SUB = R["ai02.subspace.conditions"]
ATT, MLP = R["ai02.component.attention"], R["ai02.component.mlp"]
assert all(v["n"] == 36 for v in SUB.values()) and all(v["n"] == 36 for v in ATT[0].values()) and all(v["n"] == 36 for v in MLP[0].values())
fd3 = [FD[("rank8_projection", "Block 23, answer prefix")], FD[("attention_output", "Block 23, answer prefix")],
       FD[("mlp_output", "Block 23, answer prefix")]]


def three(sub, comp_key, comp_field="counterfactual"):
    return " & ".join([str(sub), str(ATT[0][comp_key][comp_field]), str(MLP[0][comp_key][comp_field])])


rows = [
    r"Donor: complete counterfactual & " + three(SUB["donor"]["original_counterfactual"], "donor") + r" \\",
    r"Same intermediate, other decomposition & " + three(SUB["same_z_decomposition"]["original_counterfactual"], "same_z_decomposition") + r" \\",
    r"Same intermediate, other answer & " + three(SUB["same_z_different_answer"]["original_counterfactual"], "same_z_different_answer") + r" \\",
    r"Other intermediate, same answer$^{a}$ & " + three(SUB["different_z_same_answer"]["actual_source_counterfactual"], "different_z_same_answer") + r" \\",
    r"Same leading digit, other value$^{a}$ & " + three(SUB["same_leading_different_value"]["actual_source_counterfactual"], "same_leading_different_value") + r" \\",
    r"Matched noise & " + three(SUB["isotropic_delta"]["original_counterfactual"], "isotropic_delta") + r" \\",
    r"Random subspace of rank 8 & " + str(SUB["random_subspace"]["original_counterfactual"]) + r" & -- & -- \\",
    r"Basis fitted to shuffled labels & " + str(SUB["label_shuffled"]["original_counterfactual"]) + r" & -- & -- \\",
    r"\addlinespace",
    r"Donor: recipient's answer unchanged & " + three(SUB["donor"]["unchanged"], "donor", "unchanged") + r" \\",
    r"Matched noise: recipient's answer unchanged & " + three(SUB["isotropic_delta"]["unchanged"], "isotropic_delta", "unchanged") + r" \\",
    r"Donor and both same-intermediate sources succeed & " + " & ".join(
        [str(R["ai02.subspace.joint_original_and_both_same_z"]), str(ATT[1]["joint_original_and_both_same_z"]),
         str(MLP[1]["joint_original_and_both_same_z"])]) + r" \\",
    r"Donor and same-leading-digit source agree & " + " & ".join(
        [str(R["ai02.subspace.same_leading_identical_outputs"]), str(ATT[1]["same_leading_identical_outputs"]),
         str(MLP[1]["same_leading_identical_outputs"])]) + r" \\",
    r"Donor output equal to first-digit text output$^{b}$ & " + " & ".join(str(x["identical_to_text_edit"]) for x in fd3) + r" \\",
]
TABLES["table-fourb-variants.tex"] = rows

# Appendix: held-out experiment
mains = R["ai04.main.sites"]


def cat_row(label, d):
    return " & ".join([label, str(d["branches"]), str(d["target"]), str(d["recipient"]), str(d["other_valid"]),
                       str(d["invalid"]), str(d["exhausted"]), str(d["unavailable"])]) + r" \\"


rand = {b: [c[f"{b}-L26/random_{i}"] for i in range(3)] for b in ("first_z", "label_z")}
for b, rs in rand.items():
    assert all({k: v for k, v in x.items()} == rs[0] for x in rs), b                       # the three directions agree
rows = [r"\multicolumn{8}{@{}l}{\emph{Frozen sites, \HoPrograms{} held-out programs}} \\",
        cat_row("Block 20, labelled boundary", mains["label_z-L20"]),
        cat_row("Block 26, labelled boundary", mains["label_z-L26"]),
        r"\addlinespace",
        r"\multicolumn{8}{@{}l}{\emph{Control panel, block 26, first-literal boundary}} \\",
        cat_row("Activation write", c["first_z-L26/activation"]),
        cat_row("First digit as text", c["first_z-L26/text_digit"]),
        cat_row("Value as text", c["first_z-L26/text_full"]),
        cat_row("Random direction (each of three)", rand["first_z"][0]),
        r"\addlinespace",
        r"\multicolumn{8}{@{}l}{\emph{Control panel, block 26, labelled boundary}} \\",
        cat_row("Activation write", c["label_z-L26/activation"]),
        cat_row("First digit as text", c["label_z-L26/text_digit"]),
        cat_row("Value as text", c["label_z-L26/text_full"]),
        cat_row("Random direction (each of three)", rand["label_z"][0])]
TABLES["table-heldout.tex"] = rows

# Appendix: competence of the candidate configurations
Q = [("Qwen3-4B-Instruct-2507-FP8, instruction and three examples (used)",
      R["ai02.qualification.Qwen3-4B-FP8, instruction plus three demonstrations"]),
     ("Qwen3-4B-Instruct-2507-FP8, instruction only", R["ai02.qualification.Qwen3-4B-FP8, calculator instruction only"]),
     ("Qwen2.5-1.5B-Instruct, instruction only", R["ai02.qualification.Qwen2.5-1.5B, calculator instruction only"])]
rows = [" & ".join([label, frac(*q["donor_full"]), frac(*q["recipient_full"]), frac(*q["textual_counterfactual"]),
                    frac(*q["triple"])]) + r" \\" for label, q in Q]
q17 = R["ai02.qwen17b.initial"]
rows.append(" & ".join([r"Qwen3-1.7B, instruction only$^{a}$", frac(*q17["donor_full"]), frac(*q17["recipient_full"]),
                        frac(*q17["textual_counterfactual"]), "--"]) + r" \\")
TABLES["table-competence.tex"] = rows

# ---------------------------------------------------------------------------------------------- output
def render():
    lines = ["% Generated by paper/make_numbers.py from saved outputs. Do not edit."]
    for name in sorted(MACROS):
        value, source = MACROS[name]
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {source}")
    files = {"numbers.tex": "\n".join(lines) + "\n"}
    for name, rows in TABLES.items():
        files[name] = "% Generated by paper/make_numbers.py from saved outputs. Do not edit.\n" + "\n".join(rows) + "\n"
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = render()
    if args.check:
        stale = [n for n, t in files.items() if not (GEN / n).exists() or (GEN / n).read_text() != t]
        if stale:
            print("stale generated files: " + ", ".join(stale), file=sys.stderr)
            sys.exit(1)
        print(f"ok  {len(MACROS)} macros and {len(TABLES)} tables match the saved outputs")
        return
    GEN.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (GEN / n).write_text(t)
    print(f"wrote {len(MACROS)} macros to generated/numbers.tex and {len(TABLES)} tables")


if __name__ == "__main__":
    main()
